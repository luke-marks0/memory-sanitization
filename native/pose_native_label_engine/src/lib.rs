use std::cmp::Reverse;
use std::collections::BinaryHeap;
use std::mem::size_of;
use std::os::raw::c_char;
use std::thread;

use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict, PyModule};
use sha3::digest::{ExtendableOutput, Update, XofReader};
use sha3::Shake256;
use zeroize::Zeroize;

const DOMAIN_NAMESPACE: &[u8] = b"pose-db";
const ENCODING_VERSION_BYTES: [u8; 4] = 1u32.to_be_bytes();
const SOURCE_LABEL_DOMAIN: &str = "pose-db/label/source/v1";
const INTERNAL_LABEL_DOMAIN: &str = "pose-db/label/internal/v1";
const HOST_BLAKE3_PARALLEL_SLOT_THRESHOLD: usize = 2048;
const HOST_BLAKE3_PARALLEL_WORKER_CAP: usize = 8;

#[derive(Clone, Copy, Default)]
#[repr(C)]
struct CudaHbmProfileCounters {
    total_kernel_launches: u64,
    total_blocks_launched: u64,
    total_threads_launched: u64,
    launch_source_labels: u64,
    launch_internal1_copy: u64,
    launch_internal1_inplace_contiguous: u64,
    launch_internal1_inplace_indexed: u64,
    launch_internal2_inplace_contiguous: u64,
    launch_internal2_inplace_indexed: u64,
    launch_combine_buffers: u64,
    launch_connector_inplace_cooperative: u64,
    launch_connector_copy_cooperative: u64,
    launch_merged_center_ingress_cooperative: u64,
    cooperative_launch_attempts: u64,
    cooperative_launch_successes: u64,
    cooperative_launch_fallbacks: u64,
    device_to_device_copies: u64,
    device_to_device_copy_bytes: u64,
    device_synchronizes: u64,
    host_merged_plan_builds: u64,
    device_merged_plan_builds: u64,
    standalone_base_calls: u64,
    standalone_right_prefix_calls: u64,
    connected_full_calls: u64,
    connected_prefix_calls: u64,
    connector_from_inputs_calls: u64,
    connector_in_place_calls: u64,
    merged_center_ingress_calls: u64,
}

#[cfg(pose_cuda_hbm_available)]
unsafe extern "C" {
    fn pose_cuda_fill_challenge_labels_in_place_blake3(
        label_count_m: usize,
        graph_parameter_n: usize,
        output_bytes: usize,
        device: i32,
        session_seed: *const u8,
        session_seed_len: usize,
        graph_descriptor_digest: *const u8,
        graph_descriptor_digest_len: usize,
        target_pointer: *mut std::ffi::c_void,
        target_len: usize,
        scratch_peak_bytes_out: *mut u64,
        profile_out: *mut CudaHbmProfileCounters,
        error_buf: *mut c_char,
        error_buf_len: usize,
    ) -> i32;
}

#[derive(Clone, Copy)]
enum HashBackend {
    Blake3Xof,
    Shake256,
}

impl HashBackend {
    fn parse(value: &str) -> PyResult<Self> {
        match value {
            "blake3-xof" => Ok(Self::Blake3Xof),
            "shake256" => Ok(Self::Shake256),
            _ => Err(PyValueError::new_err(format!(
                "Unsupported native hash backend: {value:?}"
            ))),
        }
    }

    fn hash_into(self, payload: &[u8], out: &mut [u8]) {
        match self {
            Self::Blake3Xof => {
                let mut hasher = blake3::Hasher::new();
                hasher.update(payload);
                hasher.finalize_xof().fill(out);
            }
            Self::Shake256 => {
                let mut hasher = Shake256::default();
                hasher.update(payload);
                hasher.finalize_xof().read(out);
            }
        }
    }
}

struct NativeSpec {
    label_count_m: usize,
    graph_parameter_n: usize,
    output_bytes: usize,
    session_seed: Vec<u8>,
    graph_descriptor_digest: Vec<u8>,
    hash_backend: HashBackend,
}

impl NativeSpec {
    fn from_inputs(
        label_count_m: usize,
        graph_parameter_n: usize,
        hash_backend: &str,
        label_width_bits: usize,
        session_seed: &[u8],
        graph_descriptor_digest: &str,
    ) -> PyResult<Self> {
        if label_width_bits < 128 || (label_width_bits % 8) != 0 {
            return Err(PyValueError::new_err(format!(
                "label_width_bits must be byte-aligned and at least 128, got {label_width_bits}"
            )));
        }
        if label_count_m == 0 {
            return Err(PyValueError::new_err("label_count_m must be positive"));
        }
        Ok(Self {
            label_count_m,
            graph_parameter_n,
            output_bytes: label_width_bits / 8,
            session_seed: session_seed.to_vec(),
            graph_descriptor_digest: graph_descriptor_digest.as_bytes().to_vec(),
            hash_backend: HashBackend::parse(hash_backend)?,
        })
    }
}

struct GraphCounts {
    connector: Vec<usize>,
    standalone: Vec<usize>,
    connected: Vec<usize>,
}

impl GraphCounts {
    fn new(max_level: usize) -> Self {
        let mut connector = vec![0usize; max_level.max(1)];
        for dimension in 0..max_level {
            connector[dimension] = (dimension + 1) * (1usize << dimension);
        }
        let mut standalone = vec![0usize; max_level + 1];
        let mut connected = vec![0usize; max_level + 1];
        standalone[0] = 1;
        connected[0] = 1;
        for level in 1..=max_level {
            standalone[level] = standalone[level - 1] + connector[level - 1] + connected[level - 1];
            connected[level] = connected[level - 1] + (2 * connector[level - 1]) + connected[level - 1];
        }
        Self {
            connector,
            standalone,
            connected,
        }
    }

    fn connector_node_count(&self, dimension: usize) -> usize {
        self.connector[dimension]
    }
}

fn collect_connected_base_range(
    level: usize,
    start: usize,
    offset: usize,
    count: usize,
    counts: &GraphCounts,
    out: &mut Vec<usize>,
) {
    if count == 0 {
        return;
    }
    if level == 0 {
        out.push(start);
        return;
    }
    let half_width = 1usize << (level - 1);
    let left_take = count.min(half_width.saturating_sub(offset));
    if left_take > 0 {
        collect_connected_base_range(
            level - 1,
            start,
            offset,
            left_take,
            counts,
            out,
        );
    }
    let remaining = count - left_take;
    if remaining == 0 {
        return;
    }
    let right_start = start + counts.connected[level - 1] + (2 * counts.connector_node_count(level - 1));
    collect_connected_base_range(
        level - 1,
        right_start,
        offset.saturating_sub(half_width),
        remaining,
        counts,
        out,
    );
}

fn collect_standalone_base_range(
    level: usize,
    start: usize,
    offset: usize,
    count: usize,
    counts: &GraphCounts,
    out: &mut Vec<usize>,
) {
    if count == 0 {
        return;
    }
    if level == 0 {
        out.push(start);
        return;
    }
    let half_width = 1usize << (level - 1);
    let left_take = count.min(half_width.saturating_sub(offset));
    if left_take > 0 {
        collect_standalone_base_range(
            level - 1,
            start,
            offset,
            left_take,
            counts,
            out,
        );
    }
    let remaining = count - left_take;
    if remaining == 0 {
        return;
    }
    let right_start = start + counts.standalone[level - 1] + counts.connector_node_count(level - 1);
    collect_connected_base_range(
        level - 1,
        right_start,
        offset.saturating_sub(half_width),
        remaining,
        counts,
        out,
    );
}

fn compute_challenge_set(spec: &NativeSpec, counts: &GraphCounts) -> Vec<usize> {
    let level = spec.graph_parameter_n + 1;
    let left_width = 1usize << spec.graph_parameter_n;
    let retained_from_right = spec.label_count_m - left_width;
    let mut challenge_nodes = Vec::with_capacity(spec.label_count_m);
    collect_standalone_base_range(
        level,
        0,
        left_width,
        left_width,
        counts,
        &mut challenge_nodes,
    );
    let right_start = counts.standalone[level];
    collect_standalone_base_range(
        level,
        right_start,
        left_width,
        retained_from_right,
        counts,
        &mut challenge_nodes,
    );
    challenge_nodes
}

struct FormulaEmitter<'a, F: FnMut(u8, usize, usize)> {
    next_node_index: usize,
    consumer: &'a mut F,
    counts: &'a GraphCounts,
}

impl<'a, F: FnMut(u8, usize, usize)> FormulaEmitter<'a, F> {
    fn new(consumer: &'a mut F, counts: &'a GraphCounts) -> Self {
        Self {
            next_node_index: 0,
            consumer,
            counts,
        }
    }

    fn emit_row0(&mut self) -> usize {
        let node_index = self.next_node_index;
        self.next_node_index += 1;
        (self.consumer)(0, 0, 0);
        node_index
    }

    fn emit_row1(&mut self, predecessor: usize) -> usize {
        let node_index = self.next_node_index;
        self.next_node_index += 1;
        (self.consumer)(1, predecessor, 0);
        node_index
    }

    fn emit_row2(&mut self, first: usize, second: usize) -> usize {
        let node_index = self.next_node_index;
        self.next_node_index += 1;
        if first <= second {
            (self.consumer)(2, first, second);
        } else {
            (self.consumer)(2, second, first);
        }
        node_index
    }

    fn emit_connector(&mut self, dimension: usize, inputs: &[usize]) -> Vec<usize> {
        let width = 1usize << dimension;
        let mut previous_layer: Vec<usize> = inputs
            .iter()
            .copied()
            .map(|predecessor| self.emit_row1(predecessor))
            .collect();
        for layer_index in 0..dimension {
            let bit = 1usize << (dimension - 1 - layer_index);
            let mut current_layer = Vec::with_capacity(width);
            for offset in 0..width {
                current_layer.push(self.emit_row2(previous_layer[offset], previous_layer[offset ^ bit]));
            }
            previous_layer = current_layer;
        }
        previous_layer
    }

    fn release_local_successor(
        local_successor: usize,
        remaining_indegree: &mut [u8],
        ready_nodes: &mut BinaryHeap<Reverse<usize>>,
    ) {
        let updated_indegree = remaining_indegree[local_successor] - 1;
        remaining_indegree[local_successor] = updated_indegree;
        if updated_indegree == 0 {
            ready_nodes.push(Reverse(local_successor));
        }
    }

    fn emit_merged_center_ingress(
        &mut self,
        dimension: usize,
        primary_inputs: &[usize],
        ingress_inputs: &[usize],
    ) -> Vec<usize> {
        let width = 1usize << dimension;
        let center_node_count = self.counts.connector_node_count(dimension);
        let ingress_node_base = center_node_count;
        let total_local_nodes = center_node_count * 2;
        let mut remaining_indegree = vec![0u8; total_local_nodes];
        let mut global_ids = vec![0usize; total_local_nodes];

        for offset in 0..width {
            remaining_indegree[offset] = 1;
        }
        for layer in 1..=dimension {
            let layer_base = layer * width;
            let ingress_layer_base = ingress_node_base + layer_base;
            for offset in 0..width {
                remaining_indegree[layer_base + offset] = 2;
                remaining_indegree[ingress_layer_base + offset] = 2;
            }
        }

        let mut ready_nodes = BinaryHeap::new();
        for ingress_offset in ingress_node_base..(ingress_node_base + width) {
            ready_nodes.push(Reverse(ingress_offset));
        }

        while let Some(Reverse(local_node)) = ready_nodes.pop() {
            if local_node < ingress_node_base {
                let layer = local_node / width;
                let offset = local_node % width;
                if layer == 0 {
                    global_ids[local_node] = self.emit_row2(
                        primary_inputs[offset],
                        global_ids[ingress_node_base + (dimension * width) + offset],
                    );
                } else {
                    let previous_layer_base = (layer - 1) * width;
                    let bit = 1usize << (dimension - layer);
                    global_ids[local_node] = self.emit_row2(
                        global_ids[previous_layer_base + offset],
                        global_ids[previous_layer_base + (offset ^ bit)],
                    );
                }
                if layer < dimension {
                    let successor_layer_base = (layer + 1) * width;
                    let bit = 1usize << (dimension - 1 - layer);
                    Self::release_local_successor(
                        successor_layer_base + offset,
                        &mut remaining_indegree,
                        &mut ready_nodes,
                    );
                    Self::release_local_successor(
                        successor_layer_base + (offset ^ bit),
                        &mut remaining_indegree,
                        &mut ready_nodes,
                    );
                }
                continue;
            }

            let ingress_local = local_node - ingress_node_base;
            let layer = ingress_local / width;
            let offset = ingress_local % width;
            if layer == 0 {
                global_ids[local_node] = self.emit_row1(ingress_inputs[offset]);
            } else {
                let previous_layer_base = ingress_node_base + ((layer - 1) * width);
                let bit = 1usize << (dimension - layer);
                global_ids[local_node] = self.emit_row2(
                    global_ids[previous_layer_base + offset],
                    global_ids[previous_layer_base + (offset ^ bit)],
                );
            }
            if layer < dimension {
                let successor_layer_base = ingress_node_base + ((layer + 1) * width);
                let bit = 1usize << (dimension - 1 - layer);
                Self::release_local_successor(
                    successor_layer_base + offset,
                    &mut remaining_indegree,
                    &mut ready_nodes,
                );
                Self::release_local_successor(
                    successor_layer_base + (offset ^ bit),
                    &mut remaining_indegree,
                    &mut ready_nodes,
                );
            } else {
                Self::release_local_successor(offset, &mut remaining_indegree, &mut ready_nodes);
            }
        }

        let output_base = dimension * width;
        (0..width)
            .map(|offset| global_ids[output_base + offset])
            .collect()
    }

    fn emit_connected(&mut self, level: usize, inputs: &[usize]) -> Vec<usize> {
        if level == 0 {
            return vec![self.emit_row1(inputs[0])];
        }
        let half = inputs.len() / 2;
        let left_base = self.emit_connected(level - 1, &inputs[..half]);
        let center_outputs = self.emit_merged_center_ingress(level - 1, &left_base, &inputs[half..]);
        let right_base = self.emit_connected(level - 1, &center_outputs);
        [left_base, right_base].concat()
    }

    fn emit_standalone(&mut self, level: usize) -> Vec<usize> {
        if level == 0 {
            return vec![self.emit_row0()];
        }
        let left_base = self.emit_standalone(level - 1);
        let center_outputs = self.emit_connector(level - 1, &left_base);
        let right_base = self.emit_connected(level - 1, &center_outputs);
        [left_base, right_base].concat()
    }

    fn emit_graph(&mut self, level: usize) {
        let _ = self.emit_standalone(level);
        let _ = self.emit_standalone(level);
    }
}

fn encode_u32(value: usize) -> PyResult<[u8; 4]> {
    let value32 = u32::try_from(value).map_err(|_| {
        PyValueError::new_err(format!("Expected uint32-compatible value, got {value}"))
    })?;
    Ok(value32.to_be_bytes())
}

fn encode_u64(value: usize) -> PyResult<[u8; 8]> {
    let value64 = u64::try_from(value).map_err(|_| {
        PyValueError::new_err(format!("Expected uint64-compatible value, got {value}"))
    })?;
    Ok(value64.to_be_bytes())
}

fn length_prefix(length: usize) -> PyResult<[u8; 4]> {
    encode_u32(length)
}

fn domain_prefix(domain: &str, field_count: usize) -> PyResult<Vec<u8>> {
    let domain_bytes = domain.as_bytes();
    let mut payload = Vec::with_capacity(DOMAIN_NAMESPACE.len() + 4 + 4 + domain_bytes.len() + 4);
    payload.extend_from_slice(DOMAIN_NAMESPACE);
    payload.extend_from_slice(&ENCODING_VERSION_BYTES);
    payload.extend_from_slice(&length_prefix(domain_bytes.len())?);
    payload.extend_from_slice(domain_bytes);
    payload.extend_from_slice(&encode_u32(field_count)?);
    Ok(payload)
}

struct LabelOracle {
    hash_backend: HashBackend,
    output_bytes: usize,
    source_payload: Vec<u8>,
    source_node_index_offset: usize,
    source_blake3_base: Option<blake3::Hasher>,
    internal1_payload: Vec<u8>,
    internal1_node_index_offset: usize,
    internal1_label0_offset: usize,
    internal1_blake3_base: Option<blake3::Hasher>,
    internal2_payload: Vec<u8>,
    internal2_node_index_offset: usize,
    internal2_label0_offset: usize,
    internal2_label1_offset: usize,
    internal2_blake3_base: Option<blake3::Hasher>,
}

impl LabelOracle {
    fn new(spec: &NativeSpec) -> PyResult<Self> {
        let mut source_payload = Self::build_source_prefix(&spec.session_seed, &spec.graph_descriptor_digest)?;
        let source_node_index_offset = source_payload.len();
        source_payload.extend_from_slice(&[0u8; 8]);

        let (internal1_payload, internal1_node_index_offset, internal1_label0_offset) =
            Self::build_internal_payload(
                &spec.session_seed,
                &spec.graph_descriptor_digest,
                spec.output_bytes,
                1,
            )?;
        let (internal2_payload, internal2_node_index_offset, internal2_label0_offset, internal2_label1_offset) =
            Self::build_internal_payload2(
                &spec.session_seed,
                &spec.graph_descriptor_digest,
                spec.output_bytes,
            )?;
        let (source_blake3_base, internal1_blake3_base, internal2_blake3_base) =
            if matches!(spec.hash_backend, HashBackend::Blake3Xof) {
                (
                    Some(Self::build_blake3_base(&source_payload[..source_node_index_offset])),
                    Some(Self::build_blake3_base(
                        &internal1_payload[..internal1_node_index_offset],
                    )),
                    Some(Self::build_blake3_base(
                        &internal2_payload[..internal2_node_index_offset],
                    )),
                )
            } else {
                (None, None, None)
            };

        Ok(Self {
            hash_backend: spec.hash_backend,
            output_bytes: spec.output_bytes,
            source_payload,
            source_node_index_offset,
            source_blake3_base,
            internal1_payload,
            internal1_node_index_offset,
            internal1_label0_offset,
            internal1_blake3_base,
            internal2_payload,
            internal2_node_index_offset,
            internal2_label0_offset,
            internal2_label1_offset,
            internal2_blake3_base,
        })
    }

    fn accounted_bytes(&self) -> usize {
        self.source_payload.len() + self.internal1_payload.len() + self.internal2_payload.len()
    }

    fn build_source_prefix(session_seed: &[u8], graph_descriptor_digest: &[u8]) -> PyResult<Vec<u8>> {
        let mut payload = domain_prefix(SOURCE_LABEL_DOMAIN, 3)?;
        payload.extend_from_slice(&length_prefix(session_seed.len())?);
        payload.extend_from_slice(session_seed);
        payload.extend_from_slice(&length_prefix(graph_descriptor_digest.len())?);
        payload.extend_from_slice(graph_descriptor_digest);
        payload.extend_from_slice(&length_prefix(8)?);
        Ok(payload)
    }

    fn build_internal_prefix(
        session_seed: &[u8],
        graph_descriptor_digest: &[u8],
        predecessor_count: usize,
    ) -> PyResult<(Vec<u8>, [u8; 8])> {
        let mut payload = domain_prefix(INTERNAL_LABEL_DOMAIN, 4 + predecessor_count)?;
        payload.extend_from_slice(&length_prefix(session_seed.len())?);
        payload.extend_from_slice(session_seed);
        payload.extend_from_slice(&length_prefix(graph_descriptor_digest.len())?);
        payload.extend_from_slice(graph_descriptor_digest);
        payload.extend_from_slice(&length_prefix(8)?);
        let mut predecessor_count_field = [0u8; 8];
        predecessor_count_field[..4].copy_from_slice(&length_prefix(4)?);
        predecessor_count_field[4..].copy_from_slice(&encode_u32(predecessor_count)?);
        Ok((payload, predecessor_count_field))
    }

    fn build_internal_payload(
        session_seed: &[u8],
        graph_descriptor_digest: &[u8],
        output_bytes: usize,
        predecessor_count: usize,
    ) -> PyResult<(Vec<u8>, usize, usize)> {
        let (mut payload, predecessor_count_field) =
            Self::build_internal_prefix(session_seed, graph_descriptor_digest, predecessor_count)?;
        let node_index_offset = payload.len();
        payload.extend_from_slice(&[0u8; 8]);
        payload.extend_from_slice(&predecessor_count_field);
        payload.extend_from_slice(&length_prefix(output_bytes)?);
        let label0_offset = payload.len();
        payload.resize(label0_offset + output_bytes, 0);
        Ok((payload, node_index_offset, label0_offset))
    }

    fn build_internal_payload2(
        session_seed: &[u8],
        graph_descriptor_digest: &[u8],
        output_bytes: usize,
    ) -> PyResult<(Vec<u8>, usize, usize, usize)> {
        let (mut payload, predecessor_count_field) =
            Self::build_internal_prefix(session_seed, graph_descriptor_digest, 2)?;
        let node_index_offset = payload.len();
        payload.extend_from_slice(&[0u8; 8]);
        payload.extend_from_slice(&predecessor_count_field);
        payload.extend_from_slice(&length_prefix(output_bytes)?);
        let label0_offset = payload.len();
        payload.resize(label0_offset + output_bytes, 0);
        payload.extend_from_slice(&length_prefix(output_bytes)?);
        let label1_offset = payload.len();
        payload.resize(label1_offset + output_bytes, 0);
        Ok((payload, node_index_offset, label0_offset, label1_offset))
    }

    fn build_blake3_base(prefix: &[u8]) -> blake3::Hasher {
        let mut hasher = blake3::Hasher::new();
        hasher.update(prefix);
        hasher
    }

    fn blake3_hash_parts_into(
        base: &blake3::Hasher,
        output_bytes: usize,
        parts: &[&[u8]],
        out: &mut [u8],
    ) {
        debug_assert_eq!(out.len(), output_bytes);
        let mut hasher = base.clone();
        for part in parts {
            hasher.update(part);
        }
        if output_bytes <= blake3::OUT_LEN {
            let digest = hasher.finalize();
            out.copy_from_slice(&digest.as_bytes()[..output_bytes]);
        } else {
            hasher.finalize_xof().fill(out);
        }
    }

    fn pack_node_index(payload: &mut [u8], offset: usize, node_index: usize) -> PyResult<()> {
        payload[offset..offset + 8].copy_from_slice(&encode_u64(node_index)?);
        Ok(())
    }

    fn blake3_internal_label_2_into_shared(
        &self,
        node_index: usize,
        predecessor0: &[u8],
        predecessor1: &[u8],
        out: &mut [u8],
    ) {
        debug_assert!(matches!(self.hash_backend, HashBackend::Blake3Xof));
        let node_index_bytes = u64::try_from(node_index)
            .expect("node index must fit u64 for host blake3 labeling")
            .to_be_bytes();
        let base = self
            .internal2_blake3_base
            .as_ref()
            .expect("blake3 internal2 base must exist for blake3-xof");
        let static_suffix0 =
            &self.internal2_payload[self.internal2_node_index_offset + 8..self.internal2_label0_offset];
        let static_suffix1 =
            &self.internal2_payload[self.internal2_label0_offset + self.output_bytes..self.internal2_label1_offset];
        Self::blake3_hash_parts_into(
            base,
            self.output_bytes,
            &[
                &node_index_bytes,
                static_suffix0,
                predecessor0,
                static_suffix1,
                predecessor1,
            ],
            out,
        );
    }

    fn source_label_into(&mut self, node_index: usize, out: &mut [u8]) -> PyResult<()> {
        match self.hash_backend {
            HashBackend::Blake3Xof => {
                let node_index_bytes = encode_u64(node_index)?;
                let base = self
                    .source_blake3_base
                    .as_ref()
                    .expect("blake3 source base must exist for blake3-xof");
                Self::blake3_hash_parts_into(base, self.output_bytes, &[&node_index_bytes], out);
            }
            HashBackend::Shake256 => {
                Self::pack_node_index(&mut self.source_payload, self.source_node_index_offset, node_index)?;
                self.hash_backend.hash_into(&self.source_payload, out);
            }
        }
        Ok(())
    }

    fn internal_label_1_into(
        &mut self,
        node_index: usize,
        predecessor0: &[u8],
        out: &mut [u8],
    ) -> PyResult<()> {
        match self.hash_backend {
            HashBackend::Blake3Xof => {
                let node_index_bytes = encode_u64(node_index)?;
                let base = self
                    .internal1_blake3_base
                    .as_ref()
                    .expect("blake3 internal1 base must exist for blake3-xof");
                let static_suffix =
                    &self.internal1_payload[self.internal1_node_index_offset + 8..self.internal1_label0_offset];
                Self::blake3_hash_parts_into(
                    base,
                    self.output_bytes,
                    &[&node_index_bytes, static_suffix, predecessor0],
                    out,
                );
            }
            HashBackend::Shake256 => {
                Self::pack_node_index(
                    &mut self.internal1_payload,
                    self.internal1_node_index_offset,
                    node_index,
                )?;
                self.internal1_payload
                    [self.internal1_label0_offset..self.internal1_label0_offset + self.output_bytes]
                    .copy_from_slice(predecessor0);
                self.hash_backend.hash_into(&self.internal1_payload, out);
            }
        }
        Ok(())
    }

    fn internal_label_2_into(
        &mut self,
        node_index: usize,
        predecessor0: &[u8],
        predecessor1: &[u8],
        out: &mut [u8],
    ) -> PyResult<()> {
        match self.hash_backend {
            HashBackend::Blake3Xof => {
                let node_index_bytes = encode_u64(node_index)?;
                let base = self
                    .internal2_blake3_base
                    .as_ref()
                    .expect("blake3 internal2 base must exist for blake3-xof");
                let static_suffix0 =
                    &self.internal2_payload[self.internal2_node_index_offset + 8..self.internal2_label0_offset];
                let static_suffix1 =
                    &self.internal2_payload[self.internal2_label0_offset + self.output_bytes..self.internal2_label1_offset];
                Self::blake3_hash_parts_into(
                    base,
                    self.output_bytes,
                    &[
                        &node_index_bytes,
                        static_suffix0,
                        predecessor0,
                        static_suffix1,
                        predecessor1,
                    ],
                    out,
                );
            }
            HashBackend::Shake256 => {
                Self::pack_node_index(
                    &mut self.internal2_payload,
                    self.internal2_node_index_offset,
                    node_index,
                )?;
                self.internal2_payload
                    [self.internal2_label0_offset..self.internal2_label0_offset + self.output_bytes]
                    .copy_from_slice(predecessor0);
                self.internal2_payload
                    [self.internal2_label1_offset..self.internal2_label1_offset + self.output_bytes]
                    .copy_from_slice(predecessor1);
                self.hash_backend.hash_into(&self.internal2_payload, out);
            }
        }
        Ok(())
    }
}

#[derive(Clone, Copy)]
struct MergedIndexArithmetic {
    dimension: usize,
    width: usize,
}

impl MergedIndexArithmetic {
    fn new(dimension: usize) -> Self {
        Self {
            dimension,
            width: 1usize << dimension,
        }
    }

    fn width(&self) -> usize {
        self.width
    }

    fn ingress_index(&self, layer: usize, offset: usize) -> usize {
        debug_assert!(layer <= self.dimension);
        debug_assert!(offset < self.width);
        if layer < self.dimension {
            return (layer * self.width) + offset;
        }
        (self.dimension * self.width) + Self::final_ingress_displacement(self.dimension, offset)
    }

    fn center_index(&self, layer: usize, offset: usize) -> usize {
        debug_assert!(layer <= self.dimension);
        debug_assert!(offset < self.width);
        let remaining_dimension = self.dimension - layer;
        let block_width = 1usize << remaining_dimension;
        ((self.dimension + layer) * self.width)
            + (1usize << layer)
            + (offset / block_width)
            + ((1usize << layer) * Self::final_ingress_displacement(remaining_dimension, offset % block_width))
    }

    fn final_ingress_displacement(mut dimension: usize, mut offset: usize) -> usize {
        let mut displacement = 0usize;
        let mut scale = 1usize;
        while dimension > 0 {
            let half_width = 1usize << (dimension - 1);
            if offset < half_width {
                return displacement + (scale * (offset << 1));
            }
            displacement += scale * (1usize << dimension);
            scale <<= 1;
            offset -= half_width;
            dimension -= 1;
        }
        displacement
    }
}

#[cfg(test)]
struct MergedPlan {
    width: usize,
    ingress_indices: Vec<u32>,
    center_indices: Vec<u32>,
}

#[cfg(test)]
impl MergedPlan {
    fn build(dimension: usize) -> Self {
        let width = 1usize << dimension;
        let center_node_count = (dimension + 1) * width;
        let ingress_node_base = center_node_count;
        let total_local_nodes = center_node_count * 2;
        let mut remaining_indegree = vec![0u8; total_local_nodes];
        let mut ingress_indices = vec![0u32; (dimension + 1) * width];
        let mut center_indices = vec![0u32; (dimension + 1) * width];

        for offset in 0..width {
            remaining_indegree[offset] = 1;
        }
        for layer in 1..=dimension {
            let layer_base = layer * width;
            let ingress_layer_base = ingress_node_base + layer_base;
            for offset in 0..width {
                remaining_indegree[layer_base + offset] = 2;
                remaining_indegree[ingress_layer_base + offset] = 2;
            }
        }

        let mut ready_nodes = BinaryHeap::new();
        for local_node in ingress_node_base..(ingress_node_base + width) {
            ready_nodes.push(Reverse(local_node));
        }

        let mut next_index = 0u32;
        while let Some(Reverse(local_node)) = ready_nodes.pop() {
            if local_node < ingress_node_base {
                let layer = local_node / width;
                let offset = local_node % width;
                center_indices[(layer * width) + offset] = next_index;
                next_index += 1;
                if layer < dimension {
                    let successor_layer_base = (layer + 1) * width;
                    let bit = 1usize << (dimension - 1 - layer);
                    release_local_successor(
                        successor_layer_base + offset,
                        &mut remaining_indegree,
                        &mut ready_nodes,
                    );
                    release_local_successor(
                        successor_layer_base + (offset ^ bit),
                        &mut remaining_indegree,
                        &mut ready_nodes,
                    );
                }
                continue;
            }

            let ingress_local = local_node - ingress_node_base;
            let layer = ingress_local / width;
            let offset = ingress_local % width;
            ingress_indices[(layer * width) + offset] = next_index;
            next_index += 1;
            if layer < dimension {
                let successor_layer_base = ingress_node_base + ((layer + 1) * width);
                let bit = 1usize << (dimension - 1 - layer);
                release_local_successor(
                    successor_layer_base + offset,
                    &mut remaining_indegree,
                    &mut ready_nodes,
                );
                release_local_successor(
                    successor_layer_base + (offset ^ bit),
                    &mut remaining_indegree,
                    &mut ready_nodes,
                );
            } else {
                release_local_successor(offset, &mut remaining_indegree, &mut ready_nodes);
            }
        }

        Self {
            width,
            ingress_indices,
            center_indices,
        }
    }

    fn ingress_index(&self, layer: usize, offset: usize) -> usize {
        self.ingress_indices[(layer * self.width) + offset] as usize
    }

    fn center_index(&self, layer: usize, offset: usize) -> usize {
        self.center_indices[(layer * self.width) + offset] as usize
    }
}

#[cfg(test)]
fn release_local_successor(
    local_successor: usize,
    remaining_indegree: &mut [u8],
    ready_nodes: &mut BinaryHeap<Reverse<usize>>,
) {
    let updated_indegree = remaining_indegree[local_successor] - 1;
    remaining_indegree[local_successor] = updated_indegree;
    if updated_indegree == 0 {
        ready_nodes.push(Reverse(local_successor));
    }
}

struct InPlaceChallengeLabeler<'a> {
    spec: &'a NativeSpec,
    counts: &'a GraphCounts,
    oracle: LabelOracle,
    temp0: Vec<u8>,
    temp1: Vec<u8>,
    host_parallel_worker_budget: usize,
    peak_scratch_bytes: usize,
}

impl<'a> InPlaceChallengeLabeler<'a> {
    fn new(spec: &'a NativeSpec, counts: &'a GraphCounts) -> PyResult<Self> {
        let oracle = LabelOracle::new(spec)?;
        let temp0 = vec![0u8; spec.output_bytes];
        let temp1 = vec![0u8; spec.output_bytes];
        let host_parallel_worker_budget = thread::available_parallelism()
            .map(|count| count.get())
            .unwrap_or(1)
            .min(HOST_BLAKE3_PARALLEL_WORKER_CAP);
        let peak_scratch_bytes = oracle.accounted_bytes() + temp0.len() + temp1.len();
        Ok(Self {
            spec,
            counts,
            oracle,
            temp0,
            temp1,
            host_parallel_worker_budget,
            peak_scratch_bytes,
        })
    }

    fn slot_bytes(&self) -> usize {
        self.spec.output_bytes
    }

    fn slot_range_for(output_bytes: usize, slot_index: usize) -> std::ops::Range<usize> {
        let start = slot_index * output_bytes;
        start..(start + output_bytes)
    }

    fn slot<'b>(buffer: &'b [u8], output_bytes: usize, slot_index: usize) -> &'b [u8] {
        let range = Self::slot_range_for(output_bytes, slot_index);
        &buffer[range]
    }

    fn slot_mut<'b>(buffer: &'b mut [u8], output_bytes: usize, slot_index: usize) -> &'b mut [u8] {
        let range = Self::slot_range_for(output_bytes, slot_index);
        &mut buffer[range]
    }

    fn slot_count_for(buffer: &[u8], output_bytes: usize) -> usize {
        buffer.len() / output_bytes
    }

    fn blake3_parallel_worker_count(&self, width: usize, blocks: usize) -> usize {
        if !matches!(self.spec.hash_backend, HashBackend::Blake3Xof) {
            return 1;
        }
        if width < HOST_BLAKE3_PARALLEL_SLOT_THRESHOLD || blocks < 2 {
            return 1;
        }
        self.host_parallel_worker_budget.min(blocks)
    }

    fn try_parallel_blake3_pairwise_layer<F>(
        &self,
        buffer: &mut [u8],
        width: usize,
        bit: usize,
        index_for_slot: F,
    ) -> bool
    where
        F: Fn(usize) -> usize + Sync,
    {
        let output_bytes = self.slot_bytes();
        let block_slots = bit * 2;
        let blocks = width / block_slots;
        let worker_count = self.blake3_parallel_worker_count(width, blocks);
        if worker_count < 2 {
            return false;
        }

        let blocks_per_worker = blocks.div_ceil(worker_count);
        thread::scope(|scope| {
            let mut remainder = buffer;
            let mut next_block = 0usize;
            let mut handles = Vec::with_capacity(worker_count);
            for _ in 0..worker_count {
                let remaining_blocks = blocks.saturating_sub(next_block);
                if remaining_blocks == 0 {
                    break;
                }
                let worker_blocks = remaining_blocks.min(blocks_per_worker);
                let worker_bytes = worker_blocks * block_slots * output_bytes;
                let (chunk, rest) = remainder.split_at_mut(worker_bytes);
                let block_base_slot = next_block * block_slots;
                let oracle = &self.oracle;
                let index_for_slot = &index_for_slot;
                handles.push(scope.spawn(move || {
                    let mut temp0 = vec![0u8; output_bytes];
                    let mut temp1 = vec![0u8; output_bytes];
                    let chunk_block_bytes = block_slots * output_bytes;
                    for local_block in 0..worker_blocks {
                        let block_start = local_block * chunk_block_bytes;
                        let block_slice = &mut chunk[block_start..block_start + chunk_block_bytes];
                        let global_block_base_slot = block_base_slot + (local_block * block_slots);
                        for local in 0..bit {
                            let left_slot = local;
                            let right_slot = left_slot + bit;
                            let left_start = left_slot * output_bytes;
                            let right_start = right_slot * output_bytes;
                            temp0.copy_from_slice(&block_slice[left_start..left_start + output_bytes]);
                            temp1.copy_from_slice(&block_slice[right_start..right_start + output_bytes]);
                            oracle.blake3_internal_label_2_into_shared(
                                index_for_slot(global_block_base_slot + left_slot),
                                &temp0,
                                &temp1,
                                &mut block_slice[left_start..left_start + output_bytes],
                            );
                            oracle.blake3_internal_label_2_into_shared(
                                index_for_slot(global_block_base_slot + right_slot),
                                &temp0,
                                &temp1,
                                &mut block_slice[right_start..right_start + output_bytes],
                            );
                        }
                    }
                }));
                remainder = rest;
                next_block += worker_blocks;
            }
            for handle in handles {
                handle.join().expect("host blake3 worker must not panic");
            }
        });
        true
    }

    fn butterfly_layers_in_place_simple(
        &mut self,
        dimension: usize,
        buffer: &mut [u8],
        start_index: usize,
    ) -> PyResult<()> {
        let output_bytes = self.slot_bytes();
        let width = 1usize << dimension;
        for layer in 0..dimension {
            let bit = 1usize << (dimension - 1 - layer);
            let block = bit * 2;
            let layer_start_index = start_index + (layer * width);
            if self.try_parallel_blake3_pairwise_layer(
                buffer,
                width,
                bit,
                |slot| layer_start_index + slot,
            ) {
                continue;
            }
            for block_base in (0..width).step_by(block) {
                for local in 0..bit {
                    let left = block_base + local;
                    let right = left + bit;
                    self.temp0.copy_from_slice(Self::slot(buffer, output_bytes, left));
                    self.temp1.copy_from_slice(Self::slot(buffer, output_bytes, right));
                    {
                        let out_left = Self::slot_mut(buffer, output_bytes, left);
                        self.oracle.internal_label_2_into(
                            layer_start_index + left,
                            &self.temp0,
                            &self.temp1,
                            out_left,
                        )?;
                    }
                    {
                        let out_right = Self::slot_mut(buffer, output_bytes, right);
                        self.oracle.internal_label_2_into(
                            layer_start_index + right,
                            &self.temp0,
                            &self.temp1,
                            out_right,
                        )?;
                    }
                }
            }
        }
        Ok(())
    }

    fn connector_from_inputs_into_buffer(
        &mut self,
        dimension: usize,
        inputs: &[u8],
        outputs: &mut [u8],
        start_index: usize,
    ) -> PyResult<()> {
        let output_bytes = self.slot_bytes();
        let width = 1usize << dimension;
        for offset in 0..width {
            self.temp0.copy_from_slice(Self::slot(inputs, output_bytes, offset));
            let out_slot = Self::slot_mut(outputs, output_bytes, offset);
            self.oracle
                .internal_label_1_into(start_index + offset, &self.temp0, out_slot)?;
        }
        self.butterfly_layers_in_place_simple(dimension, outputs, start_index + width)
    }

    fn connector_in_place(
        &mut self,
        dimension: usize,
        buffer: &mut [u8],
        start_index: usize,
    ) -> PyResult<()> {
        let output_bytes = self.slot_bytes();
        let width = 1usize << dimension;
        for offset in 0..width {
            self.temp0.copy_from_slice(Self::slot(buffer, output_bytes, offset));
            let out_slot = Self::slot_mut(buffer, output_bytes, offset);
            self.oracle
                .internal_label_1_into(start_index + offset, &self.temp0, out_slot)?;
        }
        self.butterfly_layers_in_place_simple(dimension, buffer, start_index + width)
    }

    fn merged_center_ingress_in_place(
        &mut self,
        dimension: usize,
        primary_inputs: &[u8],
        ingress_workspace: &mut [u8],
        start_index: usize,
    ) -> PyResult<()> {
        let output_bytes = self.slot_bytes();
        let merged_indices = MergedIndexArithmetic::new(dimension);
        let width = merged_indices.width();

        for offset in 0..width {
            self.temp0.copy_from_slice(Self::slot(ingress_workspace, output_bytes, offset));
            let out_slot = Self::slot_mut(ingress_workspace, output_bytes, offset);
            let ingress_index = merged_indices.ingress_index(0, offset);
            self.oracle.internal_label_1_into(
                start_index + ingress_index,
                &self.temp0,
                out_slot,
            )?;
        }
        for layer in 1..=dimension {
            let bit = 1usize << (dimension - layer);
            let block = bit * 2;
            if self.try_parallel_blake3_pairwise_layer(
                ingress_workspace,
                width,
                bit,
                |slot| start_index + merged_indices.ingress_index(layer, slot),
            ) {
                continue;
            }
            for block_base in (0..width).step_by(block) {
                for local in 0..bit {
                    let left = block_base + local;
                    let right = left + bit;
                    self.temp0.copy_from_slice(Self::slot(ingress_workspace, output_bytes, left));
                    self.temp1.copy_from_slice(Self::slot(ingress_workspace, output_bytes, right));
                    {
                        let out_left = Self::slot_mut(ingress_workspace, output_bytes, left);
                        let ingress_index = merged_indices.ingress_index(layer, left);
                        self.oracle.internal_label_2_into(
                            start_index + ingress_index,
                            &self.temp0,
                            &self.temp1,
                            out_left,
                        )?;
                    }
                    {
                        let out_right = Self::slot_mut(ingress_workspace, output_bytes, right);
                        let ingress_index = merged_indices.ingress_index(layer, right);
                        self.oracle.internal_label_2_into(
                            start_index + ingress_index,
                            &self.temp0,
                            &self.temp1,
                            out_right,
                        )?;
                    }
                }
            }
        }

        for offset in 0..width {
            self.temp0.copy_from_slice(Self::slot(primary_inputs, output_bytes, offset));
            self.temp1.copy_from_slice(Self::slot(ingress_workspace, output_bytes, offset));
            let out_slot = Self::slot_mut(ingress_workspace, output_bytes, offset);
            let center_index = merged_indices.center_index(0, offset);
            self.oracle.internal_label_2_into(
                start_index + center_index,
                &self.temp0,
                &self.temp1,
                out_slot,
            )?;
        }
        for layer in 1..=dimension {
            let bit = 1usize << (dimension - layer);
            let block = bit * 2;
            if self.try_parallel_blake3_pairwise_layer(
                ingress_workspace,
                width,
                bit,
                |slot| start_index + merged_indices.center_index(layer, slot),
            ) {
                continue;
            }
            for block_base in (0..width).step_by(block) {
                for local in 0..bit {
                    let left = block_base + local;
                    let right = left + bit;
                    self.temp0.copy_from_slice(Self::slot(ingress_workspace, output_bytes, left));
                    self.temp1.copy_from_slice(Self::slot(ingress_workspace, output_bytes, right));
                    {
                        let out_left = Self::slot_mut(ingress_workspace, output_bytes, left);
                        let center_index = merged_indices.center_index(layer, left);
                        self.oracle.internal_label_2_into(
                            start_index + center_index,
                            &self.temp0,
                            &self.temp1,
                            out_left,
                        )?;
                    }
                    {
                        let out_right = Self::slot_mut(ingress_workspace, output_bytes, right);
                        let center_index = merged_indices.center_index(layer, right);
                        self.oracle.internal_label_2_into(
                            start_index + center_index,
                            &self.temp0,
                            &self.temp1,
                            out_right,
                        )?;
                    }
                }
            }
        }
        Ok(())
    }

    fn connected_full(
        &mut self,
        level: usize,
        buffer: &mut [u8],
        start_index: usize,
    ) -> PyResult<()> {
        if level == 0 {
            let output_bytes = self.slot_bytes();
            self.temp0.copy_from_slice(Self::slot(buffer, output_bytes, 0));
            let out_slot = Self::slot_mut(buffer, output_bytes, 0);
            self.oracle
                .internal_label_1_into(start_index, &self.temp0, out_slot)?;
            return Ok(());
        }
        let half_slots = 1usize << (level - 1);
        let split_at = half_slots * self.slot_bytes();
        let (left, right) = buffer.split_at_mut(split_at);
        self.connected_full(level - 1, left, start_index)?;
        let merged_start = start_index + self.counts.connected[level - 1];
        self.merged_center_ingress_in_place(level - 1, left, right, merged_start)?;
        let right_start = merged_start + (2 * self.counts.connector_node_count(level - 1));
        self.connected_full(level - 1, right, right_start)
    }

    fn connected_prefix(
        &mut self,
        level: usize,
        retain: usize,
        buffer: &mut [u8],
        start_index: usize,
    ) -> PyResult<()> {
        if retain == 0 {
            return Ok(());
        }
        if level == 0 {
            let output_bytes = self.slot_bytes();
            self.temp0.copy_from_slice(Self::slot(buffer, output_bytes, 0));
            let out_slot = Self::slot_mut(buffer, output_bytes, 0);
            self.oracle
                .internal_label_1_into(start_index, &self.temp0, out_slot)?;
            return Ok(());
        }
        let half_slots = 1usize << (level - 1);
        let split_at = half_slots * self.slot_bytes();
        let (left, right) = buffer.split_at_mut(split_at);
        if retain <= half_slots {
            return self.connected_prefix(level - 1, retain, left, start_index);
        }
        self.connected_full(level - 1, left, start_index)?;
        let merged_start = start_index + self.counts.connected[level - 1];
        self.merged_center_ingress_in_place(level - 1, left, right, merged_start)?;
        let right_start = merged_start + (2 * self.counts.connector_node_count(level - 1));
        self.connected_prefix(level - 1, retain - half_slots, right, right_start)
    }

    fn standalone_base(
        &mut self,
        level: usize,
        buffer: &mut [u8],
        start_index: usize,
    ) -> PyResult<()> {
        if level == 0 {
            let out_slot = Self::slot_mut(buffer, self.slot_bytes(), 0);
            self.oracle.source_label_into(start_index, out_slot)?;
            return Ok(());
        }
        let half_slots = 1usize << (level - 1);
        let split_at = half_slots * self.slot_bytes();
        let (left, right) = buffer.split_at_mut(split_at);
        self.standalone_base(level - 1, left, start_index)?;
        let connector_start = start_index + self.counts.standalone[level - 1];
        self.connector_from_inputs_into_buffer(level - 1, left, right, connector_start)?;
        let right_start = connector_start + self.counts.connector_node_count(level - 1);
        self.connected_full(level - 1, right, right_start)
    }

    fn standalone_right_prefix(
        &mut self,
        level: usize,
        retain: usize,
        workspace: &mut [u8],
        start_index: usize,
    ) -> PyResult<()> {
        if retain == 0 {
            return Ok(());
        }
        let half_slots = 1usize << (level - 1);
        if Self::slot_count_for(workspace, self.slot_bytes()) != half_slots {
            return Err(PyRuntimeError::new_err(
                "In-place standalone-right workspace width mismatch.",
            ));
        }
        self.standalone_base(level - 1, workspace, start_index)?;
        let connector_start = start_index + self.counts.standalone[level - 1];
        self.connector_in_place(level - 1, workspace, connector_start)?;
        let right_start = connector_start + self.counts.connector_node_count(level - 1);
        self.connected_prefix(level - 1, retain, workspace, right_start)
    }

    fn fill_challenge_buffer_in_place(&mut self, target: &mut [u8]) -> PyResult<()> {
        let level = self.spec.graph_parameter_n + 1;
        let left_width = 1usize << self.spec.graph_parameter_n;
        let retained_from_right = self.spec.label_count_m - left_width;
        let expected_bytes = self.spec.label_count_m * self.slot_bytes();
        if target.len() != expected_bytes {
            return Err(PyValueError::new_err(format!(
                "challenge buffer size mismatch: expected {expected_bytes} bytes, got {}",
                target.len()
            )));
        }
        let split_at = left_width * self.slot_bytes();
        let (left_workspace, right_output) = target.split_at_mut(split_at);
        if retained_from_right > 0 {
            let right_copy_start = self.counts.standalone[level];
            self.standalone_right_prefix(level, retained_from_right, left_workspace, right_copy_start)?;
            let right_bytes = retained_from_right * self.slot_bytes();
            right_output[..right_bytes].copy_from_slice(&left_workspace[..right_bytes]);
        }
        self.standalone_right_prefix(level, left_width, left_workspace, 0)
    }

    fn scratch_peak_bytes(&self) -> usize {
        self.peak_scratch_bytes
    }
}

struct ScratchStore {
    output_bytes: usize,
    slot_by_node: Vec<u32>,
    storage: Vec<u8>,
    free_slots: Vec<usize>,
}

impl ScratchStore {
    fn new(node_count: usize, output_bytes: usize) -> Self {
        Self {
            output_bytes,
            slot_by_node: vec![u32::MAX; node_count],
            storage: Vec::new(),
            free_slots: Vec::new(),
        }
    }

    fn bookkeeping_bytes(&self) -> usize {
        self.slot_by_node.len() * size_of::<u32>()
            + self.storage.len()
            + (self.free_slots.len() * size_of::<usize>())
    }

    fn retain(&mut self, node_index: usize, label: &[u8]) {
        let slot = self.free_slots.pop().unwrap_or_else(|| {
            let next_slot = self.storage.len() / self.output_bytes;
            self.storage.resize(self.storage.len() + self.output_bytes, 0);
            next_slot
        });
        let start = slot * self.output_bytes;
        self.storage[start..start + self.output_bytes].copy_from_slice(label);
        self.slot_by_node[node_index] = u32::try_from(slot).expect("scratch slot index exceeded u32 range");
    }

    fn load(&self, node_index: usize) -> Option<&[u8]> {
        let slot = self.slot_by_node[node_index];
        if slot == u32::MAX {
            return None;
        }
        let slot = slot as usize;
        let start = slot * self.output_bytes;
        Some(&self.storage[start..start + self.output_bytes])
    }

    fn release(&mut self, node_index: usize) {
        let slot = self.slot_by_node[node_index];
        if slot == u32::MAX {
            return;
        }
        let slot = slot as usize;
        let start = slot * self.output_bytes;
        self.storage[start..start + self.output_bytes].zeroize();
        self.slot_by_node[node_index] = u32::MAX;
        self.free_slots.push(slot);
    }
}

fn count_successors(spec: &NativeSpec, counts: &GraphCounts, node_count: usize) -> PyResult<Vec<u8>> {
    let mut successor_counts = vec![0u8; node_count];
    let mut error: Option<PyErr> = None;
    let mut consumer = |predecessor_count: u8, predecessor0: usize, predecessor1: usize| {
        if error.is_some() {
            return;
        }
        if predecessor_count >= 1 {
            let count0 = successor_counts[predecessor0];
            if count0 == u8::MAX {
                error = Some(PyRuntimeError::new_err(
                    "Graph successor count overflowed the compact native bookkeeping path.",
                ));
                return;
            }
            successor_counts[predecessor0] = count0 + 1;
        }
        if predecessor_count == 2 {
            let count1 = successor_counts[predecessor1];
            if count1 == u8::MAX {
                error = Some(PyRuntimeError::new_err(
                    "Graph successor count overflowed the compact native bookkeeping path.",
                ));
                return;
            }
            successor_counts[predecessor1] = count1 + 1;
        }
    };
    FormulaEmitter::new(&mut consumer, counts).emit_graph(spec.graph_parameter_n + 1);
    if let Some(error) = error {
        return Err(error);
    }
    Ok(successor_counts)
}

fn release_predecessor(successor_counts: &mut [u8], scratch: &mut ScratchStore, predecessor: usize) {
    let remaining = successor_counts[predecessor] - 1;
    successor_counts[predecessor] = remaining;
    if remaining == 0 {
        scratch.release(predecessor);
    }
}

fn compute_node_label_buffer_native(spec: &NativeSpec) -> PyResult<Vec<u8>> {
    let counts = GraphCounts::new(spec.graph_parameter_n + 1);
    let level = spec.graph_parameter_n + 1;
    let node_count = counts.standalone[level] * 2;
    let mut oracle = LabelOracle::new(spec)?;
    let mut labels = vec![0u8; node_count * spec.output_bytes];
    let mut node_index = 0usize;
    let mut error: Option<PyErr> = None;
    let mut consumer = |predecessor_count: u8, predecessor0: usize, predecessor1: usize| {
        if error.is_some() {
            return;
        }
        let start = node_index * spec.output_bytes;
        let (before, rest) = labels.split_at_mut(start);
        let out = &mut rest[..spec.output_bytes];
        let result = match predecessor_count {
            0 => oracle.source_label_into(node_index, out),
            1 => {
                let predecessor_start = predecessor0 * spec.output_bytes;
                oracle.internal_label_1_into(
                    node_index,
                    &before[predecessor_start..predecessor_start + spec.output_bytes],
                    out,
                )
            }
            2 => {
                let predecessor0_start = predecessor0 * spec.output_bytes;
                let predecessor1_start = predecessor1 * spec.output_bytes;
                oracle.internal_label_2_into(
                    node_index,
                    &before[predecessor0_start..predecessor0_start + spec.output_bytes],
                    &before[predecessor1_start..predecessor1_start + spec.output_bytes],
                    out,
                )
            }
            _ => Err(PyRuntimeError::new_err(
                "pose-db-drg-v1 predecessor arity exceeds the supported bound of 2.",
            )),
        };
        if let Err(runtime_error) = result {
            error = Some(runtime_error);
            return;
        }
        node_index += 1;
    };
    FormulaEmitter::new(&mut consumer, &counts).emit_graph(level);
    if let Some(error) = error {
        return Err(error);
    }
    Ok(labels)
}

fn compute_challenge_label_array_with_metrics(spec: &NativeSpec) -> PyResult<(Vec<u8>, usize)> {
    let counts = GraphCounts::new(spec.graph_parameter_n + 1);
    let mut challenge_buffer = vec![0u8; spec.label_count_m * spec.output_bytes];
    let mut labeler = InPlaceChallengeLabeler::new(spec, &counts)?;
    labeler.fill_challenge_buffer_in_place(&mut challenge_buffer)?;
    Ok((challenge_buffer, labeler.scratch_peak_bytes()))
}

fn fill_challenge_label_array_at_address_native(
    spec: &NativeSpec,
    target_address: usize,
    target_len: usize,
) -> PyResult<usize> {
    let counts = GraphCounts::new(spec.graph_parameter_n + 1);
    let expected_len = spec.label_count_m * spec.output_bytes;
    if target_len != expected_len {
        return Err(PyValueError::new_err(format!(
            "challenge buffer size mismatch: expected {expected_len} bytes, got {target_len}",
        )));
    }
    if target_address == 0 {
        return Err(PyValueError::new_err("target_address must be non-zero"));
    }
    let mut labeler = InPlaceChallengeLabeler::new(spec, &counts)?;
    let target = unsafe {
        std::slice::from_raw_parts_mut(target_address as *mut u8, target_len)
    };
    labeler.fill_challenge_buffer_in_place(target)?;
    Ok(labeler.scratch_peak_bytes())
}

#[cfg(pose_cuda_hbm_available)]
fn fill_challenge_label_array_on_gpu_profile_native(
    spec: &NativeSpec,
    device: i32,
    target_pointer: usize,
    target_len: usize,
) -> PyResult<(usize, CudaHbmProfileCounters)> {
    if !matches!(spec.hash_backend, HashBackend::Blake3Xof) {
        return Err(PyRuntimeError::new_err(
            "CUDA HBM in-place labeling currently supports only blake3-xof.",
        ));
    }
    if spec.output_bytes > 32 {
        return Err(PyRuntimeError::new_err(
            "CUDA HBM in-place labeling currently supports label widths up to 256 bits.",
        ));
    }
    let mut scratch_peak_bytes = 0u64;
    let mut profile_counters = CudaHbmProfileCounters::default();
    let mut error_buf = vec![0u8; 512];
    let status = unsafe {
        pose_cuda_fill_challenge_labels_in_place_blake3(
            spec.label_count_m,
            spec.graph_parameter_n,
            spec.output_bytes,
            device,
            spec.session_seed.as_ptr(),
            spec.session_seed.len(),
            spec.graph_descriptor_digest.as_ptr(),
            spec.graph_descriptor_digest.len(),
            target_pointer as *mut std::ffi::c_void,
            target_len,
            &mut scratch_peak_bytes as *mut u64,
            &mut profile_counters as *mut CudaHbmProfileCounters,
            error_buf.as_mut_ptr().cast::<c_char>(),
            error_buf.len(),
        )
    };
    if status != 0 {
        let nul = error_buf
            .iter()
            .position(|byte| *byte == 0)
            .unwrap_or(error_buf.len());
        let detail = String::from_utf8_lossy(&error_buf[..nul]).trim().to_string();
        let message = if detail.is_empty() {
            format!("CUDA HBM in-place labeling failed with status {status}.")
        } else {
            detail
        };
        return Err(PyRuntimeError::new_err(message));
    }
    let scratch_peak_bytes = usize::try_from(scratch_peak_bytes)
        .map_err(|_| PyRuntimeError::new_err("CUDA HBM scratch accounting overflowed usize."))?;
    Ok((scratch_peak_bytes, profile_counters))
}

#[cfg(not(pose_cuda_hbm_available))]
fn fill_challenge_label_array_on_gpu_profile_native(
    _spec: &NativeSpec,
    _device: i32,
    _target_pointer: usize,
    _target_len: usize,
) -> PyResult<(usize, CudaHbmProfileCounters)> {
    Err(PyRuntimeError::new_err(
        "CUDA HBM in-place label engine is unavailable in this build.",
    ))
}

fn fill_challenge_label_array_on_gpu_native(
    spec: &NativeSpec,
    device: i32,
    target_pointer: usize,
    target_len: usize,
) -> PyResult<usize> {
    fill_challenge_label_array_on_gpu_profile_native(spec, device, target_pointer, target_len)
        .map(|(scratch_peak_bytes, _profile_counters)| scratch_peak_bytes)
}

fn build_cuda_hbm_profile_payload(
    py: Python<'_>,
    scratch_peak_bytes: usize,
    counters: CudaHbmProfileCounters,
) -> PyResult<Py<PyDict>> {
    let payload = PyDict::new(py);
    payload.set_item("scratch_peak_bytes", scratch_peak_bytes)?;
    let profiling_counters = PyDict::new(py);
    for (key, value) in [
        ("total_kernel_launches", counters.total_kernel_launches),
        ("total_blocks_launched", counters.total_blocks_launched),
        ("total_threads_launched", counters.total_threads_launched),
        ("launch_source_labels", counters.launch_source_labels),
        ("launch_internal1_copy", counters.launch_internal1_copy),
        (
            "launch_internal1_inplace_contiguous",
            counters.launch_internal1_inplace_contiguous,
        ),
        (
            "launch_internal1_inplace_indexed",
            counters.launch_internal1_inplace_indexed,
        ),
        (
            "launch_internal2_inplace_contiguous",
            counters.launch_internal2_inplace_contiguous,
        ),
        (
            "launch_internal2_inplace_indexed",
            counters.launch_internal2_inplace_indexed,
        ),
        ("launch_combine_buffers", counters.launch_combine_buffers),
        (
            "launch_connector_inplace_cooperative",
            counters.launch_connector_inplace_cooperative,
        ),
        (
            "launch_connector_copy_cooperative",
            counters.launch_connector_copy_cooperative,
        ),
        (
            "launch_merged_center_ingress_cooperative",
            counters.launch_merged_center_ingress_cooperative,
        ),
        ("cooperative_launch_attempts", counters.cooperative_launch_attempts),
        ("cooperative_launch_successes", counters.cooperative_launch_successes),
        ("cooperative_launch_fallbacks", counters.cooperative_launch_fallbacks),
        ("device_to_device_copies", counters.device_to_device_copies),
        (
            "device_to_device_copy_bytes",
            counters.device_to_device_copy_bytes,
        ),
        ("device_synchronizes", counters.device_synchronizes),
        ("host_merged_plan_builds", counters.host_merged_plan_builds),
        ("device_merged_plan_builds", counters.device_merged_plan_builds),
        ("standalone_base_calls", counters.standalone_base_calls),
        (
            "standalone_right_prefix_calls",
            counters.standalone_right_prefix_calls,
        ),
        ("connected_full_calls", counters.connected_full_calls),
        ("connected_prefix_calls", counters.connected_prefix_calls),
        ("connector_from_inputs_calls", counters.connector_from_inputs_calls),
        ("connector_in_place_calls", counters.connector_in_place_calls),
        (
            "merged_center_ingress_calls",
            counters.merged_center_ingress_calls,
        ),
    ] {
        profiling_counters.set_item(key, value)?;
    }
    payload.set_item("profiling_counters", profiling_counters)?;
    Ok(payload.unbind())
}

fn stream_materialize_challenge_labels_native(
    py: Python<'_>,
    spec: &NativeSpec,
    writer: &Bound<'_, PyAny>,
) -> PyResult<usize> {
    let counts = GraphCounts::new(spec.graph_parameter_n + 1);
    let level = spec.graph_parameter_n + 1;
    let node_count = counts.standalone[level] * 2;
    let challenge_set = compute_challenge_set(spec, &counts);
    let mut successor_counts = count_successors(spec, &counts, node_count)?;
    let mut scratch = ScratchStore::new(node_count, spec.output_bytes);
    let mut oracle = LabelOracle::new(spec)?;
    let mut current_label = vec![0u8; spec.output_bytes];
    let base_accounted_bytes = successor_counts.len()
        + (challenge_set.len() * size_of::<usize>())
        + oracle.accounted_bytes();
    let mut peak_accounted_bytes = base_accounted_bytes + scratch.bookkeeping_bytes();
    let mut challenge_cursor = 0usize;
    let mut next_challenge_node = challenge_set.first().copied().unwrap_or(usize::MAX);
    let mut node_index = 0usize;
    let mut error: Option<PyErr> = None;

    let mut consumer = |predecessor_count: u8, predecessor0: usize, predecessor1: usize| {
        if error.is_some() {
            return;
        }
        let result = match predecessor_count {
            0 => oracle.source_label_into(node_index, &mut current_label),
            1 => match scratch.load(predecessor0) {
                Some(predecessor0_bytes) => {
                    oracle.internal_label_1_into(node_index, predecessor0_bytes, &mut current_label)
                }
                None => Err(PyRuntimeError::new_err(format!(
                    "Native scratch label for predecessor {predecessor0} was unavailable."
                ))),
            },
            2 => {
                let predecessor0_bytes = match scratch.load(predecessor0) {
                    Some(value) => value,
                    None => {
                        error = Some(PyRuntimeError::new_err(format!(
                            "Native scratch label for predecessor {predecessor0} was unavailable."
                        )));
                        return;
                    }
                };
                let predecessor1_bytes = match scratch.load(predecessor1) {
                    Some(value) => value,
                    None => {
                        error = Some(PyRuntimeError::new_err(format!(
                            "Native scratch label for predecessor {predecessor1} was unavailable."
                        )));
                        return;
                    }
                };
                oracle.internal_label_2_into(
                    node_index,
                    predecessor0_bytes,
                    predecessor1_bytes,
                    &mut current_label,
                )
            }
            _ => Err(PyRuntimeError::new_err(
                "pose-db-drg-v1 predecessor arity exceeds the supported bound of 2.",
            )),
        };
        if let Err(runtime_error) = result {
            error = Some(runtime_error);
            return;
        }

        if challenge_cursor < spec.label_count_m && node_index == next_challenge_node {
            match writer.call1((PyBytes::new(py, &current_label),)) {
                Ok(_) => {}
                Err(writer_error) => {
                    error = Some(writer_error);
                    return;
                }
            }
            challenge_cursor += 1;
            next_challenge_node = challenge_set
                .get(challenge_cursor)
                .copied()
                .unwrap_or(usize::MAX);
        }
        if successor_counts[node_index] > 0 {
            scratch.retain(node_index, &current_label);
            peak_accounted_bytes = peak_accounted_bytes.max(base_accounted_bytes + scratch.bookkeeping_bytes());
        }
        if predecessor_count >= 1 {
            release_predecessor(&mut successor_counts, &mut scratch, predecessor0);
        }
        if predecessor_count == 2 {
            release_predecessor(&mut successor_counts, &mut scratch, predecessor1);
        }
        node_index += 1;
    };
    FormulaEmitter::new(&mut consumer, &counts).emit_graph(level);
    current_label.zeroize();
    if let Some(error) = error {
        return Err(error);
    }
    Ok(peak_accounted_bytes)
}

#[pyfunction]
fn compute_node_label_buffer(
    py: Python<'_>,
    label_count_m: usize,
    graph_parameter_n: usize,
    hash_backend: &str,
    label_width_bits: usize,
    session_seed: &[u8],
    graph_descriptor_digest: &str,
) -> PyResult<Py<PyBytes>> {
    let spec = NativeSpec::from_inputs(
        label_count_m,
        graph_parameter_n,
        hash_backend,
        label_width_bits,
        session_seed,
        graph_descriptor_digest,
    )?;
    let labels = py.detach(|| compute_node_label_buffer_native(&spec))?;
    Ok(PyBytes::new(py, &labels).unbind())
}

#[pyfunction]
fn compute_challenge_label_array(
    py: Python<'_>,
    label_count_m: usize,
    graph_parameter_n: usize,
    hash_backend: &str,
    label_width_bits: usize,
    session_seed: &[u8],
    graph_descriptor_digest: &str,
) -> PyResult<Py<PyBytes>> {
    let spec = NativeSpec::from_inputs(
        label_count_m,
        graph_parameter_n,
        hash_backend,
        label_width_bits,
        session_seed,
        graph_descriptor_digest,
    )?;
    let (labels, _scratch_peak_bytes) = py.detach(|| compute_challenge_label_array_with_metrics(&spec))?;
    Ok(PyBytes::new(py, &labels).unbind())
}

#[pyfunction]
fn stream_materialize_challenge_labels(
    py: Python<'_>,
    label_count_m: usize,
    graph_parameter_n: usize,
    hash_backend: &str,
    label_width_bits: usize,
    session_seed: &[u8],
    graph_descriptor_digest: &str,
    writer: &Bound<'_, PyAny>,
) -> PyResult<usize> {
    let spec = NativeSpec::from_inputs(
        label_count_m,
        graph_parameter_n,
        hash_backend,
        label_width_bits,
        session_seed,
        graph_descriptor_digest,
    )?;
    stream_materialize_challenge_labels_native(py, &spec, writer)
}

#[pyfunction]
fn fill_challenge_label_array_at_address(
    py: Python<'_>,
    label_count_m: usize,
    graph_parameter_n: usize,
    hash_backend: &str,
    label_width_bits: usize,
    session_seed: &[u8],
    graph_descriptor_digest: &str,
    target_address: usize,
    target_len: usize,
) -> PyResult<usize> {
    let spec = NativeSpec::from_inputs(
        label_count_m,
        graph_parameter_n,
        hash_backend,
        label_width_bits,
        session_seed,
        graph_descriptor_digest,
    )?;
    py.detach(|| fill_challenge_label_array_at_address_native(&spec, target_address, target_len))
}

#[pyfunction]
fn cuda_hbm_in_place_available() -> bool {
    cfg!(pose_cuda_hbm_available)
}

#[pyfunction]
fn fill_challenge_label_array_on_gpu(
    py: Python<'_>,
    label_count_m: usize,
    graph_parameter_n: usize,
    hash_backend: &str,
    label_width_bits: usize,
    session_seed: &[u8],
    graph_descriptor_digest: &str,
    device: i32,
    target_pointer: usize,
    target_len: usize,
) -> PyResult<usize> {
    let spec = NativeSpec::from_inputs(
        label_count_m,
        graph_parameter_n,
        hash_backend,
        label_width_bits,
        session_seed,
        graph_descriptor_digest,
    )?;
    py.detach(|| fill_challenge_label_array_on_gpu_native(&spec, device, target_pointer, target_len))
}

#[pyfunction]
fn profile_challenge_label_array_on_gpu(
    py: Python<'_>,
    label_count_m: usize,
    graph_parameter_n: usize,
    hash_backend: &str,
    label_width_bits: usize,
    session_seed: &[u8],
    graph_descriptor_digest: &str,
    device: i32,
    target_pointer: usize,
    target_len: usize,
) -> PyResult<Py<PyDict>> {
    let spec = NativeSpec::from_inputs(
        label_count_m,
        graph_parameter_n,
        hash_backend,
        label_width_bits,
        session_seed,
        graph_descriptor_digest,
    )?;
    let (scratch_peak_bytes, counters) = py.detach(|| {
        fill_challenge_label_array_on_gpu_profile_native(&spec, device, target_pointer, target_len)
    })?;
    build_cuda_hbm_profile_payload(
        py,
        scratch_peak_bytes,
        counters,
    )
}

#[pyfunction]
fn native_engine_version() -> &'static str {
    env!("CARGO_PKG_VERSION")
}

#[cfg(test)]
mod tests {
    use super::{MergedIndexArithmetic, MergedPlan};

    #[test]
    fn arithmetic_indices_match_table_builder() {
        for dimension in 0..=15 {
            let width = 1usize << dimension;
            let arithmetic = MergedIndexArithmetic::new(dimension);
            let table = MergedPlan::build(dimension);
            for layer in 0..=dimension {
                for offset in 0..width {
                    assert_eq!(
                        arithmetic.ingress_index(layer, offset),
                        table.ingress_index(layer, offset),
                        "ingress mismatch at dimension={dimension} layer={layer} offset={offset}",
                    );
                    assert_eq!(
                        arithmetic.center_index(layer, offset),
                        table.center_index(layer, offset),
                        "center mismatch at dimension={dimension} layer={layer} offset={offset}",
                    );
                }
            }
        }
    }
}

#[pymodule]
fn pose_native_label_engine(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(compute_node_label_buffer, m)?)?;
    m.add_function(wrap_pyfunction!(compute_challenge_label_array, m)?)?;
    m.add_function(wrap_pyfunction!(stream_materialize_challenge_labels, m)?)?;
    m.add_function(wrap_pyfunction!(fill_challenge_label_array_at_address, m)?)?;
    m.add_function(wrap_pyfunction!(cuda_hbm_in_place_available, m)?)?;
    m.add_function(wrap_pyfunction!(fill_challenge_label_array_on_gpu, m)?)?;
    m.add_function(wrap_pyfunction!(profile_challenge_label_array_on_gpu, m)?)?;
    m.add_function(wrap_pyfunction!(native_engine_version, m)?)?;
    Ok(())
}

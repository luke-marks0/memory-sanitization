use std::cmp::Reverse;
use std::collections::BinaryHeap;
use std::mem::size_of;
use std::os::raw::c_char;

use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyModule};
use sha3::digest::{ExtendableOutput, Update, XofReader};
use sha3::Shake256;
use zeroize::Zeroize;

const DOMAIN_NAMESPACE: &[u8] = b"pose-db";
const ENCODING_VERSION_BYTES: [u8; 4] = 1u32.to_be_bytes();
const SOURCE_LABEL_DOMAIN: &str = "pose-db/label/source/v1";
const INTERNAL_LABEL_DOMAIN: &str = "pose-db/label/internal/v1";

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
    internal1_payload: Vec<u8>,
    internal1_node_index_offset: usize,
    internal1_label0_offset: usize,
    internal2_payload: Vec<u8>,
    internal2_node_index_offset: usize,
    internal2_label0_offset: usize,
    internal2_label1_offset: usize,
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

        Ok(Self {
            hash_backend: spec.hash_backend,
            output_bytes: spec.output_bytes,
            source_payload,
            source_node_index_offset,
            internal1_payload,
            internal1_node_index_offset,
            internal1_label0_offset,
            internal2_payload,
            internal2_node_index_offset,
            internal2_label0_offset,
            internal2_label1_offset,
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

    fn pack_node_index(payload: &mut [u8], offset: usize, node_index: usize) -> PyResult<()> {
        payload[offset..offset + 8].copy_from_slice(&encode_u64(node_index)?);
        Ok(())
    }

    fn source_label_into(&mut self, node_index: usize, out: &mut [u8]) -> PyResult<()> {
        Self::pack_node_index(&mut self.source_payload, self.source_node_index_offset, node_index)?;
        self.hash_backend.hash_into(&self.source_payload, out);
        Ok(())
    }

    fn internal_label_1_into(
        &mut self,
        node_index: usize,
        predecessor0: &[u8],
        out: &mut [u8],
    ) -> PyResult<()> {
        Self::pack_node_index(
            &mut self.internal1_payload,
            self.internal1_node_index_offset,
            node_index,
        )?;
        self.internal1_payload[self.internal1_label0_offset..self.internal1_label0_offset + self.output_bytes]
            .copy_from_slice(predecessor0);
        self.hash_backend.hash_into(&self.internal1_payload, out);
        Ok(())
    }

    fn internal_label_2_into(
        &mut self,
        node_index: usize,
        predecessor0: &[u8],
        predecessor1: &[u8],
        out: &mut [u8],
    ) -> PyResult<()> {
        Self::pack_node_index(
            &mut self.internal2_payload,
            self.internal2_node_index_offset,
            node_index,
        )?;
        self.internal2_payload[self.internal2_label0_offset..self.internal2_label0_offset + self.output_bytes]
            .copy_from_slice(predecessor0);
        self.internal2_payload[self.internal2_label1_offset..self.internal2_label1_offset + self.output_bytes]
            .copy_from_slice(predecessor1);
        self.hash_backend.hash_into(&self.internal2_payload, out);
        Ok(())
    }
}

struct MergedPlan {
    width: usize,
    ingress_indices: Vec<u32>,
    center_indices: Vec<u32>,
}

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

    fn accounted_bytes(&self) -> usize {
        self.ingress_indices.len() * size_of::<u32>() + self.center_indices.len() * size_of::<u32>()
    }
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

struct InPlaceChallengeLabeler<'a> {
    spec: &'a NativeSpec,
    counts: &'a GraphCounts,
    oracle: LabelOracle,
    merged_plans: Vec<MergedPlan>,
    temp0: Vec<u8>,
    temp1: Vec<u8>,
    peak_scratch_bytes: usize,
}

impl<'a> InPlaceChallengeLabeler<'a> {
    fn new(spec: &'a NativeSpec, counts: &'a GraphCounts) -> PyResult<Self> {
        let oracle = LabelOracle::new(spec)?;
        let merged_plans: Vec<MergedPlan> = (0..=spec.graph_parameter_n).map(MergedPlan::build).collect();
        let temp0 = vec![0u8; spec.output_bytes];
        let temp1 = vec![0u8; spec.output_bytes];
        let merged_plan_bytes: usize = merged_plans.iter().map(MergedPlan::accounted_bytes).sum();
        let peak_scratch_bytes = oracle.accounted_bytes() + temp0.len() + temp1.len() + merged_plan_bytes;
        Ok(Self {
            spec,
            counts,
            oracle,
            merged_plans,
            temp0,
            temp1,
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

    fn merged_plan(&self, dimension: usize) -> &MergedPlan {
        &self.merged_plans[dimension]
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
        let width = 1usize << dimension;

        for offset in 0..width {
            self.temp0.copy_from_slice(Self::slot(ingress_workspace, output_bytes, offset));
            let out_slot = Self::slot_mut(ingress_workspace, output_bytes, offset);
            let ingress_index = self.merged_plan(dimension).ingress_index(0, offset);
            self.oracle.internal_label_1_into(
                start_index + ingress_index,
                &self.temp0,
                out_slot,
            )?;
        }
        for layer in 1..=dimension {
            let bit = 1usize << (dimension - layer);
            let block = bit * 2;
            for block_base in (0..width).step_by(block) {
                for local in 0..bit {
                    let left = block_base + local;
                    let right = left + bit;
                    self.temp0.copy_from_slice(Self::slot(ingress_workspace, output_bytes, left));
                    self.temp1.copy_from_slice(Self::slot(ingress_workspace, output_bytes, right));
                    {
                        let out_left = Self::slot_mut(ingress_workspace, output_bytes, left);
                        let ingress_index = self.merged_plan(dimension).ingress_index(layer, left);
                        self.oracle.internal_label_2_into(
                            start_index + ingress_index,
                            &self.temp0,
                            &self.temp1,
                            out_left,
                        )?;
                    }
                    {
                        let out_right = Self::slot_mut(ingress_workspace, output_bytes, right);
                        let ingress_index = self.merged_plan(dimension).ingress_index(layer, right);
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
            let center_index = self.merged_plan(dimension).center_index(0, offset);
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
            for block_base in (0..width).step_by(block) {
                for local in 0..bit {
                    let left = block_base + local;
                    let right = left + bit;
                    self.temp0.copy_from_slice(Self::slot(ingress_workspace, output_bytes, left));
                    self.temp1.copy_from_slice(Self::slot(ingress_workspace, output_bytes, right));
                    {
                        let out_left = Self::slot_mut(ingress_workspace, output_bytes, left);
                        let center_index = self.merged_plan(dimension).center_index(layer, left);
                        self.oracle.internal_label_2_into(
                            start_index + center_index,
                            &self.temp0,
                            &self.temp1,
                            out_left,
                        )?;
                    }
                    {
                        let out_right = Self::slot_mut(ingress_workspace, output_bytes, right);
                        let center_index = self.merged_plan(dimension).center_index(layer, right);
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
fn fill_challenge_label_array_on_gpu_native(
    spec: &NativeSpec,
    device: i32,
    target_pointer: usize,
    target_len: usize,
) -> PyResult<usize> {
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
    usize::try_from(scratch_peak_bytes)
        .map_err(|_| PyRuntimeError::new_err("CUDA HBM scratch accounting overflowed usize."))
}

#[cfg(not(pose_cuda_hbm_available))]
fn fill_challenge_label_array_on_gpu_native(
    _spec: &NativeSpec,
    _device: i32,
    _target_pointer: usize,
    _target_len: usize,
) -> PyResult<usize> {
    Err(PyRuntimeError::new_err(
        "CUDA HBM in-place label engine is unavailable in this build.",
    ))
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
fn native_engine_version() -> &'static str {
    env!("CARGO_PKG_VERSION")
}

#[pymodule]
fn pose_native_label_engine(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(compute_node_label_buffer, m)?)?;
    m.add_function(wrap_pyfunction!(compute_challenge_label_array, m)?)?;
    m.add_function(wrap_pyfunction!(stream_materialize_challenge_labels, m)?)?;
    m.add_function(wrap_pyfunction!(fill_challenge_label_array_at_address, m)?)?;
    m.add_function(wrap_pyfunction!(cuda_hbm_in_place_available, m)?)?;
    m.add_function(wrap_pyfunction!(fill_challenge_label_array_on_gpu, m)?)?;
    m.add_function(wrap_pyfunction!(native_engine_version, m)?)?;
    Ok(())
}

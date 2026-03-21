use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::io::{Seek, Write};
use std::path::{Path, PathBuf};
use std::sync::{Mutex, Once, OnceLock};
use std::time::Instant;

use anyhow::{ensure, Context, Result};
use filecoin_proofs::{
    add_piece, generate_piece_commitment, seal_commit_phase1, seal_commit_phase2,
    seal_pre_commit_phase1, seal_pre_commit_phase2, verify_seal, Commitment, PieceInfo,
    PoRepConfig, ProverId, SectorShape2KiB, UnpaddedBytesAmount, SECTOR_SIZE_2_KIB,
};
use log::{Level, LevelFilter, Log, Metadata, Record};
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use serde::{Deserialize, Serialize};
use storage_proofs_core::api_version::ApiVersion;
use storage_proofs_core::sector::SectorId;
use tempfile::{tempdir, NamedTempFile};

const BRIDGE_STATUS: &str = "phase0-real-filecoin-bridge";
const REGISTERED_SEAL_PROOF_2KIB_V1_2: u64 = 5;
const DEFAULT_SECTOR_ID: u64 = 4242;
static BRIDGE_LOGGER: BridgeLogger = BridgeLogger;
static BRIDGE_LOGGER_INIT: Once = Once::new();
static BRIDGE_LOG_STATE: OnceLock<Mutex<BridgeLogState>> = OnceLock::new();
static BRIDGE_OPERATION_LOCK: OnceLock<Mutex<()>> = OnceLock::new();

#[derive(Debug, Serialize)]
struct BridgeStatus {
    status: &'static str,
    supports_real_filecoin_reference: bool,
    sector_size: u64,
    unpadded_piece_size: u64,
    api_version: &'static str,
    registered_seal_proof: u64,
    porep_id_hex: String,
    compiled_backends: Vec<&'static str>,
}

impl BridgeStatus {
    fn current() -> Self {
        Self {
            status: BRIDGE_STATUS,
            supports_real_filecoin_reference: true,
            sector_size: SECTOR_SIZE_2_KIB,
            unpadded_piece_size: default_config().unpadded_bytes_amount().0,
            api_version: "V1_2_0",
            registered_seal_proof: REGISTERED_SEAL_PROOF_2KIB_V1_2,
            porep_id_hex: hex::encode(default_porep_id()),
            compiled_backends: compiled_backends(),
        }
    }
}

#[derive(Debug, Deserialize)]
struct SealRequest {
    piece_bytes_hex: Option<String>,
    prover_id_hex: Option<String>,
    sector_id: Option<u64>,
    ticket_hex: Option<String>,
    seed_hex: Option<String>,
    porep_id_hex: Option<String>,
    verify_after_seal: Option<bool>,
}

#[derive(Debug, Serialize, Deserialize)]
struct SealArtifact {
    status: String,
    verified_after_seal: bool,
    sector_size: u64,
    api_version: String,
    registered_seal_proof: u64,
    porep_id_hex: String,
    prover_id_hex: String,
    sector_id: u64,
    ticket_hex: String,
    seed_hex: String,
    piece_size: u64,
    piece_commitment_hex: String,
    comm_d_hex: String,
    comm_r_hex: String,
    proof_hex: String,
    inner_timings_ms: BTreeMap<String, u64>,
    #[serde(default)]
    cpu_fallback_detected: bool,
    #[serde(default)]
    cpu_fallback_events: Vec<String>,
    #[serde(default)]
    extra_blobs_hex: BTreeMap<String, String>,
}

#[derive(Debug, Serialize)]
struct VerifyResult {
    status: &'static str,
    verified: bool,
    sector_id: u64,
    comm_d_hex: String,
    comm_r_hex: String,
}

#[derive(Default)]
struct BridgeLogState {
    active: bool,
    warnings: Vec<String>,
}

struct BridgeLogger;

impl Log for BridgeLogger {
    fn enabled(&self, metadata: &Metadata<'_>) -> bool {
        metadata.level() <= Level::Warn
    }

    fn log(&self, record: &Record<'_>) {
        if !self.enabled(record.metadata()) {
            return;
        }
        if let Ok(mut state) = bridge_log_state().lock() {
            if !state.active {
                return;
            }
            state
                .warnings
                .push(format!("[{}:{}] {}", record.level(), record.target(), record.args()));
        }
    }

    fn flush(&self) {}
}

fn default_porep_id() -> [u8; 32] {
    let mut porep_id = [0u8; 32];
    porep_id[..8].copy_from_slice(&REGISTERED_SEAL_PROOF_2KIB_V1_2.to_le_bytes());
    porep_id
}

fn default_config() -> PoRepConfig {
    PoRepConfig::new_groth16(SECTOR_SIZE_2_KIB, default_porep_id(), ApiVersion::V1_2_0)
}

fn default_piece_bytes(size: usize) -> Vec<u8> {
    (0..size).map(|index| (index % 251) as u8).collect()
}

fn default_bytes(fill: u8) -> [u8; 32] {
    [fill; 32]
}

fn compiled_backends() -> Vec<&'static str> {
    let mut backends = Vec::new();
    if cfg!(feature = "cuda") {
        backends.push("cuda");
    }
    if cfg!(feature = "opencl") {
        backends.push("opencl");
    }
    if backends.is_empty() {
        backends.push("cpu");
    }
    backends
}

fn bridge_log_state() -> &'static Mutex<BridgeLogState> {
    BRIDGE_LOG_STATE.get_or_init(|| Mutex::new(BridgeLogState::default()))
}

fn bridge_operation_lock() -> &'static Mutex<()> {
    BRIDGE_OPERATION_LOCK.get_or_init(|| Mutex::new(()))
}

fn ensure_bridge_logger() {
    BRIDGE_LOGGER_INIT.call_once(|| {
        let _ = log::set_logger(&BRIDGE_LOGGER);
        log::set_max_level(LevelFilter::Warn);
    });
}

fn capture_runtime_warnings<T>(operation: impl FnOnce() -> Result<T>) -> Result<(T, Vec<String>)> {
    ensure_bridge_logger();
    let _operation_guard = bridge_operation_lock()
        .lock()
        .expect("bridge operation lock poisoned");
    {
        let mut state = bridge_log_state()
            .lock()
            .expect("bridge log state lock poisoned");
        state.active = true;
        state.warnings.clear();
    }

    let result = operation();
    let warnings = {
        let mut state = bridge_log_state()
            .lock()
            .expect("bridge log state lock poisoned");
        state.active = false;
        std::mem::take(&mut state.warnings)
    };
    result.map(|value| (value, warnings))
}

fn has_compiled_gpu_backends() -> bool {
    compiled_backends().iter().any(|backend| *backend != "cpu")
}

fn bellman_no_gpu_forces_cpu() -> bool {
    has_compiled_gpu_backends()
        && std::env::var("BELLMAN_NO_GPU")
            .map(|value| value != "0")
            .unwrap_or(false)
}

fn extract_cpu_fallback_events(warnings: &[String]) -> Vec<String> {
    let mut events = BTreeSet::new();
    if bellman_no_gpu_forces_cpu() {
        events.insert(
            "BELLMAN_NO_GPU forced CPU execution despite compiled GPU backends.".to_string(),
        );
    }

    for warning in warnings {
        let normalized = warning.to_ascii_lowercase();
        if normalized.contains("falling back to cpu")
            || normalized.contains("no gpu found")
            || normalized.contains("cannot instantiate gpu fft kernel")
            || normalized.contains("cannot instantiate gpu multiexp kernel")
        {
            events.insert(warning.clone());
        }
    }
    events.into_iter().collect()
}

fn decode_hex_value(label: &str, value: &str) -> Result<Vec<u8>> {
    let normalized = value.trim_start_matches("0x");
    hex::decode(normalized).with_context(|| format!("invalid {label} hex value"))
}

fn decode_hex_32(label: &str, value: Option<&str>, default: [u8; 32]) -> Result<[u8; 32]> {
    match value {
        Some(value) => {
            let decoded = decode_hex_value(label, value)?;
            ensure!(decoded.len() == 32, "{label} must be exactly 32 bytes");
            let mut buffer = [0u8; 32];
            buffer.copy_from_slice(&decoded);
            Ok(buffer)
        }
        None => Ok(default),
    }
}

fn parse_piece_bytes(request: &SealRequest, config: &PoRepConfig) -> Result<Vec<u8>> {
    let expected_size = config.unpadded_bytes_amount().0 as usize;
    let piece_bytes = match request.piece_bytes_hex.as_deref() {
        Some(value) => decode_hex_value("piece_bytes_hex", value)?,
        None => default_piece_bytes(expected_size),
    };

    ensure!(
        piece_bytes.len() == expected_size,
        "piece_bytes_hex must be exactly {expected_size} bytes for a 2 KiB seal flow"
    );
    Ok(piece_bytes)
}

fn piece_info(piece_bytes: &[u8], config: &PoRepConfig) -> Result<(PieceInfo, NamedTempFile)> {
    let mut piece_file = NamedTempFile::new().context("failed to create piece file")?;
    piece_file
        .write_all(piece_bytes)
        .context("failed to write piece bytes")?;
    piece_file
        .as_file_mut()
        .sync_all()
        .context("failed to sync piece file")?;
    piece_file
        .as_file_mut()
        .rewind()
        .context("failed to rewind piece file")?;

    let piece_info = generate_piece_commitment(
        piece_file.as_file_mut(),
        config.unpadded_bytes_amount(),
    )
    .context("failed to generate piece commitment")?;
    piece_file
        .as_file_mut()
        .rewind()
        .context("failed to rewind piece file after commitment generation")?;
    Ok((piece_info, piece_file))
}

fn create_staged_sector(
    piece_file: &mut NamedTempFile,
    piece_size: UnpaddedBytesAmount,
) -> Result<NamedTempFile> {
    let mut staged_sector_file = NamedTempFile::new().context("failed to create staged sector file")?;
    add_piece(piece_file, &mut staged_sector_file, piece_size, &[])
        .context("failed to add piece to staged sector")?;
    Ok(staged_sector_file)
}

fn archive_entries(entries: &[(String, Vec<u8>)]) -> Vec<u8> {
    let mut encoded = Vec::new();
    for (path, payload) in entries {
        let path_bytes = path.as_bytes();
        encoded.extend_from_slice(&(path_bytes.len() as u32).to_be_bytes());
        encoded.extend_from_slice(path_bytes);
        encoded.extend_from_slice(&(payload.len() as u64).to_be_bytes());
        encoded.extend_from_slice(payload);
    }
    encoded
}

fn blob_kind_for_cache_path(relative_path: &Path) -> &'static str {
    let file_name = relative_path
        .file_name()
        .and_then(|value| value.to_str())
        .unwrap_or_default();
    if file_name == "p_aux" {
        "persistent_aux"
    } else if file_name == "t_aux" {
        "temporary_aux"
    } else if file_name.starts_with("tree-c") {
        "tree_c"
    } else if file_name.starts_with("tree-r-last") {
        "tree_r_last"
    } else if file_name.starts_with("layer-") {
        "labels"
    } else {
        "cache_file"
    }
}

fn collect_cache_files(root: &Path, path: &Path, output: &mut Vec<PathBuf>) -> Result<()> {
    for entry in fs::read_dir(path)
        .with_context(|| format!("failed to read cache directory {}", path.display()))?
    {
        let entry = entry.with_context(|| format!("failed to read entry under {}", path.display()))?;
        let entry_path = entry.path();
        let file_type = entry
            .file_type()
            .with_context(|| format!("failed to inspect {}", entry_path.display()))?;
        if file_type.is_dir() {
            collect_cache_files(root, &entry_path, output)?;
        } else if file_type.is_file() {
            output.push(
                entry_path
                    .strip_prefix(root)
                    .with_context(|| {
                        format!(
                            "cache file {} was not nested under {}",
                            entry_path.display(),
                            root.display()
                        )
                    })?
                    .to_path_buf(),
            );
        }
    }
    Ok(())
}

fn collect_extra_blobs(
    sealed_sector_path: &Path,
    cache_dir: &Path,
) -> Result<BTreeMap<String, String>> {
    let mut grouped: BTreeMap<String, Vec<(String, Vec<u8>)>> = BTreeMap::new();
    grouped.insert(
        "sealed_replica".to_string(),
        vec![(
            "sealed-sector".to_string(),
            fs::read(sealed_sector_path).with_context(|| {
                format!(
                    "failed to read sealed replica bytes from {}",
                    sealed_sector_path.display()
                )
            })?,
        )],
    );

    let mut relative_paths = Vec::new();
    collect_cache_files(cache_dir, cache_dir, &mut relative_paths)?;
    relative_paths.sort();

    for relative_path in relative_paths {
        let absolute_path = cache_dir.join(&relative_path);
        let payload = fs::read(&absolute_path)
            .with_context(|| format!("failed to read cache artifact {}", absolute_path.display()))?;
        grouped
            .entry(blob_kind_for_cache_path(&relative_path).to_string())
            .or_default()
            .push((relative_path.to_string_lossy().replace('\\', "/"), payload));
    }

    let mut encoded = BTreeMap::new();
    for (kind, entries) in grouped {
        let payload = if kind == "sealed_replica" {
            entries.into_iter().next().map(|(_, payload)| payload).unwrap_or_default()
        } else {
            archive_entries(&entries)
        };
        encoded.insert(kind, hex::encode(payload));
    }
    Ok(encoded)
}

fn seal(request: SealRequest) -> Result<SealArtifact> {
    let (mut artifact, warnings) = capture_runtime_warnings(|| {
        let porep_id = decode_hex_32(
            "porep_id_hex",
            request.porep_id_hex.as_deref(),
            default_porep_id(),
        )?;
        let config = PoRepConfig::new_groth16(SECTOR_SIZE_2_KIB, porep_id, ApiVersion::V1_2_0);
        let prover_id: ProverId = decode_hex_32(
            "prover_id_hex",
            request.prover_id_hex.as_deref(),
            default_bytes(7),
        )?;
        let sector_id_value = request.sector_id.unwrap_or(DEFAULT_SECTOR_ID);
        let sector_id = SectorId::from(sector_id_value);
        let ticket = decode_hex_32("ticket_hex", request.ticket_hex.as_deref(), default_bytes(1))?;
        let seed = decode_hex_32("seed_hex", request.seed_hex.as_deref(), default_bytes(2))?;
        let verify_after_seal = request.verify_after_seal.unwrap_or(true);
        let piece_bytes = parse_piece_bytes(&request, &config)?;
        let (piece_info, mut piece_file) = piece_info(&piece_bytes, &config)?;
        let piece_infos = vec![piece_info.clone()];
        let mut staged_sector_file =
            create_staged_sector(&mut piece_file, config.unpadded_bytes_amount())?;
        staged_sector_file
            .as_file_mut()
            .rewind()
            .context("failed to rewind staged sector file")?;

        let sealed_sector_file = NamedTempFile::new().context("failed to create sealed sector file")?;
        let cache_dir = tempdir().context("failed to create cache directory")?;
        let mut inner_timings_ms = BTreeMap::new();

        let pre_commit_phase1_started = Instant::now();
        let pre_commit_phase1 = seal_pre_commit_phase1::<_, _, _, SectorShape2KiB>(
            &config,
            cache_dir.path(),
            staged_sector_file.path(),
            sealed_sector_file.path(),
            prover_id,
            sector_id,
            ticket,
            &piece_infos,
        )
        .context("seal_pre_commit_phase1 failed")?;
        inner_timings_ms.insert(
            "seal_pre_commit_phase1".to_string(),
            pre_commit_phase1_started.elapsed().as_millis() as u64,
        );

        let pre_commit_phase2_started = Instant::now();
        let pre_commit = seal_pre_commit_phase2(
            &config,
            pre_commit_phase1,
            cache_dir.path(),
            sealed_sector_file.path(),
        )
        .context("seal_pre_commit_phase2 failed")?;
        inner_timings_ms.insert(
            "seal_pre_commit_phase2".to_string(),
            pre_commit_phase2_started.elapsed().as_millis() as u64,
        );

        let commit_phase1_started = Instant::now();
        let commit_phase1 = seal_commit_phase1::<_, SectorShape2KiB>(
            &config,
            cache_dir.path(),
            sealed_sector_file.path(),
            prover_id,
            sector_id,
            ticket,
            seed,
            pre_commit.clone(),
            &piece_infos,
        )
        .context("seal_commit_phase1 failed")?;
        inner_timings_ms.insert(
            "seal_commit_phase1".to_string(),
            commit_phase1_started.elapsed().as_millis() as u64,
        );

        let commit_phase2_started = Instant::now();
        let commit = seal_commit_phase2(&config, commit_phase1, prover_id, sector_id)
            .context("seal_commit_phase2 failed")?;
        inner_timings_ms.insert(
            "seal_commit_phase2".to_string(),
            commit_phase2_started.elapsed().as_millis() as u64,
        );

        let verified_after_seal = if verify_after_seal {
            let verify_started = Instant::now();
            verify_seal::<SectorShape2KiB>(
                &config,
                pre_commit.comm_r,
                pre_commit.comm_d,
                prover_id,
                sector_id,
                ticket,
                seed,
                &commit.proof,
            )
            .context("verify_seal failed")
            .inspect(|_| {
                inner_timings_ms.insert(
                    "verify_seal".to_string(),
                    verify_started.elapsed().as_millis() as u64,
                );
            })?
        } else {
            false
        };

        if verify_after_seal {
            ensure!(verified_after_seal, "the bridge produced a seal that did not verify");
        }

        let extra_blobs_hex = collect_extra_blobs(sealed_sector_file.path(), cache_dir.path())?;

        Ok(SealArtifact {
            status: BRIDGE_STATUS.to_string(),
            verified_after_seal,
            sector_size: SECTOR_SIZE_2_KIB,
            api_version: "V1_2_0".to_string(),
            registered_seal_proof: REGISTERED_SEAL_PROOF_2KIB_V1_2,
            porep_id_hex: hex::encode(porep_id),
            prover_id_hex: hex::encode(prover_id),
            sector_id: sector_id_value,
            ticket_hex: hex::encode(ticket),
            seed_hex: hex::encode(seed),
            piece_size: piece_info.size.0,
            piece_commitment_hex: hex::encode(piece_info.commitment),
            comm_d_hex: hex::encode(pre_commit.comm_d),
            comm_r_hex: hex::encode(pre_commit.comm_r),
            proof_hex: hex::encode(commit.proof),
            inner_timings_ms,
            cpu_fallback_detected: false,
            cpu_fallback_events: Vec::new(),
            extra_blobs_hex,
        })
    })?;

    artifact.cpu_fallback_events = extract_cpu_fallback_events(&warnings);
    artifact.cpu_fallback_detected = !artifact.cpu_fallback_events.is_empty();
    Ok(artifact)
}

fn verify(artifact: &SealArtifact) -> Result<VerifyResult> {
    let porep_id = decode_hex_32("porep_id_hex", Some(&artifact.porep_id_hex), default_porep_id())?;
    let config = PoRepConfig::new_groth16(SECTOR_SIZE_2_KIB, porep_id, ApiVersion::V1_2_0);
    let prover_id = decode_hex_32("prover_id_hex", Some(&artifact.prover_id_hex), default_bytes(7))?;
    let comm_d: Commitment = decode_hex_32("comm_d_hex", Some(&artifact.comm_d_hex), [0u8; 32])?;
    let comm_r: Commitment = decode_hex_32("comm_r_hex", Some(&artifact.comm_r_hex), [0u8; 32])?;
    let ticket = decode_hex_32("ticket_hex", Some(&artifact.ticket_hex), default_bytes(1))?;
    let seed = decode_hex_32("seed_hex", Some(&artifact.seed_hex), default_bytes(2))?;
    let proof = decode_hex_value("proof_hex", &artifact.proof_hex)?;
    let sector_id = SectorId::from(artifact.sector_id);

    let verified = verify_seal::<SectorShape2KiB>(
        &config, comm_r, comm_d, prover_id, sector_id, ticket, seed, &proof,
    )
    .context("verify_seal failed")?;

    Ok(VerifyResult {
        status: BRIDGE_STATUS,
        verified,
        sector_id: artifact.sector_id,
        comm_d_hex: artifact.comm_d_hex.clone(),
        comm_r_hex: artifact.comm_r_hex.clone(),
    })
}

fn to_py_error(error: anyhow::Error) -> PyErr {
    PyRuntimeError::new_err(format!("{error:#}"))
}

#[pyfunction]
fn bridge_status_json() -> PyResult<String> {
    serde_json::to_string(&BridgeStatus::current()).map_err(|error| to_py_error(error.into()))
}

#[pyfunction]
fn seal_json(request_json: &str) -> PyResult<String> {
    let request: SealRequest =
        serde_json::from_str(request_json).map_err(|error| to_py_error(error.into()))?;
    let artifact = seal(request).map_err(to_py_error)?;
    serde_json::to_string(&artifact).map_err(|error| to_py_error(error.into()))
}

#[pyfunction]
fn verify_json(artifact_json: &str) -> PyResult<String> {
    let artifact: SealArtifact =
        serde_json::from_str(artifact_json).map_err(|error| to_py_error(error.into()))?;
    let verification = verify(&artifact).map_err(to_py_error)?;
    serde_json::to_string(&verification).map_err(|error| to_py_error(error.into()))
}

#[pymodule]
fn pose_filecoin_bridge(_py: Python<'_>, module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(bridge_status_json, module)?)?;
    module.add_function(wrap_pyfunction!(seal_json, module)?)?;
    module.add_function(wrap_pyfunction!(verify_json, module)?)?;
    Ok(())
}

use std::collections::BTreeMap;
use std::io::{Seek, Write};
use std::time::Instant;

use anyhow::{ensure, Context, Result};
use filecoin_proofs::{
    add_piece, generate_piece_commitment, seal_commit_phase1, seal_commit_phase2,
    seal_pre_commit_phase1, seal_pre_commit_phase2, verify_seal, Commitment, PieceInfo,
    PoRepConfig, ProverId, SectorShape2KiB, UnpaddedBytesAmount, SECTOR_SIZE_2_KIB,
};
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use serde::{Deserialize, Serialize};
use storage_proofs_core::api_version::ApiVersion;
use storage_proofs_core::sector::SectorId;
use tempfile::{tempdir, NamedTempFile};

const BRIDGE_STATUS: &str = "phase0-real-filecoin-bridge";
const REGISTERED_SEAL_PROOF_2KIB_V1_2: u64 = 5;
const DEFAULT_SECTOR_ID: u64 = 4242;

#[derive(Debug, Serialize)]
struct BridgeStatus {
    status: &'static str,
    supports_real_filecoin_reference: bool,
    sector_size: u64,
    unpadded_piece_size: u64,
    api_version: &'static str,
    registered_seal_proof: u64,
    porep_id_hex: String,
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
}

#[derive(Debug, Serialize)]
struct VerifyResult {
    status: &'static str,
    verified: bool,
    sector_id: u64,
    comm_d_hex: String,
    comm_r_hex: String,
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

fn seal(request: SealRequest) -> Result<SealArtifact> {
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
    })
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

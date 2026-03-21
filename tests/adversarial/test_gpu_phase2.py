from __future__ import annotations

from dataclasses import dataclass

from pose.benchmarks.profiles import load_profile
from pose.common.hashing import sha256_hex
from pose.common.merkle import commit_payload
from pose.filecoin.reference import SealArtifact
from pose.protocol.messages import CleanupPolicy, LeaseRecord
from pose.protocol.region_payloads import RegionManifest
from pose.verifier import grpc_gpu_session as gpu_session_mod
from pose.verifier.grpc_gpu_session import run_gpu_session_via_grpc


class FakeReference:
    def seal_porep_unit(
        self,
        request=None,
        *,
        storage_profile: str = "minimal",
        leaf_alignment_bytes: int = 4096,
        extra_blobs=None,
    ):
        @dataclass(frozen=True)
        class DummyUnit:
            serialized_bytes: bytes

        return DummyUnit(serialized_bytes=b"\xaa" * leaf_alignment_bytes)

    def verify(self, artifact: SealArtifact) -> bool:
        return artifact.comm_r_hex == ("33" * 32)


@dataclass
class FakeProcess:
    terminated: bool = False
    waited: bool = False

    def terminate(self) -> None:
        self.terminated = True

    def wait(self, timeout: int | None = None) -> None:
        self.waited = True


@dataclass
class FakeGpuLease:
    record: LeaseRecord
    buffer: bytearray
    closed: bool = False

    def read(self, length: int | None = None, offset: int = 0) -> bytes:
        requested = len(self.buffer) if length is None else length
        return bytes(self.buffer[offset : offset + requested])

    def close(self) -> None:
        self.closed = True


def test_hbm_region_not_actually_filled_is_rejected(
    monkeypatch,
) -> None:
    profile = load_profile("single-h100-hbm-max")
    fake_process = FakeProcess()
    created_leases: list[FakeGpuLease] = []

    artifact = SealArtifact(
        status="phase0-real-filecoin-bridge",
        verified_after_seal=True,
        sector_size=2048,
        api_version="V1_2_0",
        registered_seal_proof=5,
        porep_id_hex="05" + ("00" * 31),
        prover_id_hex="07" * 32,
        sector_id=4242,
        ticket_hex="01" * 32,
        seed_hex="02" * 32,
        piece_size=2032,
        piece_commitment_hex="11" * 32,
        comm_d_hex="22" * 32,
        comm_r_hex="33" * 32,
        proof_hex="aabbccddeeff",
        inner_timings_ms={
            "seal_pre_commit_phase1": 17,
            "seal_pre_commit_phase2": 19,
            "seal_commit_phase1": 23,
            "seal_commit_phase2": 29,
            "verify_seal": 31,
        },
    )

    def fake_create_gpu_lease(
        *,
        session_id: str,
        region_id: str,
        device: int,
        usable_bytes: int,
        cleanup_policy: CleanupPolicy,
        lease_duration_ms: int,
        runtime=None,
    ) -> FakeGpuLease:
        lease = FakeGpuLease(
            record=LeaseRecord(
                region_id=region_id,
                region_type="gpu",
                usable_bytes=usable_bytes,
                lease_handle="cuda-ipc:0:ZmFrZQ==",
                lease_expiry="2099-01-01T00:00:00+00:00",
                cleanup_policy=cleanup_policy,
            ),
            buffer=bytearray(usable_bytes),
        )
        created_leases.append(lease)
        return lease

    def fake_materialize_region_payloads(socket_path: str, session_id: str):
        expected_payload = b"\xaa" * profile.leaf_size
        commitment = commit_payload(expected_payload, profile.leaf_size)
        manifest = RegionManifest(
            region_id="gpu-0",
            region_type="gpu",
            usable_bytes=profile.leaf_size,
            leaf_size=profile.leaf_size,
            payload_length_bytes=profile.leaf_size,
            real_porep_bytes=profile.leaf_size,
            tail_filler_bytes=0,
            unit_count=1,
            unit_digests_hex=("00" * 32,),
            payload_sha256_hex=sha256_hex(expected_payload),
            merkle_root_hex=commitment.root_hex,
        )
        return (
            {"gpu-0": (manifest, manifest.manifest_root_hex)},
            {
                "copy_to_hbm": 1,
                "copy_to_host": 0,
                "object_serialization": 1,
                "outer_tree_build": 1,
            },
        )

    def fake_challenge_outer(
        socket_path: str,
        *,
        session_id: str,
        region_id: str,
        session_manifest_root: str,
        challenge_indices: list[int],
    ):
        zero_payload = bytes(profile.leaf_size)
        commitment = commit_payload(zero_payload, profile.leaf_size)
        openings = []
        for index in challenge_indices:
            opening = commitment.opening(
                index,
                zero_payload[index * profile.leaf_size : (index + 1) * profile.leaf_size],
            )
            openings.append(
                {
                    "region_id": region_id,
                    "session_manifest_root": session_manifest_root,
                    "leaf_index": index,
                    "leaf_hex": opening.leaf.hex(),
                    "sibling_hashes_hex": [item.hex() for item in opening.sibling_hashes],
                }
            )
        return openings, 0

    monkeypatch.setattr(gpu_session_mod, "VendoredFilecoinReference", FakeReference)
    monkeypatch.setattr(gpu_session_mod, "create_gpu_lease", fake_create_gpu_lease)
    monkeypatch.setattr(
        gpu_session_mod,
        "detect_gpu_memory_bytes",
        lambda device: (2 * profile.leaf_size, 2 * profile.leaf_size),
    )
    monkeypatch.setattr(
        gpu_session_mod,
        "start_ephemeral_prover_server",
        lambda socket_path: fake_process,
    )
    monkeypatch.setattr(gpu_session_mod, "plan_session", lambda *args, **kwargs: None)
    monkeypatch.setattr(gpu_session_mod, "lease_regions", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        gpu_session_mod,
        "generate_inner_porep",
        lambda *args, **kwargs: {"gpu-0": [artifact]},
    )
    monkeypatch.setattr(gpu_session_mod, "materialize_region_payloads", fake_materialize_region_payloads)
    monkeypatch.setattr(gpu_session_mod, "commit_regions", lambda *args, **kwargs: None)
    monkeypatch.setattr(gpu_session_mod, "verify_inner_proofs", lambda *args, **kwargs: None)
    monkeypatch.setattr(gpu_session_mod, "challenge_outer", fake_challenge_outer)
    monkeypatch.setattr(gpu_session_mod, "verify_outer", lambda *args, **kwargs: None)
    monkeypatch.setattr(gpu_session_mod, "finalize_session", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        gpu_session_mod,
        "cleanup_session",
        lambda *args, **kwargs: "ZEROIZED_AND_RELEASED",
    )

    result = run_gpu_session_via_grpc(profile)

    assert result.success is False
    assert result.verdict == "OUTER_PROOF_INVALID"
    assert result.inner_filecoin_verified is True
    assert result.outer_pose_verified is False
    assert created_leases and created_leases[0].closed is True
    assert fake_process.terminated is True
    assert fake_process.waited is True

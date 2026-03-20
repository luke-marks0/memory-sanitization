from __future__ import annotations

KiB = 1024
MiB = 1024 * KiB
GiB = 1024 * MiB


def format_bytes(value: int) -> str:
    thresholds = (
        (GiB, "GiB"),
        (MiB, "MiB"),
        (KiB, "KiB"),
    )
    for divisor, suffix in thresholds:
        if value >= divisor:
            return f"{value / divisor:.2f} {suffix}"
    return f"{value} B"


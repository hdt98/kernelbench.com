"""Hardware peak-throughput lookup tables for KernelBench-AMD."""
from src.hardware.mi325x import MI325X
from src.hardware.mi350x import MI350X

TARGETS = {
    "MI325X": MI325X,
    "MI350X": MI350X,
}


def get(name: str):
    if name not in TARGETS:
        raise ValueError(f"Unknown hardware {name!r}; available: {list(TARGETS)}")
    return TARGETS[name]

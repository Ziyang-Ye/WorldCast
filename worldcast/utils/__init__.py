"""Numerics switches, seeding, weight files, file digests, fingerprints and lazy package exports.

The names below are imported on first use (PEP 562).
"""

from .lazy import lazy_exports

__getattr__, __dir__, __all__ = lazy_exports(
    __name__,
    {
        "files": ("sha256_file",),
        "fingerprints": ("fingerprint", "float32_array", "sha256_float32"),
        "lazy": ("lazy_exports",),
        "precision": (
            "GENERATOR_DTYPE",
            "HALF_DTYPES",
            "cast_floating_tensors",
            "enable_tf32",
            "fp32_island",
            "generator_autocast",
            "generator_dtype",
        ),
        "seed": ("set_seed",),
        "weights": (
            "check_state_dict",
            "indexed_count",
            "is_safetensors",
            "leading_size",
            "module_from_state",
            "read_state_dict",
        ),
    },
)

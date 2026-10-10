"""Inference: the client, the shared world state the clients exchange, and the decoders.

:class:`Client` (and :func:`run_client`, the offline client) with its :class:`ServingOptions` and
:class:`BlockControls`; the shared world state (:class:`WorldState`) on a directory
(:class:`DirectoryWorldState`), usable without a client; the decoders of a client's latents
(:func:`decode_to_mp4`, :class:`WanFrameDecoder`); the clients of a round as processes
(:func:`rounds_of`, :func:`run_round`) and their comparison with the reference runs
(:func:`verify_client`, :func:`verify_latents`). The names are imported on first use (PEP 562), so
the serving options are read without torch.
"""

from worldcast.utils.lazy import lazy_exports

__getattr__, __dir__, __all__ = lazy_exports(
    __name__,
    {
        "client": (
            "BlockRecord",
            "Client",
            "Frame",
            "StepControls",
            "client_world_state",
            "run_client",
        ),
        "controls": ("BlockControls",),
        "decode": (
            "DEFAULT_DECODE_CHUNK",
            "MP4_WRITER_KWARGS",
            "WanFrameDecoder",
            "decode_frames",
            "decode_pixels",
            "decode_to_mp4",
            "load_latents",
            "mp4_decoded_frames",
            "pixels_to_uint8_frames",
            "stream_decode_pixels",
            "write_mp4",
        ),
        "directory": ("DirectoryWorldState",),
        "fast": ("FastGenerator", "patchify_linear"),
        "loading": (
            "ClientModels",
            "client_latents",
            "load_closed_loop",
            "load_window",
        ),
        "memory_blocks": ("BlockAtFn", "MemoryBlocks"),
        "reference": (
            "Check",
            "ClientCheck",
            "ReferenceRun",
            "check_inputs",
            "entry_noise_fingerprint",
            "environment",
            "first_frame_fingerprint",
            "latent_fingerprints",
            "read_reference_runs",
            "verify_client",
            "verify_latents",
        ),
        "rollout": ("BlockRead", "PreparedBlock", "Rollout"),
        "serving": ("DECODERS", "ServingOptions"),
        "session": (
            "CLIENT_ENV",
            "CLIENT_MEMORY_GIB",
            "MAX_ROUND_SECONDS",
            "ROUND_STEP_SECONDS",
            "check_gpu_memory",
            "gpu_memory",
            "place_clients",
            "round_latents",
            "round_name",
            "rounds_of",
            "run_round",
            "visible_gpus",
        ),
        "world_state": (
            "DONE_FAILED",
            "DONE_OK",
            "BlockKey",
            "ClientFailedError",
            "LockstepTimeoutError",
            "PublishedBlock",
            "StepRecord",
            "WaitStats",
            "WorldState",
            "WorldStateError",
        ),
    },
)

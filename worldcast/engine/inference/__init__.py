"""Inference: the client, the shared world state the clients exchange, and the decoders.

:class:`Client` (and :func:`run_client`, the offline client) with its :class:`ServingOptions` and
:class:`BlockControls` (:class:`LiveBlockControls` from a player's live input, which a
:class:`LiveRollout` plays, and :class:`IdleContext` keeps its context while the player is
idle), and the rows its player state field read (:class:`FieldRows`, for the live demo's page);
the shared world state (:class:`WorldState`) on a directory (:class:`DirectoryWorldState`) or over
messages (:class:`MessageWorldState`, :class:`MessagePart`), usable without a client; the decoders
of a client's latents (:func:`decode_to_mp4`, :class:`WanFrameDecoder`,
:class:`StreamingTinyDecoder`); a recorded client played as a live player
(:class:`RecordedControls`, :func:`replay_latents`); the clients of a round as processes
(:func:`rounds_of`, :func:`run_round`) and their comparison with the reference runs
(:func:`verify_client`, :func:`verify_latents`). The names are imported on first use (PEP 562): the
demo's coordinator reads the serving options without torch.
"""

from worldcast.utils.lazy import lazy_exports

__getattr__, __dir__, __all__ = lazy_exports(
    __name__,
    {
        "client": (
            "BlockRecord",
            "Client",
            "Frame",
            "RefreshRecord",
            "StepControls",
            "Switch",
            "client_world_state",
            "run_client",
        ),
        "controls": ("BlockControls", "LiveBlockControls"),
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
        "field_rows": ("FieldRows",),
        "idle_context": (
            "CLIP_BLOCKS",
            "SAME_VIEW_DEGREES",
            "SAME_VIEW_U",
            "IdleContext",
            "is_idle",
            "same_view",
        ),
        "live_rollout": (
            "LIVE_ROUND_LATENTS",
            "PAGE_BLOCKS",
            "BlockCut",
            "LiveRollout",
            "live_round_latents",
        ),
        "loading": (
            "ClientModels",
            "client_latents",
            "load_closed_loop",
            "load_live_loop",
            "load_window",
        ),
        "memory_blocks": ("BlockAtFn", "LiveMemoryBlocks", "MemoryBlocks"),
        "messages": ("LIVE_STATES_PER_ENTRY", "MessagePart", "MessageWorldState"),
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
        "replay": ("RecordedControls", "replay_latents"),
        "replicas": ("GeneratorReplica", "card_device", "rng_state", "set_rng_state"),
        "rollout": ("BlockRead", "PreparedBlock", "Rollout", "context_noise"),
        "serving": ("DECODERS", "LOW_LATENCY_ATTENTION", "ServingOptions"),
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
        "taehv": ("FRAMES_TO_TRIM", "StreamingTinyDecoder", "TinyDecoder", "load_taew2_2"),
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

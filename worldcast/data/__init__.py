"""Recorded rounds: indices, tick tables, controls, labels, the latent frames' time grid, the latent
cache, a client's window and the memory frames.

The training windows are :mod:`worldcast.data.training`, drawn by :mod:`worldcast.data.stream`; the
choice of their memory frames is :mod:`worldcast.data.memory_selection`. The names below are
imported on first use (PEP 562): the recordings, the controls and the latent frames' constants load
without torch.
"""

from worldcast.utils.lazy import lazy_exports

__getattr__, __dir__, __all__ = lazy_exports(
    __name__,
    {
        "camera": ("c2w_from_state_rows", "camera_tans", "half_angle_tangents", "window_cameras"),
        "controls": (
            "CAMERA_DELTA_SCALE",
            "CONTROL_BUTTONS",
            "DEFAULT_CAMERA_ENCODING",
            "OPENCS2_WEAPONS",
            "STATE_MODEL_CONTROL_DIM",
            "STATE_MODEL_SUBSTEPS",
            "align_ticks_to_ordered_substeps",
            "align_ticks_to_video_frames",
            "encode_turn",
            "encode_weapon",
            "normalize_weapon_name",
            "quantize_camera_delta",
            "state_model_controls",
        ),
        "game": ("EYE_HEIGHT", "HFOV_DEGREES", "TICK_RATE"),
        "labels": (
            "OBSERVER_SIGNAL_KEYS",
            "SCOPE_LEVELS",
            "load_observer_signals",
            "visibility_rows",
        ),
        "latent_cache": ("load_first_latent", "load_window_latents"),
        "latents": (
            "BLOCK",
            "FIRST_TARGET",
            "FPS",
            "FRAME_SIZE",
            "FRAME_TOKENS",
            "LATENT_CHANNELS",
            "LATENT_GRID",
            "LATENT_HEIGHT",
            "LATENT_SHAPE",
            "LATENT_WIDTH",
            "RECENT",
            "SOURCE_FRAMES_PER_LATENT",
            "SOURCE_FRAMES_PER_VIDEO_FRAME",
            "TOKEN_GRID",
            "VIDEO_FRAMES_PER_BLOCK",
            "VIDEO_FRAMES_PER_LATENT",
            "block_span",
            "last_video_frames",
            "video_frame_count",
            "video_frames_of",
            "window_key",
        ),
        "map_mesh": ("MeshLibrary",),
        "memory_frames": (
            "MEMORY_FRAMES_PREFIX",
            "MemoryBlock",
            "MemoryFrames",
            "batch_keys",
            "block_memory_frames",
            "recorded_block",
        ),
        "memory_selection": ("BIDIRECTIONAL", "BLOCK_CAUSAL", "MemoryFrameSource"),
        "recordings": (
            "ALIVE_INDEX",
            "NUM_PLAYERS",
            "MediaIndex",
            "MediaRecord",
            "RoundIndexRow",
            "TickTable",
            "load_round_index_row",
            "read_player_ticks",
            "read_round_index",
        ),
        "stream": ("ResumableDataStream",),
        "training": (
            "RAW_VIDEO_MAX_TICK_GAP_SECONDS",
            "BucketWindow",
            "BucketWindows",
            "RawVideoWindows",
            "read_bucket_index",
            "read_media_exclusion",
        ),
        "window": (
            "ClientWindow",
            "DataPaths",
            "PlayerFrames",
            "WindowRefused",
            "WindowSpec",
            "collate_windows",
            "covered_frames",
            "load_client_window",
        ),
    },
)

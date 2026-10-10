"""Player states: the player state field (Sec. 3.2) and the closed loop (Sec. 3.4).

The field needs a batch's player states and a camera, not a client:
:func:`player_state_conditions` packs the states and :func:`player_state_field` projects them into
the camera, gated by the client's visibility labels of the batch (:func:`visible_latent_frames`):
the recording's GT labels, or the ones :class:`PredictedVisibility` writes from the depth head;
:func:`field_builder` is the callable the generator takes. :class:`ClosedLoop` is one client's
predicted states: its state model in the block loop (:class:`StateReader`), the complementary filter
of Eq. (4) and the extrapolation within a block.

The names below are imported on first use (PEP 562).
"""

from worldcast.utils.lazy import lazy_exports

__getattr__, __dir__, __all__ = lazy_exports(
    __name__,
    {
        "closed_loop": ("ClosedLoop", "ComplementaryFilter", "PositionExchange", "StateReader"),
        "extrapolation": ("PhysicsPrior", "PositionTrack", "extrapolate"),
        "field": (
            "PlayerStateFieldConfig",
            "compose",
            "field_builder",
            "player_state_field",
            "splat",
        ),
        "predicted_visibility": ("PredictedVisibility", "block_depth_frames"),
        "projection": ("project_players", "project_view"),
        "states": (
            "PlayerState",
            "camera_turns",
            "continuous_row_state",
            "integrate_camera_angles",
            "latent_frame_rows",
            "memory_continuous_columns",
            "pack_substeps",
            "player_state_conditions",
            "wrap_degrees",
        ),
        "visibility": ("latent_visibility", "live_eligibility", "visible_latent_frames"),
    },
)

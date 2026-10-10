"""Training (stages 1-4): the step loop every stage shares, its recipes and their losses.

:func:`build_trainer` builds the trainer of a config's stage: :class:`BidirectionalTrainer`
(stages 1-2) or :class:`TeacherForcingTrainer` (stage 3), the two recipes of
:class:`FlowMatchingTrainer`, or :class:`DistillationTrainer` (stage 4); :class:`Trainer` is the
loop they share. The weighted flow-matching loss of Eq. (5) and the distribution matching losses
are functions of tensors (:func:`flow_matching_loss`, :func:`compose_weight`, ...). The names are
imported on first use (PEP 562).
"""

from worldcast.utils.lazy import lazy_exports

__getattr__, __dir__, __all__ = lazy_exports(
    __name__,
    {
        "build": (
            "build_data_stream",
            "build_dataset",
            "build_distillation_parts",
            "build_score_model",
            "build_trainer",
            "build_trainer_parts",
            "build_training_generator",
            "initial_generator_state",
            "start_run",
            "wrap",
        ),
        "losses": (
            "BETA_MAX",
            "BETA_RADIUS",
            "CONTEXT_BAND",
            "FOREGROUND_LAMBDA",
            "FOREGROUND_MIN_SIGMA",
            "SCORE_TIMESTEP_RANGE",
            "FlowMatchingSample",
            "apply_frame_loss_mask",
            "compose_weight",
            "context_band_index_range",
            "critic_loss",
            "distribution_matching_gradient",
            "distribution_matching_loss",
            "distribution_matching_mask",
            "flow_matching_loss",
            "foreground_angles",
            "foreground_weight",
            "foreground_weight_map",
            "memory_weight",
            "pin_clean_frames",
            "sample_flow_matching",
            "sample_score_timestep",
            "sample_timestep_index",
            "sample_window_timestep_index",
            "training_weight",
            "training_weight_table",
        ),
        "recipes.bidirectional": ("BidirectionalTrainer",),
        "recipes.distillation": (
            "CRITIC_LR",
            "GENERATOR_EVERY",
            "DiffusedRollout",
            "DistillationParts",
            "DistillationTrainer",
            "RolloutForward",
            "RolloutInputs",
            "RolloutSampler",
            "ScoreForward",
        ),
        "recipes.flow_matching": (
            "FlowMatchingTrainer",
            "GeneratorForward",
            "TrainerParts",
            "encode_raw_frames",
            "visibility_inputs",
        ),
        "recipes.teacher_forcing": ("TeacherForcingTrainer",),
        "trainer": ("STEP_TIMES", "BatchStream", "Trainer"),
        "window": (
            "MEMORY_PROB",
            "MemoryWindowConfig",
            "TrainingWindow",
            "memory_frame_config",
            "memory_this_step",
            "memory_windows",
            "training_window",
        ),
    },
)

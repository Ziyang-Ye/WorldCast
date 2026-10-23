"""What a client loads: its resident weights, its window of the recordings and, for predicted
player states, its closed loop (under live controls, its live player states)."""

from dataclasses import dataclass
from typing import Any

import torch

from worldcast.config.inference import InferenceConfig
from worldcast.data.latents import BLOCK, FIRST_TARGET, VIDEO_FRAMES_PER_LATENT
from worldcast.data.recordings import MediaIndex, RoundIndexRow, read_player_ticks
from worldcast.data.window import (
    ClientWindow,
    DataPaths,
    WindowSpec,
    covered_frames,
    load_client_window,
)
from worldcast.engine.generator import backbone_config, load_prompt_embeds, load_vae
from worldcast.modeling.build import load_generator
from worldcast.modeling.depth_head import DepthPredictor, load_depth_predictor
from worldcast.modeling.state_model import (
    MAPS,
    StateModel,
    load_cell_table,
    load_state_model,
    map_index,
)
from worldcast.modeling.wan22.model import WorldCastGenerator
from worldcast.modeling.wan22.vae import Wan22VAE
from worldcast.player_state import (
    ClosedLoop,
    LiveLoop,
    LiveStateReader,
    PhysicsPrior,
    PositionExchange,
    StateReader,
    Walkable,
)

from .messages import MessageWorldState
from .serving import ServingOptions
from .taehv import TinyDecoder, load_taew2_2
from .world_state import WorldState

__all__ = ["ClientModels", "client_latents", "load_closed_loop", "load_live_loop", "load_window"]

#: The generator's modules a client's block loop conditions on: the player state field's injector
#: and the two of scene state, which a generator trained without scene state does not have.
_CLIENT_MODULES = ("state_injector", "observer_signals", "ray_embedding")
_SCENE_STATE_MODULES = ("observer_signals", "ray_embedding")


@dataclass
class ClientModels:
    """The weights a client keeps resident: loaded once, shared by its rollouts.

    Attributes:
        generator (WorldCastGenerator): the generator, bf16 on CUDA, float32 on the CPU.
        prompt_embeds (Tensor): ``[1, 512, 4096]`` the fixed prompt's embedding.
        depth (DepthPredictor): the depth head and its read-out.
        wan_vae (Wan22VAE | None): the Wan2.2 VAE, for the decoder ``wan``.
        tiny (TinyDecoder | None): the tiny decoder, for the decoder ``taehv``.
        state_model (StateModel | None): the state model, for predicted player states and live
            controls.
    """

    generator: WorldCastGenerator
    prompt_embeds: torch.Tensor
    depth: DepthPredictor
    wan_vae: Wan22VAE | None = None
    tiny: TinyDecoder | None = None
    state_model: StateModel | None = None

    @classmethod
    def load(cls, cfg: InferenceConfig, serving: ServingOptions) -> "ClientModels":
        """The generator, prompt embedding, depth head, decoder and state model, on ``run.device``.

        The backbone dimensions come from the Wan2.2 snapshot's ``config.json`` under
        ``paths.wan22_root`` (without one: the 5B model's).

        Raises:
            ValueError: the checkpoint holds a generator without the player state field, or, with
                ``model.scene_state``, without scene state, which a client does not run; or,
                without, the modules of scene state, which that generator does not run.
        """
        p, device = cfg.paths, torch.device(cfg.run.device)
        p.require("checkpoint", "depth_head", "depth_readout")
        generator = load_generator(p.checkpoint, backbone_config(p.wan22_root), device=device)
        if cfg.model.scene_state:
            absent = [name for name in _CLIENT_MODULES if getattr(generator, name) is None]
            if absent:
                raise ValueError(
                    f"{p.checkpoint} holds a generator without {', '.join(absent)}: a client runs"
                    " the generator with the player state field and the scene state, or with"
                    " model.scene_state false the one trained without scene state"
                    " (tools/evaluate.py scores the others)"
                )
        else:
            if generator.state_injector is None:
                raise ValueError(
                    f"{p.checkpoint} holds a generator without state_injector: a client runs the"
                    " generator with the player state field"
                )
            held = [name for name in _SCENE_STATE_MODULES if getattr(generator, name) is not None]
            if held:
                raise ValueError(
                    f"{p.checkpoint} holds {', '.join(held)}, modules of scene state: with"
                    " model.scene_state false paths.checkpoint is the client weight file of the"
                    " generator trained without scene state, which tools/make_noscene_generator.py"
                    " makes"
                )
        embeds = load_prompt_embeds(p.prompt_embedding, p.wan22_root, device)
        depth = load_depth_predictor(p.depth_head, p.depth_readout, device=device)
        models = cls(generator, embeds, depth)
        models.load_missing(cfg, serving)
        return models

    def load_missing(self, cfg: InferenceConfig, serving: ServingOptions) -> None:
        """Load what ``cfg`` and ``serving`` need and is not loaded yet: the decoder, on
        ``serving.decoder_device`` (a decoder loaded for another device is moved there; the Wan2.2
        VAE's convolutions in the layout of ``serving.decoder_channels_last``), and, for predicted
        player states and live controls, the state model."""
        p, device = cfg.paths, torch.device(cfg.run.device)
        where = torch.device(serving.decoder_device or cfg.run.device)
        if cfg.player_state.source in ("predicted", "live") and self.state_model is None:
            p.require("state_model", "state_model_cells")
            tables = load_cell_table(p.state_model_cells)
            self.state_model = load_state_model(p.state_model, tables, device=device)
        if serving.decoder == "wan":
            if self.wan_vae is None:
                p.require("wan22_root")
                self.wan_vae = load_vae(p.wan22_root, where)
            self.wan_vae.to(where)
            self.wan_vae.set_decoder_channels_last(serving.decoder_channels_last)
        if serving.decoder == "taehv":
            if self.tiny is None:
                self.tiny = load_taew2_2(serving.taehv_path, device=where)
            self.tiny.to(where)


def client_latents(cfg: InferenceConfig, covered: int) -> int:
    """The latent frames N of a client's rollout, ``1 + 4 k``.

    ``run.latent_frames`` clipped to the client's coverage and to ``run.max_blocks``, rounded down
    to ``1 + 4 k``; raises if not even one block that reads the scene state fits.

    Args:
        cfg (InferenceConfig): ``run.latent_frames`` and ``run.max_blocks``.
        covered (int): the video frames the client's recording covers from the rollout's start.
    """
    n = min(int(cfg.run.latent_frames), 1 + max(0, int(covered) - 1) // VIDEO_FRAMES_PER_LATENT)
    if cfg.run.max_blocks:
        n = min(n, FIRST_TARGET + BLOCK * int(cfg.run.max_blocks))
    n = 1 + ((n - 1) // BLOCK) * BLOCK
    if n < FIRST_TARGET + BLOCK:
        raise ValueError(
            f"the client's recording covers {covered} video frames ({n} latent frames): shorter"
            f" than the first six blocks plus one ({FIRST_TARGET + BLOCK} latent frames)"
        )
    return n


def load_window(cfg: InferenceConfig, row: RoundIndexRow) -> ClientWindow:
    """The client window of ``row``, as long as its recording and ``run.latent_frames`` allow,
    its turns in the camera encoding of the client's generator (``model.scene_state``)."""
    paths = DataPaths.from_config(cfg.paths, "paths")
    media_index = MediaIndex.load(paths.media_index)
    media = media_index.media(row.media_id)
    ticks = read_player_ticks(media, paths.dataset_root)
    covered = covered_frames(ticks, media, row.start_frame, WindowSpec(cfg.run.latent_frames))
    spec = WindowSpec(client_latents(cfg, covered), camera_encoding=cfg.model.stage.camera_encoding)
    return load_client_window(row, media_index, paths, spec)


def load_closed_loop(
    cfg: InferenceConfig,
    window: ClientWindow,
    batch: dict[str, Any],
    models: ClientModels,
    world_state: PositionExchange,
) -> ClosedLoop:
    """The client's predicted player states (``player_state.source = predicted``).

    Args:
        cfg (InferenceConfig): ``paths.dataset_root`` (the client's controls) and
            ``paths.physics_prior``.
        window (ClientWindow): the client's window.
        batch (dict[str, Any]): the round batch; its player-state table is replaced by the
            predicted one.
        models (ClientModels): the depth head and the state model.
        world_state (PositionExchange): the shared world state, where the clients' positions are
            published.

    Returns:
        ClosedLoop: the client's closed loop.
    """
    p, row, media = cfg.paths, window.row, window.media
    p.require("physics_prior")
    reader = StateReader(
        models.state_model,
        # the recorded buttons, without jump recall: as the state model was trained
        read_player_ticks(media, p.dataset_root, jump_recall=False),
        map_name=media.map_name,
        source_fps=media.fps,
        start_frame=row.start_frame,
        device=cfg.run.device,
    )
    return ClosedLoop.build(
        batch,
        reader=reader,
        client=row.media_id,
        client_slot=media.player_slot,
        round_media={s: m.media_id for s, m in window.round_slots.items()},
        others=row.other_clients(),
        latent_frames=window.spec.latent_frames,
        start_frame=row.start_frame,
        prior=PhysicsPrior.load(p.physics_prior),
        depth_fn=models.depth.log_depth,
        world_state=world_state,
    )


def load_live_loop(
    cfg: InferenceConfig,
    window: ClientWindow,
    batch: dict[str, Any],
    models: ClientModels,
    world_state: WorldState,
) -> LiveLoop:
    """The client's player states under live controls (``player_state.source = live``).

    Args:
        cfg (InferenceConfig): ``paths.dataset_root`` (the client's recorded first ticks),
            ``paths.physics_prior``, ``live.wall_keys``, ``live.dead_zone_u``, ``live.hop`` and
            ``run.device``.
        window (ClientWindow): the client's window.
        batch (dict[str, Any]): the round batch, rewritten for live play.
        models (ClientModels): the depth head and the state model.
        world_state (WorldState): the shared world state, over messages: where the clients publish
            their rows.

    Returns:
        LiveLoop: the client's live player states.

    Raises:
        ValueError: the world state is not over messages (live play is the live demo's), or the
            round's map is not one the state model knows.
    """
    p, row, media = cfg.paths, window.row, window.media
    if not isinstance(world_state, MessageWorldState):
        raise ValueError(
            "player_state.source live plays over the live demo's messages (a MessageWorldState),"
            f" not on a {type(world_state).__name__}"
        )
    if media.map_name not in MAPS:
        raise ValueError(
            "player_state.source live reads the own position with the state model, which knows"
            f" the maps {', '.join(MAPS)}; the round is on {media.map_name}"
        )
    p.require("physics_prior")
    reader = LiveStateReader(
        models.state_model,
        # the recorded first ticks without jump recall: as the state model was trained
        read_player_ticks(media, p.dataset_root, jump_recall=False),
        map_name=media.map_name,
        source_fps=media.fps,
        start_frame=row.start_frame,
        device=cfg.run.device,
    )
    return LiveLoop.build(
        batch,
        reader=reader,
        client=row.media_id,
        client_slot=media.player_slot,
        round_media={s: m.media_id for s, m in window.round_slots.items()},
        latent_frames=window.spec.latent_frames,
        start_frame=row.start_frame,
        prior=PhysicsPrior.load(p.physics_prior),
        walkable=Walkable(models.state_model.place_head.cell_columns(map_index(media.map_name))),
        depth_fn=models.depth.log_depth,
        world_state=world_state,
        device=cfg.run.device,
        wall_keys=cfg.live.wall_keys,
        camera_encoding=window.spec.camera_encoding,
        dead_zone_u=cfg.live.dead_zone_u,
        hop=cfg.live.hop,
    )

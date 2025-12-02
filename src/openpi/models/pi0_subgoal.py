"""Pi0-Subgoal: Triple-system architecture with subgoal planning.

Extends Pi0 from dual system (prefix + suffix) to triple system (prefix + infix + suffix)
by adding a subgoal module for long-horizon planning.

Architecture:
- Prefix: [images, text] - VLM encoding (bidirectional attention within block)
- Infix: [current state, subgoal trace] - Subgoal prediction via flow matching
- Suffix: [actions] - Action generation via flow matching

Attention pattern (block-wise):
- Within each block: bidirectional
- Between blocks: causal (later blocks attend to earlier blocks)
"""

import dataclasses
import logging
from typing import TYPE_CHECKING

import einops
import flax.nnx as nnx
import flax.nnx.bridge as nnx_bridge
import jax
import jax.numpy as jnp
from typing_extensions import override

from openpi.models import model as _model
import openpi.models.gemma as _gemma
import openpi.models.siglip as _siglip
from openpi.shared import array_typing as at

if TYPE_CHECKING:
    pass

logger = logging.getLogger("openpi")


@dataclasses.dataclass(frozen=True)
class Pi0SubgoalConfig(_model.BaseModelConfig):
    """Configuration for Pi0-Subgoal model."""

    dtype: str = "bfloat16"
    paligemma_variant: _gemma.Variant = "gemma_2b"
    action_expert_variant: _gemma.Variant = "gemma_300m"

    # Action space configuration
    action_dim: int = 32
    action_horizon: int = 50
    max_token_len: int = 48

    # Subgoal configuration
    subgoal_interval: int = 100  # How far ahead subgoals look (in timesteps)
    subgoal_horizon: int = 10  # Number of waypoints in subgoal trace (like action_horizon)
    subgoal_expert_variant: _gemma.Variant = "gemma_300m"
    subgoal_loss_weight: float = 1.0  # Lambda for loss weighting: L = L_subgoal + lambda * L_action

    # Inference configuration
    subgoal_num_steps: int = 10  # Denoising steps for subgoal generation
    action_num_steps: int = 10  # Denoising steps for action generation

    @property
    @override
    def model_type(self) -> _model.ModelType:
        return _model.ModelType.PI0_SUBGOAL

    @override
    def create(self, rng: at.KeyArrayLike) -> "Pi0Subgoal":
        return Pi0Subgoal(self, rngs=nnx.Rngs(rng))

    @override
    def inputs_spec(self, *, batch_size: int = 1) -> tuple[_model.Observation, _model.Actions]:
        image_spec = jax.ShapeDtypeStruct([batch_size, *_model.IMAGE_RESOLUTION, 3], jnp.float32)
        image_mask_spec = jax.ShapeDtypeStruct([batch_size], jnp.bool_)

        with at.disable_typechecking():
            observation_spec = _model.Observation(
                images={
                    "base_0_rgb": image_spec,
                    "left_wrist_0_rgb": image_spec,
                    "right_wrist_0_rgb": image_spec,
                },
                image_masks={
                    "base_0_rgb": image_mask_spec,
                    "left_wrist_0_rgb": image_mask_spec,
                    "right_wrist_0_rgb": image_mask_spec,
                },
                state=jax.ShapeDtypeStruct([batch_size, self.action_dim], jnp.float32),
                tokenized_prompt=jax.ShapeDtypeStruct([batch_size, self.max_token_len], jnp.int32),
                tokenized_prompt_mask=jax.ShapeDtypeStruct([batch_size, self.max_token_len], bool),
            )
        action_spec = jax.ShapeDtypeStruct([batch_size, self.action_horizon, self.action_dim], jnp.float32)

        return observation_spec, action_spec

    def subgoal_spec(self, *, batch_size: int = 1) -> at.Float[at.Array, "b sl ad"]:
        """Returns the subgoal trace specification."""
        return jax.ShapeDtypeStruct([batch_size, self.subgoal_horizon, self.action_dim], jnp.float32)


@at.typecheck
def posemb_sincos(
    pos: at.Real[at.Array, " b"], embedding_dim: int, min_period: float, max_period: float
) -> at.Float[at.Array, "b {embedding_dim}"]:
    """Computes sine-cosine positional embedding vectors for scalar positions."""
    if embedding_dim % 2 != 0:
        raise ValueError(f"embedding_dim ({embedding_dim}) must be divisible by 2")

    fraction = jnp.linspace(0.0, 1.0, embedding_dim // 2)
    period = min_period * (max_period / min_period) ** fraction
    sinusoid_input = jnp.einsum(
        "i,j->ij",
        pos,
        1.0 / period * 2 * jnp.pi,
        precision=jax.lax.Precision.HIGHEST,
    )
    return jnp.concatenate([jnp.sin(sinusoid_input), jnp.cos(sinusoid_input)], axis=-1)


def make_attn_mask(input_mask, mask_ar):
    """Adapted from big_vision.

    Tokens can attend to valid inputs tokens which have a cumulative mask_ar
    smaller or equal to theirs. This way `mask_ar` bool[?B, N] can be used to
    setup several types of attention, for example:

      [[1 1 1 1 1 1]]: pure causal attention.

      [[0 0 0 1 1 1]]: prefix-lm attention. The first 3 tokens can attend between
          themselves and the last 3 tokens have a causal attention. The first
          entry could also be a 1 without changing behaviour.

      [[1 0 1 0 1 0 0 1 0 0]]: causal attention between 4 blocks. Tokens of a
          block can attend all previous blocks and all tokens on the same block.

    Args:
      input_mask: bool[B, N] true if its part of the input, false if padding.
      mask_ar: bool[?B, N] mask that's true where previous tokens cannot depend on
        it and false where it shares the same attention mask as the previous token.
    """
    mask_ar = jnp.broadcast_to(mask_ar, input_mask.shape)
    cumsum = jnp.cumsum(mask_ar, axis=1)
    attn_mask = cumsum[:, None, :] <= cumsum[:, :, None]
    valid_mask = input_mask[:, None, :] * input_mask[:, :, None]
    return jnp.logical_and(attn_mask, valid_mask)


class Pi0Subgoal(_model.BaseModel):
    """Pi0-Subgoal model with triple-system architecture."""

    def __init__(self, config: Pi0SubgoalConfig, rngs: nnx.Rngs):
        super().__init__(config.action_dim, config.action_horizon, config.max_token_len)

        self.config = config
        self.subgoal_horizon = config.subgoal_horizon

        paligemma_config = _gemma.get_config(config.paligemma_variant)
        subgoal_expert_config = _gemma.get_config(config.subgoal_expert_variant)
        action_expert_config = _gemma.get_config(config.action_expert_variant)

        # Shared PaliGemma backbone for prefix (VLM)
        # Uses configs list: [paligemma, subgoal_expert, action_expert]
        llm = nnx_bridge.ToNNX(
            _gemma.Module(
                configs=[paligemma_config, subgoal_expert_config, action_expert_config],
                embed_dtype=config.dtype,
            )
        )
        llm.lazy_init(rngs=rngs, method="init", use_adarms=[False, False, False])

        img = nnx_bridge.ToNNX(
            _siglip.Module(
                num_classes=paligemma_config.width,
                variant="So400m/14",
                pool_type="none",
                scan=True,
                dtype_mm=config.dtype,
            )
        )
        img.lazy_init(next(iter(config.fake_obs().images.values())), train=False, rngs=rngs)

        self.PaliGemma = nnx.Dict(llm=llm, img=img)

        # Subgoal module projections
        self.subgoal_state_proj = nnx.Linear(config.action_dim, subgoal_expert_config.width, rngs=rngs)
        self.subgoal_in_proj = nnx.Linear(config.action_dim, subgoal_expert_config.width, rngs=rngs)
        self.subgoal_time_mlp_in = nnx.Linear(subgoal_expert_config.width, subgoal_expert_config.width, rngs=rngs)
        self.subgoal_time_mlp_out = nnx.Linear(subgoal_expert_config.width, subgoal_expert_config.width, rngs=rngs)
        self.subgoal_out_proj = nnx.Linear(subgoal_expert_config.width, config.action_dim, rngs=rngs)

        # Action module projections
        self.action_state_proj = nnx.Linear(config.action_dim, action_expert_config.width, rngs=rngs)
        self.action_subgoal_proj = nnx.Linear(config.action_dim, action_expert_config.width, rngs=rngs)
        self.action_in_proj = nnx.Linear(config.action_dim, action_expert_config.width, rngs=rngs)
        self.action_time_mlp_in = nnx.Linear(action_expert_config.width, action_expert_config.width, rngs=rngs)
        self.action_time_mlp_out = nnx.Linear(action_expert_config.width, action_expert_config.width, rngs=rngs)
        self.action_out_proj = nnx.Linear(action_expert_config.width, config.action_dim, rngs=rngs)

        # This attribute gets automatically set by model.train() and model.eval().
        self.deterministic = True

    @at.typecheck
    def embed_prefix(
        self, obs: _model.Observation
    ) -> tuple[at.Float[at.Array, "b s emb"], at.Bool[at.Array, "b s"], at.Bool[at.Array, " s"]]:
        """Embed prefix (images + text).

        Returns:
            tokens: Embedded prefix tokens [batch, seq, embed_dim]
            input_mask: Valid token mask [batch, seq]
            ar_mask: Autoregressive mask pattern [seq]
        """
        input_mask = []
        ar_mask = []
        tokens = []

        # Embed images
        for name in obs.images:
            image_tokens, _ = self.PaliGemma.img(obs.images[name], train=False)

            tokens.append(image_tokens)
            input_mask.append(
                einops.repeat(
                    obs.image_masks[name],
                    "b -> b s",
                    s=image_tokens.shape[1],
                )
            )
            # Image tokens attend to each other (bidirectional within block)
            ar_mask += [False] * image_tokens.shape[1]

        # Add language (tokenized inputs)
        if obs.tokenized_prompt is not None:
            tokenized_inputs = self.PaliGemma.llm(obs.tokenized_prompt, method="embed")
            tokens.append(tokenized_inputs)
            input_mask.append(obs.tokenized_prompt_mask)
            # Full attention between image and language inputs
            ar_mask += [False] * tokenized_inputs.shape[1]

        tokens = jnp.concatenate(tokens, axis=1)
        input_mask = jnp.concatenate(input_mask, axis=1)
        ar_mask = jnp.array(ar_mask)
        return tokens, input_mask, ar_mask

    @at.typecheck
    def embed_infix(
        self,
        obs: _model.Observation,
        noisy_subgoals: at.Float[at.Array, "b sl ad"],
        timestep: at.Float[at.Array, " b"],
    ) -> tuple[
        at.Float[at.Array, "b s emb"],
        at.Bool[at.Array, "b s"],
        at.Bool[at.Array, " s"],
    ]:
        """Embed infix (current state + subgoal trace).

        Args:
            obs: Observation containing current state
            noisy_subgoals: Noisy subgoal trace [batch, subgoal_horizon, action_dim]
            timestep: Flow matching timestep [batch]

        Returns:
            tokens: Embedded infix tokens [batch, 1 + subgoal_horizon, embed_dim]
            input_mask: Valid token mask [batch, 1 + subgoal_horizon]
            ar_mask: Autoregressive mask pattern [1 + subgoal_horizon]
        """
        batch_size = obs.state.shape[0]

        # Project current state to single token
        state_token = self.subgoal_state_proj(obs.state)[:, None, :]  # [b, 1, emb]

        # Project noisy subgoals
        subgoal_tokens = self.subgoal_in_proj(noisy_subgoals)  # [b, sl, emb]

        # Embed timestep using sine-cosine positional encoding
        time_emb = posemb_sincos(timestep, self.subgoal_in_proj.out_features, min_period=4e-3, max_period=4.0)
        time_emb = self.subgoal_time_mlp_in(time_emb)
        time_emb = nnx.swish(time_emb)
        time_emb = self.subgoal_time_mlp_out(time_emb)
        time_emb = nnx.swish(time_emb)

        # Add time embedding to subgoal tokens
        subgoal_tokens = subgoal_tokens + time_emb[:, None, :]

        # Concatenate: [current_state, subgoal_trace]
        tokens = jnp.concatenate([state_token, subgoal_tokens], axis=1)
        input_mask = jnp.ones((batch_size, 1 + self.subgoal_horizon), dtype=jnp.bool_)

        # Block-wise attention: bidirectional within infix
        # First token (state) starts a new block, rest attend bidirectionally
        ar_mask = jnp.array([True] + [False] * self.subgoal_horizon)

        return tokens, input_mask, ar_mask

    @at.typecheck
    def embed_suffix(
        self,
        obs: _model.Observation,
        subgoal_trace: at.Float[at.Array, "b sl ad"],
        noisy_actions: at.Float[at.Array, "b ah ad"],
        timestep: at.Float[at.Array, " b"],
    ) -> tuple[
        at.Float[at.Array, "b s emb"],
        at.Bool[at.Array, "b s"],
        at.Bool[at.Array, " s"],
    ]:
        """Embed suffix (current state + subgoal trace + actions) for action module.

        The action module needs the infix context (state + subgoals) to condition on.

        Args:
            obs: Observation containing current state
            subgoal_trace: Subgoal trace (ground-truth during training, predicted during inference)
            noisy_actions: Noisy actions [batch, action_horizon, action_dim]
            timestep: Flow matching timestep [batch]

        Returns:
            tokens: Embedded tokens [batch, 1 + subgoal_horizon + action_horizon, embed_dim]
            input_mask: Valid token mask
            ar_mask: Autoregressive mask pattern
        """
        batch_size = obs.state.shape[0]

        # Project current state for action module
        state_token = self.action_state_proj(obs.state)[:, None, :]  # [b, 1, emb]

        # Project subgoal trace for action module
        subgoal_tokens = self.action_subgoal_proj(subgoal_trace)  # [b, sl, emb]

        # Project noisy actions
        action_tokens = self.action_in_proj(noisy_actions)  # [b, ah, emb]

        # Embed timestep
        time_emb = posemb_sincos(timestep, self.action_in_proj.out_features, min_period=4e-3, max_period=4.0)
        time_emb = self.action_time_mlp_in(time_emb)
        time_emb = nnx.swish(time_emb)
        time_emb = self.action_time_mlp_out(time_emb)
        time_emb = nnx.swish(time_emb)

        # Add time embedding to action tokens only
        action_tokens = action_tokens + time_emb[:, None, :]

        # Concatenate: [current_state, subgoal_trace, actions]
        # This forms the "infix + suffix" for the action expert
        tokens = jnp.concatenate([state_token, subgoal_tokens, action_tokens], axis=1)

        total_len = 1 + self.subgoal_horizon + self.action_horizon
        input_mask = jnp.ones((batch_size, total_len), dtype=jnp.bool_)

        # Block-wise attention pattern:
        # - [state, subgoals] form the infix block (bidirectional within)
        # - [actions] form the suffix block (bidirectional within)
        # - suffix attends to infix (causal between blocks)
        ar_mask = jnp.array(
            [True]  # state starts infix block
            + [False] * self.subgoal_horizon  # subgoals bidirectional with state
            + [True]  # first action starts suffix block
            + [False] * (self.action_horizon - 1)  # actions bidirectional within suffix
        )

        return tokens, input_mask, ar_mask

    @override
    def compute_loss(
        self, rng: at.KeyArrayLike, observation: _model.Observation, actions: _model.Actions, *, train: bool = False
    ) -> at.Float[at.Array, "*b ah"]:
        """Compute combined loss for subgoal and action prediction.

        Note: This method expects `observation` to contain a `subgoal_trace` field
        with the ground-truth subgoal trace.
        """
        preprocess_rng, subgoal_noise_rng, subgoal_time_rng, action_noise_rng, action_time_rng = jax.random.split(
            rng, 5
        )
        observation = _model.preprocess_observation(preprocess_rng, observation, train=train)

        # Get subgoal trace from observation (must be added by data transform)
        # For now, assume it's passed via a custom field or we extract from future states
        # TODO: Add subgoal_trace to Observation or pass separately
        subgoal_trace = observation.subgoal_trace  # [b, sl, ad]

        batch_shape = actions.shape[:-2]

        # === Subgoal Flow Matching ===
        subgoal_noise = jax.random.normal(subgoal_noise_rng, subgoal_trace.shape)
        subgoal_time = jax.random.beta(subgoal_time_rng, 1.5, 1, batch_shape) * 0.999 + 0.001
        subgoal_time_expanded = subgoal_time[..., None, None]
        subgoal_x_t = subgoal_time_expanded * subgoal_noise + (1 - subgoal_time_expanded) * subgoal_trace
        subgoal_u_t = subgoal_noise - subgoal_trace

        # Embed prefix
        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)

        # Embed infix for subgoal prediction
        infix_tokens, infix_mask, infix_ar_mask = self.embed_infix(observation, subgoal_x_t, subgoal_time)

        # Combined attention mask for prefix + infix
        input_mask = jnp.concatenate([prefix_mask, infix_mask], axis=1)
        ar_mask = jnp.concatenate([prefix_ar_mask, infix_ar_mask], axis=0)
        attn_mask = make_attn_mask(input_mask, ar_mask)
        positions = jnp.cumsum(input_mask, axis=1) - 1

        # Forward pass: prefix (expert 0) + infix (expert 1)
        (prefix_out, infix_out, _), _ = self.PaliGemma.llm(
            [prefix_tokens, infix_tokens, None], mask=attn_mask, positions=positions
        )

        # Extract subgoal predictions (skip state token)
        subgoal_v_t = self.subgoal_out_proj(infix_out[:, 1:])  # [b, sl, ad]
        subgoal_loss = jnp.mean(jnp.square(subgoal_v_t - subgoal_u_t), axis=-1)  # [b, sl]

        # === Action Flow Matching ===
        action_noise = jax.random.normal(action_noise_rng, actions.shape)
        action_time = jax.random.beta(action_time_rng, 1.5, 1, batch_shape) * 0.999 + 0.001
        action_time_expanded = action_time[..., None, None]
        action_x_t = action_time_expanded * action_noise + (1 - action_time_expanded) * actions
        action_u_t = action_noise - actions

        # Embed suffix for action prediction (using ground-truth subgoals)
        suffix_tokens, suffix_mask, suffix_ar_mask = self.embed_suffix(
            observation, subgoal_trace, action_x_t, action_time
        )

        # Combined attention mask for prefix + suffix (suffix includes infix context)
        input_mask = jnp.concatenate([prefix_mask, suffix_mask], axis=1)
        ar_mask = jnp.concatenate([prefix_ar_mask, suffix_ar_mask], axis=0)
        attn_mask = make_attn_mask(input_mask, ar_mask)
        positions = jnp.cumsum(input_mask, axis=1) - 1

        # Forward pass: prefix (expert 0) + suffix with infix (expert 2)
        (prefix_out, _, suffix_out), _ = self.PaliGemma.llm(
            [prefix_tokens, None, suffix_tokens], mask=attn_mask, positions=positions
        )

        # Extract action predictions (skip state + subgoal tokens)
        action_v_t = self.action_out_proj(suffix_out[:, 1 + self.subgoal_horizon :])  # [b, ah, ad]
        action_loss = jnp.mean(jnp.square(action_v_t - action_u_t), axis=-1)  # [b, ah]

        # Combined loss: L = L_subgoal + lambda * L_action
        # Return action_loss shape for compatibility with training loop
        # Average subgoal loss over subgoal_horizon and broadcast to action_horizon
        subgoal_loss_mean = jnp.mean(subgoal_loss, axis=-1, keepdims=True)  # [b, 1]
        subgoal_loss_broadcast = jnp.broadcast_to(subgoal_loss_mean, action_loss.shape)  # [b, ah]

        return subgoal_loss_broadcast + self.config.subgoal_loss_weight * action_loss

    @override
    def sample_actions(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        *,
        subgoal_num_steps: int | at.Int[at.Array, ""] | None = None,
        action_num_steps: int | at.Int[at.Array, ""] | None = None,
        subgoal_noise: at.Float[at.Array, "b sl ad"] | None = None,
        action_noise: at.Float[at.Array, "b ah ad"] | None = None,
    ) -> _model.Actions:
        """Sample actions using sequential flow matching (subgoals first, then actions)."""
        observation = _model.preprocess_observation(None, observation, train=False)

        if subgoal_num_steps is None:
            subgoal_num_steps = self.config.subgoal_num_steps
        if action_num_steps is None:
            action_num_steps = self.config.action_num_steps

        subgoal_rng, action_rng = jax.random.split(rng)
        batch_size = observation.state.shape[0]

        # === Step 1: Embed prefix and cache KV ===
        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
        prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)
        positions = jnp.cumsum(prefix_mask, axis=1) - 1
        _, prefix_kv_cache = self.PaliGemma.llm(
            [prefix_tokens, None, None], mask=prefix_attn_mask, positions=positions
        )

        # === Step 2: Generate subgoals via flow matching ===
        subgoal_dt = -1.0 / subgoal_num_steps
        if subgoal_noise is None:
            subgoal_noise = jax.random.normal(subgoal_rng, (batch_size, self.subgoal_horizon, self.action_dim))

        def subgoal_step(carry):
            subgoals, time = carry

            infix_tokens, infix_mask, infix_ar_mask = self.embed_infix(
                observation, subgoals, jnp.broadcast_to(time, batch_size)
            )

            # Attention: infix attends to prefix via KV cache
            infix_attn_mask = make_attn_mask(infix_mask, infix_ar_mask)
            prefix_attn_mask_for_infix = einops.repeat(prefix_mask, "b p -> b s p", s=infix_tokens.shape[1])
            full_attn_mask = jnp.concatenate([prefix_attn_mask_for_infix, infix_attn_mask], axis=-1)

            positions = jnp.sum(prefix_mask, axis=-1)[:, None] + jnp.cumsum(infix_mask, axis=-1) - 1

            (_, infix_out, _), _ = self.PaliGemma.llm(
                [None, infix_tokens, None],
                mask=full_attn_mask,
                positions=positions,
                kv_cache=prefix_kv_cache,
            )

            v_t = self.subgoal_out_proj(infix_out[:, 1:])  # Skip state token
            return subgoals + subgoal_dt * v_t, time + subgoal_dt

        def subgoal_cond(carry):
            _, time = carry
            return time >= -subgoal_dt / 2

        predicted_subgoals, _ = jax.lax.while_loop(subgoal_cond, subgoal_step, (subgoal_noise, 1.0))

        # === Step 3: Generate actions via flow matching ===
        action_dt = -1.0 / action_num_steps
        if action_noise is None:
            action_noise = jax.random.normal(action_rng, (batch_size, self.action_horizon, self.action_dim))

        def action_step(carry):
            actions, time = carry

            suffix_tokens, suffix_mask, suffix_ar_mask = self.embed_suffix(
                observation, predicted_subgoals, actions, jnp.broadcast_to(time, batch_size)
            )

            # Attention: suffix attends to prefix via KV cache
            suffix_attn_mask = make_attn_mask(suffix_mask, suffix_ar_mask)
            prefix_attn_mask_for_suffix = einops.repeat(prefix_mask, "b p -> b s p", s=suffix_tokens.shape[1])
            full_attn_mask = jnp.concatenate([prefix_attn_mask_for_suffix, suffix_attn_mask], axis=-1)

            positions = jnp.sum(prefix_mask, axis=-1)[:, None] + jnp.cumsum(suffix_mask, axis=-1) - 1

            (_, _, suffix_out), _ = self.PaliGemma.llm(
                [None, None, suffix_tokens],
                mask=full_attn_mask,
                positions=positions,
                kv_cache=prefix_kv_cache,
            )

            # Skip state + subgoal tokens to get action predictions
            v_t = self.action_out_proj(suffix_out[:, 1 + self.subgoal_horizon :])
            return actions + action_dt * v_t, time + action_dt

        def action_cond(carry):
            _, time = carry
            return time >= -action_dt / 2

        predicted_actions, _ = jax.lax.while_loop(action_cond, action_step, (action_noise, 1.0))

        return predicted_actions

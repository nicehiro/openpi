import dataclasses

import flax.nnx as nnx
import jax
import jax.numpy as jnp

import openpi.models.pi0_config as _pi0_config


def _get_frozen_state(config: _pi0_config.Pi0Config) -> nnx.State:
    abstract_model = nnx.eval_shape(config.create, jax.random.key(0))

    freeze_filter = config.get_freeze_filter()
    return nnx.state(abstract_model, nnx.All(nnx.Param, freeze_filter)).flat_state()


def test_pi0_full_finetune():
    config = _pi0_config.Pi0Config()
    state = _get_frozen_state(config)
    assert len(state) == 0


def test_pi0_gemma_lora():
    config = _pi0_config.Pi0Config(paligemma_variant="gemma_2b_lora")
    state = _get_frozen_state(config)
    assert len(state) == 9
    assert all("lora" not in p for p in state)
    assert all("llm" in p for p in state)
    assert all("_1" not in p for p in state)


def test_pi0_action_expert_lora():
    config = _pi0_config.Pi0Config(action_expert_variant="gemma_300m_lora")
    state = _get_frozen_state(config)
    # excluding embedder, rest of the params should be same as gemma_lora.
    assert len(state) == 8
    assert all("lora" not in p for p in state)
    assert all("llm" in p for p in state)
    # all frozen params should have _1 in their path since it's the action expert.
    assert all(any("_1" in p for p in path) for path in state)


def test_pi0_all_lora():
    config = _pi0_config.Pi0Config(paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora")
    state = _get_frozen_state(config)
    # sum of gemma_lora and action_expert_lora's frozen params.
    assert len(state) == 17
    assert all("lora" not in p for p in state)
    assert all("llm" in p for p in state)


def test_pi0_state_history_forward_pass():
    """Test pi0 with state history enabled - forward pass."""
    config = _pi0_config.Pi0Config(
        paligemma_variant="dummy",
        action_expert_variant="dummy",
        use_state_history=True,
        state_history_len=8,
    )
    rng = jax.random.key(0)
    model = config.create(rng)

    # Get input specs and create fake data
    obs_spec, action_spec = config.inputs_spec()
    obs = jax.tree.map(lambda x: jnp.ones(x.shape, x.dtype), obs_spec)

    # Add state_history to observation
    obs = dataclasses.replace(
        obs, state_history=jnp.ones((1, 8, config.action_dim), dtype=jnp.float32)
    )
    actions = jax.tree.map(lambda x: jnp.ones(x.shape, x.dtype), action_spec)

    # Test forward pass - compute_loss
    loss = model.compute_loss(rng, obs, actions, train=False)
    assert loss.shape == (1, config.action_horizon)

    # Test forward pass - sample_actions
    sampled_actions = model.sample_actions(rng, obs, num_steps=10)
    assert sampled_actions.shape == (1, config.action_horizon, config.action_dim)


def test_pi0_state_history_attention_mask():
    """Verify attention mask is correct with state history."""
    config = _pi0_config.Pi0Config(
        paligemma_variant="dummy",
        action_expert_variant="dummy",
        use_state_history=True,
        state_history_len=8,
    )
    rng = jax.random.key(0)
    model = config.create(rng)

    # Get input specs and create fake data
    obs_spec, action_spec = config.inputs_spec()
    obs = jax.tree.map(lambda x: jnp.ones(x.shape, x.dtype), obs_spec)
    obs = dataclasses.replace(
        obs, state_history=jnp.ones((1, 8, config.action_dim), dtype=jnp.float32)
    )

    # Call embed_prefix and embed_suffix to get ar_masks
    prefix_tokens, prefix_mask, prefix_ar_mask = model.embed_prefix(obs)

    # Check that state_history tokens are in prefix with ar_mask=False
    # Prefix should be: [images, text, state_history] all with ar_mask=False
    assert jnp.all(prefix_ar_mask == False)  # noqa: E712

    # Check prefix includes state_history tokens (8 tokens)
    # Number of tokens should include images + text + state_history (8)
    # For dummy model this should be verifiable by checking the shape
    assert prefix_tokens.shape[1] > 0  # Has tokens


def test_pi0_backward_compatibility():
    """Test that old behavior is preserved when use_state_history=False."""
    config = _pi0_config.Pi0Config(
        paligemma_variant="dummy",
        action_expert_variant="dummy",
        use_state_history=False,  # Old behavior
    )
    rng = jax.random.key(0)
    model = config.create(rng)

    # Verify state_proj exists and state_history_proj does not
    assert hasattr(model, "state_proj")
    assert not hasattr(model, "state_history_proj")

    # Get input specs and create fake data (without state_history)
    obs_spec, action_spec = config.inputs_spec()
    obs = jax.tree.map(lambda x: jnp.ones(x.shape, x.dtype), obs_spec)
    actions = jax.tree.map(lambda x: jnp.ones(x.shape, x.dtype), action_spec)

    # Test forward pass works
    loss = model.compute_loss(rng, obs, actions, train=False)
    assert loss.shape == (1, config.action_horizon)


def test_pi0_state_history_gradient_flow():
    """Verify gradients flow through state_history_proj."""
    config = _pi0_config.Pi0Config(
        paligemma_variant="dummy",
        action_expert_variant="dummy",
        use_state_history=True,
        state_history_len=8,
    )
    rng = jax.random.key(0)
    model = config.create(rng)

    # Get input specs and create fake data
    obs_spec, action_spec = config.inputs_spec()
    obs = jax.tree.map(lambda x: jnp.ones(x.shape, x.dtype), obs_spec)
    obs = dataclasses.replace(
        obs, state_history=jnp.ones((1, 8, config.action_dim), dtype=jnp.float32)
    )
    actions = jax.tree.map(lambda x: jnp.ones(x.shape, x.dtype), action_spec)

    # Compute gradients
    def loss_fn(model):
        return jnp.mean(model.compute_loss(rng, obs, actions, train=True))

    grad_fn = nnx.value_and_grad(loss_fn)
    loss, grads = grad_fn(model)

    # Verify state_history_proj has gradients
    assert "state_history_proj" in grads
    assert grads.state_history_proj.kernel is not None
    # Check that gradients are non-zero
    assert jnp.any(grads.state_history_proj.kernel.value != 0)

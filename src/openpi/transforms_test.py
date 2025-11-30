import numpy as np
import pytest

import openpi.models.tokenizer as _tokenizer
import openpi.transforms as _transforms


def test_repack_transform():
    transform = _transforms.RepackTransform(
        structure={
            "a": {"b": "b/c"},
            "d": "e/f",
        }
    )
    item = {"b": {"c": 1}, "e": {"f": 2}}
    assert transform(item) == {"a": {"b": 1}, "d": 2}


def test_delta_actions():
    item = {"state": np.array([1, 2, 3]), "actions": np.array([[3, 4, 5], [5, 6, 7]])}

    transform = _transforms.DeltaActions(mask=[False, True])
    transformed = transform(item)

    assert np.all(transformed["state"] == np.array([1, 2, 3]))
    assert np.all(transformed["actions"] == np.array([[3, 2, 5], [5, 4, 7]]))


def test_delta_actions_noop():
    item = {"state": np.array([1, 2, 3]), "actions": np.array([[3, 4, 5], [5, 6, 7]])}

    # No-op when the mask is disabled.
    transform = _transforms.DeltaActions(mask=None)
    assert transform(item) is item

    # No-op when there are no actions in the input.
    del item["actions"]
    transform = _transforms.DeltaActions(mask=[True, False])
    assert transform(item) is item


def test_absolute_actions():
    item = {"state": np.array([1, 2, 3]), "actions": np.array([[3, 4, 5], [5, 6, 7]])}

    transform = _transforms.AbsoluteActions(mask=[False, True])
    transformed = transform(item)

    assert np.all(transformed["state"] == np.array([1, 2, 3]))
    assert np.all(transformed["actions"] == np.array([[3, 6, 5], [5, 8, 7]]))


def test_absolute_actions_noop():
    item = {"state": np.array([1, 2, 3]), "actions": np.array([[3, 4, 5], [5, 6, 7]])}

    # No-op when the mask is disabled.
    transform = _transforms.AbsoluteActions(mask=None)
    assert transform(item) is item

    # No-op when there are no actions in the input.
    del item["actions"]
    transform = _transforms.AbsoluteActions(mask=[True, False])
    assert transform(item) is item


def test_make_bool_mask():
    assert _transforms.make_bool_mask(2, -2, 2) == (True, True, False, False, True, True)
    assert _transforms.make_bool_mask(2, 0, 2) == (True, True, True, True)


def test_tokenize_prompt():
    tokenizer = _tokenizer.PaligemmaTokenizer(max_len=12)
    transform = _transforms.TokenizePrompt(tokenizer)

    data = transform({"prompt": "Hello, world!"})

    tok_prompt, tok_mask = tokenizer.tokenize("Hello, world!")
    assert np.allclose(tok_prompt, data["tokenized_prompt"])
    assert np.allclose(tok_mask, data["tokenized_prompt_mask"])


def test_tokenize_no_prompt():
    transform = _transforms.TokenizePrompt(_tokenizer.PaligemmaTokenizer())

    with pytest.raises(ValueError, match="Prompt is required"):
        transform({})


def test_transform_dict():
    # Rename and remove keys.
    input = {"a": {"b": 1, "c": 2}}
    output = _transforms.transform_dict({"a/b": "a/c", "a/c": None}, input)
    assert output == {"a": {"c": 1}}

    # Raises and error since the renamed key conflicts with an existing key.
    with pytest.raises(ValueError, match="Key 'a/c' already exists in output"):
        _transforms.transform_dict({"a/b": "a/c"}, input)

    # Full match is required and so nothing will be removed.
    input = {"a": {"b": 1, "c": 2}}
    output = _transforms.transform_dict({"a": None}, input)
    assert output == input

    # The regex matches the entire key and so the entire input will be removed.
    input = {"a": {"b": 1, "c": 2}}
    output = _transforms.transform_dict({"a.+": None}, input)
    assert output == {}

    # Replace keys using backreferences. All leaves named 'c' are replaced with 'd'.
    input = {"a": {"b": 1, "c": 1}, "b": {"c": 2}}
    output = _transforms.transform_dict({"(.+)/c": r"\1/d"}, input)
    assert output == {"a": {"b": 1, "d": 1}, "b": {"d": 2}}


def test_extract_prompt_from_task():
    transform = _transforms.PromptFromLeRobotTask({1: "Hello, world!"})

    data = transform({"task_index": 1})
    assert data["prompt"] == "Hello, world!"

    with pytest.raises(ValueError, match="task_index=2 not found in task mapping"):
        transform({"task_index": 2})


def test_state_history_transform_full_history():
    """Test StateHistoryTransform with full history from LeRobot."""
    transform = _transforms.StateHistoryTransform(history_len=8)
    # Simulate LeRobot providing 8 timesteps of state history
    original_state = np.array([[1, 2, 3], [2, 3, 4], [3, 4, 5], [4, 5, 6], [5, 6, 7], [6, 7, 8], [7, 8, 9], [8, 9, 10]])
    item = {"state": original_state.copy()}

    transformed = transform(item)

    # Should create state_history with same data as original
    assert np.all(transformed["state_history"] == original_state)
    assert transformed["state_history"].shape == (8, 3)
    # Current state should be the last timestep
    assert np.all(transformed["state"] == np.array([8, 9, 10]))


def test_state_history_transform_single_state():
    """Test StateHistoryTransform with single state (inference case)."""
    transform = _transforms.StateHistoryTransform(history_len=8)
    # Simulate inference with single state
    item = {"state": np.array([1, 2, 3])}

    transformed = transform(item)

    # Should tile the single state to create history
    expected_history = np.tile(np.array([[1, 2, 3]]), (8, 1))
    assert np.all(transformed["state_history"] == expected_history)
    assert transformed["state_history"].shape == (8, 3)
    # Current state should remain the same
    assert np.all(transformed["state"] == np.array([1, 2, 3]))


def test_state_history_transform_short_history():
    """Test StateHistoryTransform with short history (padding needed)."""
    transform = _transforms.StateHistoryTransform(history_len=8)
    # Simulate only 3 timesteps available (episode boundary)
    item = {"state": np.array([[1, 2, 3], [2, 3, 4], [3, 4, 5]])}

    transformed = transform(item)

    # Should pad with first state
    expected_history = np.array([
        [1, 2, 3],  # padding
        [1, 2, 3],  # padding
        [1, 2, 3],  # padding
        [1, 2, 3],  # padding
        [1, 2, 3],  # padding
        [1, 2, 3],  # actual first
        [2, 3, 4],  # actual second
        [3, 4, 5],  # actual third (current)
    ])
    assert np.all(transformed["state_history"] == expected_history)
    assert transformed["state_history"].shape == (8, 3)
    # Current state should be the last timestep
    assert np.all(transformed["state"] == np.array([3, 4, 5]))

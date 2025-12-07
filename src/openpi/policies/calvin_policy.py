import dataclasses

import einops
import numpy as np

from openpi import transforms
from openpi.models import model as _model


def _parse_image(image) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


@dataclasses.dataclass(frozen=True)
class CalvinInputs(transforms.DataTransformFn):
    """Transforms CALVIN LeRobot samples into model-ready inputs."""

    model_type: _model.ModelType

    def __call__(self, data: dict) -> dict:
        base_image = _parse_image(data["observation/image"])
        wrist_image = _parse_image(data["observation/wrist_image"])

        # Handle state - may be a sequence (for subgoal training) or single state (for inference)
        state = np.asarray(data["observation/state"])
        if state.ndim == 2:
            # State is a sequence from delta_timestamps, take first element (current state)
            state = state[0]

        inputs = {
            "state": state,
            "image": {
                "base_0_rgb": base_image,
                "left_wrist_0_rgb": wrist_image,
                "right_wrist_0_rgb": np.zeros_like(base_image),  # CALVIN has only one wrist cam
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.True_,
                "right_wrist_0_rgb": np.True_ if self.model_type == _model.ModelType.PI0_FAST else np.False_,
            },
        }

        if "actions" in data:
            inputs["actions"] = data["actions"]

        if "prompt" in data:
            inputs["prompt"] = data["prompt"]

        # Pass through future_states for subgoal training
        if "future_states" in data:
            inputs["future_states"] = data["future_states"]

        return inputs


@dataclasses.dataclass(frozen=True)
class CalvinOutputs(transforms.DataTransformFn):
    """Converts model outputs back to CALVIN action dimensionality."""

    action_dim: int = 7

    def __call__(self, data: dict) -> dict:
        # Use .copy() to ensure the array is writeable, as CALVIN's robot.relative_to_absolute()
        # performs in-place operations on the action array.
        return {"actions": np.asarray(data["actions"][:, : self.action_dim]).copy()}

"""Evaluate Pi0/Pi0.5 model on CALVIN benchmark.

This script evaluates a trained policy on the CALVIN benchmark for
language-conditioned robot manipulation tasks.

Usage:
    # First, start the policy server in a separate terminal:
    uv run scripts/serve_policy.py policy:checkpoint \
        --policy.config=pi05_calvin \
        --policy.dir=/model/fywang/pi05_base

    # Then run evaluation (use underscores for arguments):
    uv run examples/calvin/main.py \
        --dataset_path=/data/fywang/Calvin/calvin_debug_dataset \
        --eval_seq_len=50
"""

import collections
from collections import Counter
import contextlib
import dataclasses
import hashlib
import json
import logging
import os
import pathlib
import time

# CALVIN environment imports
from calvin_env.envs.play_table_env import get_env_without_tactile
import hydra
import numpy as np
from numpy import pi
from omegaconf import OmegaConf
from openpi_client import image_tools
from openpi_client import websocket_client_policy as _websocket_client_policy
from tqdm import tqdm
import tyro

# Configure logging
logger = logging.getLogger(__name__)


def fnv1_32_hash(data: str) -> int:
    """FNV-1 32-bit hash function replacement using hashlib."""
    return int(hashlib.md5(data.encode()).hexdigest()[:8], 16)


# ============================================================================
# CALVIN Evaluation Utilities (from calvin_agent.evaluation.utils)
# ============================================================================


@contextlib.contextmanager
def temp_seed(seed):
    """Temporarily set numpy random seed."""
    state = np.random.get_state()
    np.random.seed(seed)
    try:
        yield
    finally:
        np.random.set_state(state)


def get_env_state_for_initial_condition(initial_condition):
    """Convert initial condition dict to robot and scene observations."""
    robot_obs = np.array(
        [
            0.02586889,
            -0.2313129,
            0.5712808,
            3.09045411,
            -0.02908596,
            1.50013585,
            0.07999963,
            -1.21779124,
            1.03987629,
            2.11978254,
            -2.34205014,
            -0.87015899,
            1.64119093,
            0.55344928,
            1.0,
        ]
    )
    block_rot_z_range = (pi / 2 - pi / 8, pi / 2 + pi / 8)
    block_slider_left = np.array([-2.40851662e-01, 9.24044687e-02, 4.60990009e-01])
    block_slider_right = np.array([7.03416330e-02, 9.24044687e-02, 4.60990009e-01])
    block_table = [
        np.array([5.00000896e-02, -1.20000177e-01, 4.59990009e-01]),
        np.array([2.29995412e-01, -1.19995140e-01, 4.59990010e-01]),
    ]
    # we want to have a "deterministic" random seed for each initial condition
    seed = fnv1_32_hash(str(initial_condition.values()))
    with temp_seed(seed):
        np.random.shuffle(block_table)

        scene_obs = np.zeros(24)
        if initial_condition["slider"] == "left":
            scene_obs[0] = 0.28
        if initial_condition["drawer"] == "open":
            scene_obs[1] = 0.22
        if initial_condition["lightbulb"] == 1:
            scene_obs[3] = 0.088
        scene_obs[4] = initial_condition["lightbulb"]
        scene_obs[5] = initial_condition["led"]
        # red block
        if initial_condition["red_block"] == "slider_right":
            scene_obs[6:9] = block_slider_right
        elif initial_condition["red_block"] == "slider_left":
            scene_obs[6:9] = block_slider_left
        else:
            scene_obs[6:9] = block_table[0]
        scene_obs[11] = np.random.uniform(*block_rot_z_range)
        # blue block
        if initial_condition["blue_block"] == "slider_right":
            scene_obs[12:15] = block_slider_right
        elif initial_condition["blue_block"] == "slider_left":
            scene_obs[12:15] = block_slider_left
        elif initial_condition["red_block"] == "table":
            scene_obs[12:15] = block_table[1]
        else:
            scene_obs[12:15] = block_table[0]
        scene_obs[17] = np.random.uniform(*block_rot_z_range)
        # pink block
        if initial_condition["pink_block"] == "slider_right":
            scene_obs[18:21] = block_slider_right
        elif initial_condition["pink_block"] == "slider_left":
            scene_obs[18:21] = block_slider_left
        else:
            scene_obs[18:21] = block_table[1]
        scene_obs[23] = np.random.uniform(*block_rot_z_range)

    return robot_obs, scene_obs


def count_success(results):
    """Count success rates for chain of tasks."""
    count = Counter(results)
    step_success = []
    for i in range(1, 6):
        n_success = sum(count[j] for j in reversed(range(i, 6)))
        sr = n_success / len(results)
        step_success.append(sr)
    return step_success


def get_log_dir(log_dir):
    """Get or create log directory."""
    if log_dir is not None:
        log_dir = pathlib.Path(log_dir)
        os.makedirs(log_dir, exist_ok=True)
    else:
        log_dir = pathlib.Path("/tmp/evaluation")
        os.makedirs(log_dir, exist_ok=True)
    print(f"Logging to {log_dir}")
    return log_dir


# ============================================================================
# Main Evaluation Code
# ============================================================================


@dataclasses.dataclass
class Args:
    """Arguments for CALVIN evaluation."""

    # Model server parameters
    host: str = "0.0.0.0"
    port: int = 8000
    resize_size: int = 224
    replan_steps: int = 5

    # CALVIN environment parameters
    dataset_path: str = "/data/fywang/Calvin/calvin_debug_dataset"
    split: str = "validation"
    eval_seq_len: int = 50  # Number of evaluation sequences: 50, 100, or 1000
    ep_len: int = 360  # Max steps per subtask (CALVIN default)

    # Output options
    eval_log_dir: str = "/output/logs/calvin/eval/"
    debug: bool = False

    seed: int = 42


def make_env(dataset_path: str, split: str = "validation"):
    """Create CALVIN environment."""
    val_folder = pathlib.Path(dataset_path) / split
    env = get_env_without_tactile(val_folder, show_gui=False)
    return env


def format_observation(obs: dict, resize_size: int) -> dict:
    """Format CALVIN observation for the openpi policy.

    CALVIN observation keys:
    - rgb_obs: dict with 'rgb_static' (200x200x3) and 'rgb_gripper' (84x84x3)
    - robot_obs: array with [tcp_pos(3), tcp_orn(3), gripper_width(1), arm_joint_pos(7), gripper_action(1)]

    OpenPI CalvinInputs expects:
    - observation/image: base camera image
    - observation/wrist_image: wrist camera image
    - observation/state: robot state (15 dim for CALVIN)
    - prompt: language instruction
    """
    # Extract RGB images from CALVIN observation
    rgb_static = obs["rgb_obs"]["rgb_static"]  # 200x200x3
    rgb_gripper = obs["rgb_obs"]["rgb_gripper"]  # 84x84x3

    # Resize images to model input size with padding
    static_img = image_tools.convert_to_uint8(
        image_tools.resize_with_pad(np.asarray(rgb_static), resize_size, resize_size)
    )
    gripper_img = image_tools.convert_to_uint8(
        image_tools.resize_with_pad(np.asarray(rgb_gripper), resize_size, resize_size)
    )

    # Extract robot state (full 15-dim state for CALVIN)
    robot_obs = obs["robot_obs"]

    return {
        "observation/image": static_img,
        "observation/wrist_image": gripper_img,
        "observation/state": np.asarray(robot_obs, dtype=np.float32),
    }


def print_and_save(results, sequences, log_dir, epoch=None):
    """Print and save evaluation results."""
    log_dir = pathlib.Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    current_data = {}
    print(f"\nResults for Epoch {epoch}:")
    avg_seq_len = np.mean(results)
    chain_sr = {i + 1: sr for i, sr in enumerate(count_success(results))}
    print(f"Average successful sequence length: {avg_seq_len:.3f}")
    print("Success rates for i instructions in a row:")
    for i, sr in chain_sr.items():
        print(f"{i}: {sr * 100:.1f}%")

    cnt_success = Counter()
    cnt_fail = Counter()

    for result, (_, sequence) in zip(results, sequences):
        for successful_tasks in sequence[:result]:
            cnt_success[successful_tasks] += 1
        if result < len(sequence):
            failed_task = sequence[result]
            cnt_fail[failed_task] += 1

    total = cnt_success + cnt_fail
    task_info = {}
    for task in total:
        task_info[task] = {"success": cnt_success[task], "total": total[task]}
        print(f"{task}: {cnt_success[task]} / {total[task]} |  SR: {cnt_success[task] / total[task] * 100:.1f}%")

    # Convert sequences to serializable format
    sequences_data = []
    for result, (initial_state, sequence) in zip(results, sequences):
        sequences_data.append({"result": int(result), "initial_state": initial_state, "sequence": sequence})

    data = {
        "avg_seq_len": float(avg_seq_len),
        "chain_sr": {str(k): v for k, v in chain_sr.items()},
        "task_info": task_info,
        "results": [int(r) for r in results],
        "sequences": sequences_data,
    }
    current_data[str(epoch)] = data

    # Load previous data and merge
    results_path = log_dir / "results.json"
    previous_data = {}
    try:
        with open(results_path, "r") as file:
            previous_data = json.load(file)
    except FileNotFoundError:
        pass

    json_data = {**previous_data, **current_data}
    with open(results_path, "w") as file:
        json.dump(json_data, file, indent=2)

    print(f"\nResults saved to: {results_path}")
    if json_data:
        best_epoch = max(json_data, key=lambda x: json_data[x]["avg_seq_len"])
        print(
            f"Best model: epoch {best_epoch} "
            f"with average sequences length of {json_data[best_epoch]['avg_seq_len']:.3f}"
        )

    return json_data


def rollout(
    env,
    client: _websocket_client_policy.WebsocketClientPolicy,
    task_oracle,
    subtask: str,
    val_annotations: dict,
    resize_size: int,
    replan_steps: int,
    ep_len: int,
    debug: bool = False,
):
    """
    Run the actual rollout on one subtask (which is one natural language instruction).

    Returns:
        bool: True if task was completed successfully, False otherwise
    """
    if debug:
        print(f"{subtask} ", end="", flush=True)
        time.sleep(0.5)

    obs = env.get_obs()

    # Get language annotation for subtask
    lang_annotation = val_annotations[subtask][0]

    start_info = env.get_info()
    action_plan = collections.deque()

    for step in range(ep_len):
        # Format observation for policy
        formatted_obs = format_observation(obs, resize_size)
        formatted_obs["prompt"] = lang_annotation

        # Get action from policy (replan when action queue is empty)
        if not action_plan:
            result = client.infer(formatted_obs)
            action_chunk = result["actions"].copy()  # .copy() ensures array is writeable for CALVIN
            action_plan.extend(action_chunk[:replan_steps])

        action = action_plan.popleft()
        # if gripper > 0, gripper = 1, if gripper < 0, gripper = -1
        action[-1] = 1 if action[-1] > 0 else -1

        # Step environment
        obs, _, _, current_info = env.step(action)

        # Check if current step solves the task
        current_task_info = task_oracle.get_task_info_for_set(start_info, current_info, {subtask})
        if len(current_task_info) > 0:
            if debug:
                print("\033[92msuccess\033[0m ", end="", flush=True)
            return True

    if debug:
        print("\033[91mfail\033[0m ", end="", flush=True)
    return False


def evaluate_sequence(
    env,
    client: _websocket_client_policy.WebsocketClientPolicy,
    task_oracle,
    initial_state: dict,
    eval_sequence: list,
    val_annotations: dict,
    resize_size: int,
    replan_steps: int,
    ep_len: int,
    debug: bool = False,
):
    """
    Evaluates a sequence of language instructions.

    Returns:
        int: Number of successfully completed subtasks in the sequence
    """
    # Reset environment to initial state
    robot_obs, scene_obs = get_env_state_for_initial_condition(initial_state)
    env.reset(robot_obs=robot_obs, scene_obs=scene_obs)

    success_counter = 0
    if debug:
        time.sleep(1)
        print()
        print(f"Evaluating sequence: {' -> '.join(eval_sequence)}")
        print("Subtask: ", end="", flush=True)

    for subtask in eval_sequence:
        success = rollout(
            env,
            client,
            task_oracle,
            subtask,
            val_annotations,
            resize_size,
            replan_steps,
            ep_len,
            debug,
        )
        if success:
            success_counter += 1
        else:
            return success_counter

    return success_counter


def eval_calvin(args: Args) -> None:
    """Main evaluation function for CALVIN benchmark."""
    np.random.seed(args.seed)

    logging.info(f"Dataset path: {args.dataset_path}")
    logging.info(f"Connecting to policy server at {args.host}:{args.port}")

    # Load task oracle and annotations from config files
    conf_dir = pathlib.Path(__file__).absolute().parent / "conf"
    task_cfg = OmegaConf.load(conf_dir / "callbacks/rollout/tasks/new_playtable_tasks.yaml")
    task_oracle = hydra.utils.instantiate(task_cfg)
    val_annotations = OmegaConf.load(conf_dir / "annotations/new_playtable_validation.yaml")

    # Get log directory
    eval_log_dir = get_log_dir(args.eval_log_dir)

    # Load evaluation sequences
    eval_seq_path = pathlib.Path(__file__).absolute().parent / f"eval_seq_{args.eval_seq_len}.json"
    if not eval_seq_path.exists():
        raise FileNotFoundError(
            f"Evaluation sequence file not found: {eval_seq_path}\n"
            f"Available options: eval_seq_50.json, eval_seq_100.json, eval_seq_1000.json"
        )

    with open(eval_seq_path, "r") as f:
        eval_sequences = json.load(f)

    logging.info(f"Loaded {len(eval_sequences)} evaluation sequences from {eval_seq_path}")

    # Initialize environment
    logging.info("Initializing CALVIN environment...")
    env = make_env(args.dataset_path, args.split)

    # Connect to policy server
    logging.info(f"Connecting to policy server at {args.host}:{args.port}...")
    client = _websocket_client_policy.WebsocketClientPolicy(args.host, args.port)

    # Run evaluation
    results = []

    # Store original sequences for result aggregation
    original_sequences = eval_sequences.copy()

    if not args.debug:
        eval_sequences = tqdm(eval_sequences, position=0, leave=True, desc="Evaluating")

    for initial_state, eval_sequence in eval_sequences:
        result = evaluate_sequence(
            env,
            client,
            task_oracle,
            initial_state,
            eval_sequence,
            val_annotations,
            args.resize_size,
            args.replan_steps,
            args.ep_len,
            args.debug,
        )
        results.append(result)

        if not args.debug and hasattr(eval_sequences, "set_description"):
            # Update progress bar with current success rates
            eval_sequences.set_description(
                " ".join([f"{i + 1}/5: {v * 100:.1f}%" for i, v in enumerate(count_success(results))])
            )

    # Print and save final results
    print_and_save(results, original_sequences, eval_log_dir, epoch="pi05_calvin")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    args = tyro.cli(Args)
    eval_calvin(args)

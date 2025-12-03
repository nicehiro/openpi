"""
Convert CALVIN dataset to LeRobot v2 format.

This script converts the CALVIN dataset to LeRobot v2 format compatible with
OpenPI's LeRobotCalvinDataConfig training configuration.

Usage:
    uv run examples/calvin/to_lerobot_v2.py --calvin_data_path /path/to/calvin/data

To push to Hugging Face Hub:
    uv run examples/calvin/to_lerobot_v2.py --calvin_data_path /path/to/calvin/data --push_to_hub --repo_id your_username/calvin-lerobot
"""

import multiprocessing
import os
import shutil
from dataclasses import dataclass
from functools import partial
from pathlib import Path

import numpy as np
from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME, LeRobotDataset
from tqdm import tqdm
import tyro


@dataclass
class Args:
    """Arguments for CALVIN to LeRobot conversion."""

    calvin_data_path: str
    """Path to the CALVIN dataset directory."""

    repo_id: str = "calvin_lerobot"
    """Repository ID for the output dataset (also used for Hugging Face Hub)."""

    push_to_hub: bool = False
    """Whether to push the dataset to Hugging Face Hub."""

    debug: bool = False
    """Debug mode: only process a small subset of episodes."""

    chunk_size: int = 100
    """Number of episodes to process in each parallel batch (controls memory usage)."""


def process_episode(
    episode: tuple,
    data_path: Path,
    split: str,
) -> list[dict]:
    """Extract frames from an episode without writing to dataset.

    This function is designed to be used with multiprocessing.Pool.
    It returns a list of frame dictionaries that can later be added to the dataset.
    """
    ann, task, index_range = episode[0], episode[1], episode[2]

    # Language instruction combining task and annotation
    language_instruction = f"{task}: {ann}"

    frames = []
    for step in tqdm(
        range(index_range[0], index_range[1] + 1),
        leave=False,
        desc=f"Processing Ep {index_range[0]}-{index_range[1]}",
    ):
        # Load data for current step
        step_file = data_path / split / f"episode_{str(step).zfill(7)}.npz"
        if not step_file.exists():
            raise FileNotFoundError(f"Invalid data path: {step_file}")

        total_data = np.load(step_file)
        rgb_static = total_data["rgb_static"]  # uint8
        rgb_gripper = total_data["rgb_gripper"]  # uint8
        robot_obs = total_data["robot_obs"]  # float64

        # Get action from next step's rel_actions
        if step < index_range[1]:
            next_file = data_path / split / f"episode_{str(step + 1).zfill(7)}.npz"
            actions = np.load(next_file)["rel_actions"]
        else:
            # For the last step, use zero action
            actions = np.zeros(7, dtype=np.float32)

        # Collect frame data
        frames.append({
            "observation.images.top": rgb_static,
            "observation.images.wrist": rgb_gripper,
            "observation.state": robot_obs.astype(np.float32),
            "action": actions.astype(np.float32),
            "task": language_instruction,
        })

    return frames


def build_lerobot_dataset(args: Args) -> LeRobotDataset:
    """Build a LeRobotDataset from CALVIN data."""
    data_path = Path(args.calvin_data_path)

    # Clean up any existing dataset in the output directory
    output_path = HF_LEROBOT_HOME / args.repo_id
    if output_path.exists():
        shutil.rmtree(output_path)

    # Create LeRobot dataset with features matching LeRobotCalvinDataConfig expectations
    dataset = LeRobotDataset.create(
        repo_id=args.repo_id,
        robot_type="franka_emika",
        fps=10,  # CALVIN control frequency
        features={
            "observation.images.top": {
                "dtype": "image",
                "shape": (200, 200, 3),  # CALVIN image size
                "names": ["height", "width", "channel"],
            },
            "observation.images.wrist": {
                "dtype": "image",
                "shape": (84, 84, 3),  # CALVIN image size
                "names": ["height", "width", "channel"],
            },
            "observation.state": {
                "dtype": "float32",
                "shape": (15,),  # CALVIN robot observation dimension
                "names": ["state"],
            },
            "action": {
                "dtype": "float32",
                "shape": (7,),  # CALVIN action dimension (6D arm + 1D gripper)
                "names": ["action"],
            },
        },
        image_writer_threads=10,
        image_writer_processes=0,  # Use threading only to avoid multiprocessing conflicts
    )

    # Process training and validation splits
    for split in ["training", "validation"]:
        print(f"\n=== Processing {split} split ===")
        split_path = data_path / split
        if not split_path.exists():
            print(f"Skipping {split} split (not found at {split_path})")
            continue

        # Load language annotations
        lang_info = split_path / "lang_annotations" / "auto_lang_ann.npy"
        if not lang_info.exists():
            print(f"Skipping {split} split (no language annotations at {lang_info})")
            continue

        print(f"Loading language annotations from {lang_info}...")
        ann_data = np.load(lang_info, allow_pickle=True).item()
        lang_ann = ann_data["language"]["ann"]
        lang_task = ann_data["language"]["task"]
        lang_index = ann_data["info"]["indx"]

        print(f"Found {len(lang_ann)} episodes in {split} split")

        # Create partial function for processing
        partial_process = partial(
            process_episode,
            data_path=data_path,
            split=split,
        )

        # Prepare episode list
        episodes = list(zip(lang_ann, lang_task, lang_index))

        if not args.debug:
            # Process in chunks to limit memory usage
            total_chunks = (len(episodes) + args.chunk_size - 1) // args.chunk_size
            for chunk_idx, chunk_start in enumerate(range(0, len(episodes), args.chunk_size)):
                chunk = episodes[chunk_start:chunk_start + args.chunk_size]
                print(f"\nProcessing chunk {chunk_idx + 1}/{total_chunks} ({len(chunk)} episodes)...")

                # Extract episode data in parallel
                with multiprocessing.Pool(processes=os.cpu_count()) as pool:
                    chunk_results = pool.map(partial_process, chunk)

                # Write chunk to dataset immediately (frees memory after each chunk)
                for episode_frames in tqdm(chunk_results, desc="Writing episodes"):
                    for frame in episode_frames:
                        dataset.add_frame(frame)
                    dataset.save_episode()
        else:
            # Debug mode: process only the last episode
            if lang_ann:
                i = len(lang_ann) - 1
                results = partial_process(
                    (lang_ann[i], lang_task[i], lang_index[i])
                )
                if results:
                    for frame in results:
                        dataset.add_frame(frame)
                    dataset.save_episode()
            else:
                print(f"Warning: No episodes found for split {split} in debug mode.")

    return dataset


def validate(repo_id: str) -> None:
    """Validate the generated dataset."""
    dataset = LeRobotDataset(repo_id=repo_id)
    print("Dataset validated successfully!")
    print(f"  Number of episodes: {dataset.num_episodes}")
    print(f"  Number of frames: {dataset.num_frames}")
    print(f"  Features: {list(dataset.features.keys())}")


def main(args: Args) -> None:
    """Main entry point."""
    print(f"Converting CALVIN data from: {args.calvin_data_path}")
    print(f"Output repo_id: {args.repo_id}")
    print(f"Debug mode: {args.debug}")
    print()

    # Check data path exists
    data_path = Path(args.calvin_data_path)
    if not data_path.exists():
        print(f"ERROR: Data path does not exist: {data_path}")
        return

    dataset = build_lerobot_dataset(args)

    # Validate the dataset
    validate(args.repo_id)

    # Optionally push to Hugging Face Hub
    if args.push_to_hub:
        print("Pushing to Hugging Face Hub...")
        dataset.push_to_hub(
            tags=["calvin", "franka", "simulation"],
            private=False,
            push_videos=True,
            license="mit",
        )
        print(f"Dataset pushed to: https://huggingface.co/datasets/{args.repo_id}")


if __name__ == "__main__":
    main(tyro.cli(Args))

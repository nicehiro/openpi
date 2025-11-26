# CALVIN Benchmark Training and Evaluation

This example provides training and evaluation scripts for Pi0/Pi0.5 models on the CALVIN benchmark for language-conditioned robot manipulation.

## Training

### 1. Compute Normalization Stats

```bash
uv run scripts/compute_norm_stats.py --config-name pi05_calvin
```

### 2. Run Training

**JAX (auto-detects all GPUs):**
```bash
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py pi05_calvin --exp-name=my_calvin_exp
```

To select specific GPUs:
```bash
CUDA_VISIBLE_DEVICES=0,1 XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py pi05_calvin --exp-name=my_calvin_exp
```

**PyTorch (multi-GPU):**
```bash
uv run torchrun --standalone --nnodes=1 --nproc_per_node=<num_gpus> \
    scripts/train_pytorch.py pi05_calvin --exp_name my_calvin_exp
```

## Evaluation

### 1. Start Policy Server

```bash
uv run scripts/serve_policy.py policy:checkpoint \
    --policy.config=pi05_calvin \
    --policy.dir=/path/to/checkpoint
```

### 2. Run Evaluation

```bash
uv run examples/calvin/main.py \
    --dataset_path=/path/to/calvin_dataset \
    --eval_seq_len=50
```

> **Note:** If using a local proxy, bypass it for localhost connections:
> ```bash
> export no_proxy=localhost,127.0.0.1,0.0.0.0
> export NO_PROXY=localhost,127.0.0.1,0.0.0.0
> ```
>

**Options:**
- `--eval_seq_len`: Number of sequences to evaluate (50, 100, or 1000)
- `--eval_log_dir`: Output directory for results (default: `data/calvin/eval_logs`)
- `--debug`: Enable verbose output

## Output

Results are saved to `eval_log_dir/results.json` with:
- `avg_seq_len`: Average consecutive tasks completed (0-5)
- `chain_sr`: Success rate for completing 1-5 tasks in a row
- `task_info`: Per-task success/failure counts

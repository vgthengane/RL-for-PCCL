# RL-for-PCCL

## Toy parity experiment (RL's Razor)

This repository now contains a standalone PyTorch script that reproduces the
core empirical findings from *RL’s Razor* in a minimal setting. It pre-trains a
wide MLP on Fashion-MNIST (prior task) and then repeatedly fine-tunes it on
three synthetic MNIST tasks (parity, mod-3, and high/low) while tracking both
standard (SFT) and on-policy (REINFORCE-style) updates. After each task the
script reports:

1. New-task (parity) accuracy.
2. Prior-task (Fashion) accuracy to quantify forgetting.
3. KL divergence from the base model measured over the new-task inputs.

### Quick start

```bash
pip install -r requirements.txt
python toy_rl_razor.py
```

You can tweak `--task-epochs`, `--batch-size`, `--hidden`, and `--methods`
(`sft`, `rl`, or both) to control the runtime/accuracy trade-offs and decide
whether to run only the supervised or RL pathways (the default runs both). The
script prints the new-task meta-accuracy (e.g., parity correctness), the prior
task accuracy on Fashion-MNIST, and the KL shift from the base model so you can
compare how the selected methods behave as the agent moves through all three
tasks sequentially.

For convenience, run `./run_continual_tasks.sh` (after `chmod +x`) to execute
the experiment twice: once with `--methods sft` and once with `--methods rl`,
while preserving other CLI options you provide.

### Extending to point-cloud continual learning

1. Replace the MNIST/Fashion datasets with a point-cloud dataset that exposes a
   prior skill (e.g., object detection, alignment, or reconstruction) and define
   three downstream tasks that can be sequenced (parity-style binary labels,
   multi-class discretizations, etc.).
2. Keep the same dual-headed evaluation: use the prior task head (trained on your
   point-cloud benchmark) to measure forgetting, and expose a discrete policy
   distribution for the new task so that both SFT and RL can be applied.
3. Reuse the evaluation helpers (`evaluate_task`, `evaluate_fashion`, `compute_kl`)
   on your new inputs to verify whether KL divergence on the new task still
   predicts forgetting as the agent traverses all three tasks.

Those steps will let you confirm whether the KL-based forgetting law carries
over to your point-cloud scenario before integrating more complex RL pipelines.

## Continual LwF (PointNet ModelNet)

`train_lwf.py` now reuses PointNet’s official `ModelNetDataset`. It expects a
ModelNet40 directory with the `train.txt`, `test.txt`, and `.ply` files that the
PointNet utilities use (you can follow the instructions in `PointNet/utils/` to
prepare those data). Point the script to that directory with

```bash
./train_lwf.sh --modelnet-root /path/to/modelnet40
```

The script still performs 10 tasks (4 classes per task) with LwF/KD following
PyCIL’s schedule, and you can pass additional PyTorch CLI args via the wrapper
thanks to the `PYTHON`/`train_lwf.sh` launcher. If you want to override `batch`
size, learning rate, or epochs, edit the constants at the top of `train_lwf.py`
or extend the argument parser as needed.

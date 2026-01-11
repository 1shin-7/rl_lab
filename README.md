# DQN CartPole CLI

A simple Deep Reinforcement Learning project using TensorFlow to solve the CartPole-v0 environment.

## Requirements

- Python 3.12+
- `uv` package manager (recommended)

## Installation

```bash
uv sync
```

## Usage

This project provides a CLI with `train` and `infer` commands.

### Training

To train the agent:

```bash
# Run with default settings (200 episodes)
uv run dqn train

# Run with custom number of episodes
uv run dqn train --episodes 500
```

This will save:
- Model weights to `dqn_cartpole_model.pth`
- Training plot to `training_plot.png`
- Logs to `training.log`

### Inference

To run the trained agent:

```bash
# Run inference for 5 episodes
uv run dqn infer --episodes 5

# Run with rendering (visualize the cartpole)
uv run dqn infer --episodes 5 --render
```

### Global Options

- `--debug`: Enable verbose logging.

## Project Structure

- `src/dqn_cartpole/`: Source code.
    - `agent.py`: DQN Agent implementation.
    - `train.py`: Training loop.
    - `infer.py`: Inference loop.
    - `config.py`: Hyperparameters.
- `pyproject.toml`: Dependencies.

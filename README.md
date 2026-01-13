# Deep Reinforcement Learning Lab (DRL Lab)

<div align="center">

![Socialify](https://socialify.git.ci/1shin-7/rl_lab/image?description=1&font=Inter&language=1&name=1&owner=1&pattern=Plus&theme=Auto)

[![Python](https://img.shields.io/badge/Python-3.12+-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-GPL%20v2-green.svg)](LICENSE)

[English](README.md) | [简体中文](docs/README_zh.md)

</div>

> "Bridging theory and practice: An exploration from Markov Decision Processes to Rainbow DQN."

`drl_lab` is a modular, extensible Deep Reinforcement Learning (DRL) experimental platform designed to provide out-of-the-box implementations of classic control tasks with a modern debugging experience.

## ✨ Features

*   **Modern Architecture**: Built on **PyTorch** and **Gymnasium**, featuring a modular design for `Agent`, `Task`, and `Trainer`.
*   **Rainbow DQN Integration**:
    *   ✅ **Double DQN (DDQN)**: Mitigates Q-value overestimation.
    *   ✅ **Dueling Networks**: Separates value and advantage streams for faster convergence.
    *   ✅ **Huber Loss**: Gradient clipping for stability.
*   **Interactive TUI**: Terminal User Interface powered by **Textual**, enabling **real-time visualization** of training and inference (Braille animations, status dashboards, live logs).
*   **Developer Friendly**: Lifecycle hooks (`pre_training`, `on_step`, etc.) and a standardized `BaseTask` interface.
*   **Practical Optimizations**: Includes Reward Shaping for sparse reward tasks like CartPole.

## 🚀 Quickstart

### Installation

This project uses `uv` for package management:

```bash
# Clone the repository
git clone https://github.com/1shin-7/rl_lab.git
cd rl_lab

# Sync dependencies
uv sync
```

### Training

Start training the CartPole task with TUI visualization enabled:

```bash
uv run rlab train cartpole --visual --episodes 500
```

### Inference

Run inference using a trained model:

```bash
uv run rlab infer cartpole --visual --weight outputs/cartpole.pth
```

### Clean

Remove generated models and plots for a task:

```bash
uv run rlab clean cartpole
```

## 📚 Documentation

*   [Development Overview](docs/development/task.md): Architecture, Tasks, and Hooks.
*   [TUI Guide](docs/development/tui.md): Visualization and UI design.
*   [CLI Reference](docs/commands.md): Detailed documentation for all CLI commands.

## 🤝 Credits

*   **[PyTorch](https://pytorch.org/)**: The backbone of the entire project, enabling flexible and efficient deep learning.
*   [Gymnasium](https://gymnasium.farama.org/): Standardized RL environment API.
*   UI/UX: [Textual](https://textual.textualize.io/), [Loguru](https://github.com/Delgan/loguru), and [Rich](https://github.com/Textualize/rich) were used to provide a convenient debugging interface.
*   **[Gemini](https://gemini.google.com/)**: For assisting in problem-solving throughout the internship.
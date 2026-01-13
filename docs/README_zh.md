# Deep Reinforcement Learning Lab (DRL Lab)

<div align="center">

[![Python](https://img.shields.io/badge/Python-3.12+-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-GPL%20v2-green.svg)](LICENSE)

[English](../README.md) | [简体中文](README_zh.md)

</div>

> "理论与实践的桥梁：从马尔可夫决策过程到 Rainbow DQN 的探索之旅。"

`drl_lab` 是一个模块化、易扩展的深度强化学习（DRL）实验平台，旨在提供开箱即用的经典控制任务实现与现代化的调试体验。

## ✨ 特性 (Features)

*   **现代化架构**: 基于 **PyTorch** 和 **Gymnasium** 构建，模块化设计 `Agent`、`Task` 和 `Trainer`。
*   **Rainbow DQN 集成**:
    *   ✅ **Double DQN (DDQN)**: 消除 Q 值过估计。
    *   ✅ **Dueling Networks**: 分离价值与优势流，加速收敛。
    *   ✅ **Huber Loss**: 梯度裁剪与稳定性优化。
*   **交互式 TUI**: 使用 **Textual** 构建的终端用户界面，支持训练与推理过程的**实时可视化**（Braille 动画、状态仪表盘、实时日志）。
*   **开发友好**: 提供生命周期 Hook (`pre_training`, `on_step` 等) 和标准化的 `BaseTask` 接口。
*   **实战优化**: 针对 CartPole 等任务实现了稀疏奖励的 Reward Shaping。

## 🚀 快速开始 (Quickstart)

### 安装

本项目使用 `uv` 进行包管理：

```bash
# 克隆仓库
git clone https://github.com/1shin-7/rl_lab.git
cd rl_lab

# 同步依赖
uv sync
```

### 训练 (Training)

启动 CartPole 任务训练，并开启 TUI 可视化模式：

```bash
uv run rlab train cartpole --visual --episodes 500
```

### 推理 (Inference)

加载训练好的模型进行推理演示：

```bash
uv run rlab infer cartpole --visual --weight outputs/cartpole.pth
```

### 清理 (Clean)

一键清除任务产生的模型和图表：

```bash
uv run rlab clean cartpole
```

## 📚 文档

*   [开发概述 (Task & Hooks)](development/task.md): 了解架构、任务定义与钩子机制。
*   [TUI 指南](development/tui.md): 可视化设计与 UI 开发。
*   [命令参考 (CLI Reference)](commands.md): 所有 CLI 命令的详细说明。

## 🤝 致谢 (Credits)

*   **[PyTorch](https://pytorch.org/)**: 整个项目的核心功臣，提供了灵活高效的深度学习支撑。
*   [Gymnasium](https://gymnasium.farama.org/): 标准化的强化学习环境接口。
*   UI/体验: 使用了 [Textual](https://textual.textualize.io/)、[Loguru](https://github.com/Delgan/loguru) 和 [Rich](https://github.com/Textualize/rich) 库来辅助构建更便利的可视化调试界面。
*   **[Gemini](https://gemini.google.com/)**: 感谢 Gemini 在实习过程中协助我解决技术难题。

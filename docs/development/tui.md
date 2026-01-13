# TUI Visualization Guide

`rl_lab` leverages **Textual** to provide a high-performance, real-time visualization of the reinforcement learning process directly in the terminal.

## Design Philosophy

The TUI is decoupled from the Task logic. A task provides a "Renderable" interface which the App mounts into its layout.

### Layout Structure
The standard `compose_view()` in `BaseTaskTUI` implements a two-part layout:
1.  **Header**: A compact, docked-top bar showing Episode, Step, Device, and Reward.
2.  **Content**: A centered container (`1fr`) that hosts the task-specific main widget.

## Implementing a Custom TUI

1.  **Inherit from `BaseTaskTUI`**: Define how your state should be rendered.
2.  **Define the Main Widget**: Use Textual widgets (e.g., `Static`, `Label`, or custom `Container`) to draw the environment.
3.  **Update Logic**:
    *   `update_state(state, info)`: Updates the main visualization (e.g., moving the cart in CartPole).
    *   `update_stats(episode, step, reward)`: Automatically updates the header (rarely needs overriding).

### Braille Rendering
For high-density character graphics (like CartPole's pole), use the `BrailleCanvas` utility located in `tasks/cartpole/tui.py`. It allows 2x4 dot-matrix drawing per character cell.

## Logging Interaction

Logging in TUI mode is non-blocking and thread-safe:
*   **Structured Sink**: Logs are intercepted via a Loguru sink in the TUI thread.
*   **Manual Construction**: To ensure perfect colors and formatting, the TUI manually builds `Rich.Text` objects from the raw `record` data.
*   **ANSI Support**: Avoid relying on raw ANSI透传; use `Text.from_ansi()` if strictly necessary.

## References

*   [Textual Official Documentation](https://textual.textualize.io/): The definitive guide for widgets, events, and CSS.
*   [Rich Text](https://rich.readthedocs.io/): For advanced terminal formatting inside widgets.

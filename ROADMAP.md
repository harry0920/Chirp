# Roadmap

## Planned Features

- **Linux support** — Bring Chirp to Linux with full feature parity.
- **Overlay on active monitor** — Recording pill should appear on the monitor being used, not always primary. Use `cursorPosition()` + `availableMonitors()` with DPI-aware positioning.
- **Multiple paste methods** — Let users choose Ctrl+Shift+V (terminals), Shift+Insert (legacy), or direct typing (no clipboard) instead of just Ctrl+V.
- **Transcription coordinator** — Move recording lifecycle from frontend to a Rust mpsc channel state machine. Serializes hotkey events, prevents race conditions from rapid presses, adds stuck-state recovery.
- **Tap-to-talk mode** — Tap hotkey to start, tap to stop. RMS-based SmoothedVad with onset/hangover/prefill auto-stops recording after sustained silence.
- **Tray icon quick settings popup** — Right-click system tray for quick access to settings (like Whisper Flow).
- **Auto-learn corrections** — When user manually edits a transcription, learn the correction for next time.
- **Cmd+Enter to submit feedback** — Keyboard shortcut for the feedback form.
- **F13+ function key support** — Extended function keys as hotkey options.
- **Moveable/resizable overlay** — Draggable recording pill with size and color options.
- **Arabic language support** — Requires additional ASR model.
- **LLM download progress after onboarding** — Show model download status on Home page when skipping onboarding.
- **Dark mode for content area** — Currently only sidebar is dark.

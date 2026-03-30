# Known Issues

## Mac

- **Check for updates is broken** — The update check mechanism does not work correctly on macOS.
- **Launch on startup is glitchy** — Enabling "launch on startup" on macOS results in unreliable behavior.
- **Overlay not visible over fullscreen apps** — The recording overlay pill does not appear on top of macOS fullscreen/Space apps despite window level configuration.

## Windows

- **Tray menu quit button may not appear** — The "Quit Chirp" menu item is defined in code but may not render on some Windows configurations (likely a Tauri rendering bug).

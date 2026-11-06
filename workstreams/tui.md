# TUI: a terminal controller for AudioViz

> **Status:** active · **Branch:** `agent/tui` → `origin/tui/lenovo` · **Updated:** 2026-11-05
> **Now:** tile hiding landed (`dd`, `u`, `T`); render axis only, hidden tiles persist.
> **Next:** serve low-res previews over the deck socket (attached tiles show "waiting for data"), then paged knobs from `main.py` config sections.
> **Blocked:** none. Integration with `main` is deferred: `main` has moved to active-inference rendering that changes the canvas dynamics.
> **Resume:** `copilot --resume efa7117b-d1b5-4bdb-bf79-7d9c8036c52e`
> **Run:** `uv run python main.py` (`--no-tui`: window only)

**Essence.** A Vim-modal Textual app that works as a DJ deck: `main.py` renders in its
standalone Qt window and serves a deck socket; the TUI attaches and turns the window's knobs.

**Terminal resizing.** Textual resize events refit floating tiles independently of
preview polling. Shared edges scale from a stable reference; manual edits rebase
that reference. Knobs keep their height and bottom offset. Zoom fits the current
viewport while retaining the scaled underlying placement and stack depth. Saved
layouts include their viewport size; loading one refits it to the current terminal.
No window-manager integration is needed.

**Startup.** `main.py` opens one idle TUI first, before any Qt window or local
graph. Usable saved audio devices auto-start the window; missing or invalid
selections open the existing Vim checklists. Startup remains responsive while
the child initializes, and errors leave device selection and retry available.
The same TUI then attaches to the window; quitting terminates its owned child.
Window-only `--no-tui` retains terminal-based setup.

**Live readback.** Device and window tiles display state reported by the Qt
process. Deck exchanges publish their state in request order. A timed-out or
closed connection is discarded; the next poll reconnects and reads current
state. Edits with an unknown outcome are not replayed.

**Logs and files.** Enter the log with Enter: j/k scroll, Ctrl-f/b page, gg goes
to the start, and G returns to the tail and follows new entries. Browsing keeps
the viewport fixed while messages arrive; Enter/Esc leaves log navigation.
`:w [name|path]` and `:save [name|path]` write one configuration: runtime values
and TUI layout/settings. Bare names resolve in the application's `config/`
directory; the default is `config/deck.json`. Explicit paths and quoted spaces
are supported. `:e [name|path]` applies that configuration, including live
attached runtime values. Automatic UI cache restoration never writes to the
attached runtime. Tab/Shift-Tab cycle matching JSON paths and directories;
save/load results remain in the bottom status bar.

Configurations are JSON, with typed runtime values under `mixer`. They record
their backend because standalone and Qt wave-speed units differ. Existing flat
deck configs still load. `:w` does not create images or recordings: `:shot` takes
a screenshot, and `:record` defaults to the project's `recordings/` directory.
Project configs are available for version control; device-specific
`config/audio_devices.json` remains local.

**Settings tabs.** Tab / Shift-Tab cycle Overview, Audio, Wave, Inference, Learning,
Sources and Interface; `:tab <name>` selects one directly. Overview retains
floating tiles. Settings tabs use a scrollable two-column grid: h/j/k/l or
w/b navigate, Enter enters a knob, h/l adjusts, and Enter/Esc leaves it. n/p or
Ctrl-f/b page through longer grids. These are views of the same live controls,
so switching tabs never starts another graph or resets learned state.

The attached settings include every writable Ripple Controls section:
source parameters, synthetic frequencies, inference steps/rates, learning and
clipping, excitation physics, canvas color scaling and conductance overlays.
The source metadata supplies ranges, steps, labels and tooltips; existing
callbacks remain responsible for updating the window's graph. `:reset` clears
field/belief state while retaining weights. Changing the standalone resolution
rebuilds the graph; reloading an unchanged resolution does not.

Terminal-owned launches hide the legacy Qt controls button. `--no-tui` retains
that panel. Plot/scene interaction remains in the visual windows. In attached
mode, device and audio-model selection are in Audio; Qt plays input audio and
its output volume belongs to the system mixer. Standalone mode retains the
input/prediction/error listening controls.
There is one renderer. The TUI shows only low-res previews, and terminal pixel protocols and
window-manager tiling are out of scope.

**Present.**
- Launch: `main.py` starts the window as a child and attaches the TUI; quitting the TUI closes the window.
- Tiles: subgraph views, mixer and settings knobs, log. They layer, move, zoom and group (`v`), and refit to the terminal size. `Esc` rises to a group, `Enter` dives back; the mixer group pages with `n`/`p`.
- Devices: Enter opens a checklist: one audio device per direction, up to two cameras. Configs store selections as lists and match device names across changed indices. `:w` / `:save` persist runtime controls and TUI preferences together.
- Hiding: `dd` hides the focused tile or group, `u` undoes, `T` lists tiles by group. Hidden tiles keep their placement and persist in `~/.cache/audioviz/state.json`.
- Keys are data in `src/audioviz/tui/keys.py`; `?` shows them and `~/.config/audioviz/keys.toml` rebinds them.
- Verified by pilot tests (`tests/tui`) and live tmux screenshots. Hiding has pilot coverage only.

**Open.**
- Hiding is render-only. Whether a hidden tile may also stop computing depends on who reads its output (`infer_joint` is ~86% of a step); undecided.
- Live checks outstanding: hiding layout and `T` list screenshots, `?` showing `d d`, knob edit-mode cue.
- `audio out` is silent with the Jabra selected (unverified cause). The sound source/sink design moves to the audio rework (cochlea workstream).
- Camera selection of several devices: multi-device workstream.
- About 1 in 4 full pilot runs fail one flaky test and pass on rerun.
- Undecided: git-ignoring `config/deck.json` and `outputs/`, evidence gain default, learning on/off.

**Future.** Nested groups, gaze-driven focus over a Unix socket, `t-K` sampling into saved state, theme-aware colormaps.

**Notes.**
- Deck protocol: JSON lines (`controls`, `nudge`, `set`, `select`, `save`), served on the Qt thread.
- `describe()["grid"]` is an `"RxC"` string.
- Drive it live: `python -m audioviz.tui.control keys <names> | run "<cmd>" | state | screenshot` (colleague's tool; `XDG_RUNTIME_DIR=/run/user/1000`).
- Tests: `uv run --group dev python -m pytest tests/tui -q`.
- Live-edit `views.py` or `halfblock.py` while running; a syntax error keeps the old code.

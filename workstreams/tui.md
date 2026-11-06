# TUI: a terminal controller for AudioViz

> **Status:** active · **Branch:** `agent/tui` → `origin/tui/lenovo` · **Updated:** 2026-11-05
> **Now:** tile hiding landed (`dd`, `u`, `T`); render axis only, hidden tiles persist.
> **Next:** serve low-res previews over the deck socket (attached tiles show "waiting for data"), then paged knobs from `main.py` config sections.
> **Blocked:** none. Integration with `main` is deferred: `main` has moved to active-inference rendering that changes the canvas dynamics.
> **Resume:** `copilot --resume efa7117b-d1b5-4bdb-bf79-7d9c8036c52e`
> **Run:** `uv run python main.py` (`--no-tui`: window only)

**Essence.** A Vim-modal Textual app that works as a DJ deck: `main.py` renders in its
standalone Qt window and serves a deck socket; the TUI attaches and turns the window's knobs.

**Startup.** `main.py` opens one idle TUI first, before any Qt window or local
graph. Usable saved audio devices auto-start the window; missing or invalid
selections open the existing Vim checklists. Startup remains responsive while
the child initializes, and errors leave device selection and retry available.
The same TUI then attaches to the window; quitting terminates its owned child.
Window-only `--no-tui` retains terminal-based setup.
There is one renderer. The TUI shows only low-res previews, and terminal pixel protocols and
window-manager tiling are out of scope.

**Present.**
- Launch: `main.py` starts the window as a child and attaches the TUI; quitting the TUI closes the window.
- Tiles: subgraph views, mixer and settings knobs, log. They layer, move, zoom and group (`v`), and refit to the terminal size. `Esc` rises to a group, `Enter` dives back; the mixer group pages with `n`/`p`.
- Devices: Enter on a device knob opens a checklist (capacity 1 today; protocol and UI are list-shaped). `:save` writes the knobs by name to `config/deck.json`.
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

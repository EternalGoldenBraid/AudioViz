# TUI: a terminal-native front end

**Essence.** AudioViz as a Vim-modal terminal app. Tiles are subgraph views of the
predictive-coding graph (waveform, spectrogram, prediction, error, canvas, hidden),
plus a mixer and a log. Tiles float on layers with opacity, can be lifted, moved and
zoomed, and render at a density that follows their size (half-block to braille).
Real microphone input drives the graph.

**Past.** Textual spike, then NORMAL/COMMAND modes, layered tiles, live audio,
mixer, fullscreen, resolution 96, themes (black first). Verified by headless
screenshots and by driving a live tmux pane.

**Present.** Live mic and camera by default (camera tile, camera drives the canvas), settings
and mixer tiles, Vim visual mode `v` to group tiles, autosave to `~/.cache/audioviz/state.json`,
`:w [name]` writes state + screenshot to `~/.config/audioviz/saves/`. Grid 128 runs slower than
realtime (~80 ms/step); the hidden tile sometimes waits for data; the canvas is flat.

**Learned.** Step time is ~86% predictive-coding inference (`infer_joint`), not wave propagation,
so the OpenGL/CuPy engine backends would not speed the TUI (and OpenGL has no observation
correction). Resolution is bounded by that readout.

**Live editing.** Edit `views.py` or `halfblock.py` while the app runs: tiles pick it up on the
next frame (`reloaded views` in the log); a syntax error keeps the old code live.

**Future.** Knobs as tiles and a node tree (a group is a node whose children are nodes; mixer and
settings become preset groups; viewer distance could pick the tree level). Speed up `infer_joint`.
`t-K` sampling of every signal (or a chosen subset) into the saved state; Gaze daemon over a Unix socket (position from the ray, layer depth from
viewer distance), attractor dynamics in the hidden view, theme-aware data colormaps.

**Restore.**

    copilot --resume efa7117b-d1b5-4bdb-bf79-7d9c8036c52e
    uv run --group tui --with-editable . python -m audioviz.tui.app --audio synthetic

Drive it live: `python -m audioviz.tui.control keys w j | run "theme amber" | state | screenshot`.

Keys: `w` focus, `z` zoom, `r` render, `H/J/K/L` move, `[ ]` depth, `:` commands.

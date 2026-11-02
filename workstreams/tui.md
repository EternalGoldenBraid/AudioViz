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

**Future.** `t-K` sampling of every signal (or a chosen subset) into the saved state; Gaze daemon over a Unix socket (position from the ray, layer depth from
viewer distance), attractor dynamics in the hidden view, theme-aware data colormaps.

**Restore.**

    copilot --resume efa7117b-d1b5-4bdb-bf79-7d9c8036c52e
    uv run --group tui --with-editable . python -m audioviz.tui.app --audio synthetic

Drive it live: `python -m audioviz.tui.control keys w j | run "theme amber" | state | screenshot`.

Keys: `w` focus, `z` zoom, `r` render, `H/J/K/L` move, `[ ]` depth, `:` commands.

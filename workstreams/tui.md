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

**Visual prediction.** `visual_prediction` shows what the model expects the canvas to be; `visual_error` is camera observation minus that (grey while `canvas feed` is off). Default layout is three rows: canvas, camera, visual prediction, visual error / spectrogram, waveform, prediction, error / hidden, log.

**Direction (2026-11-05).** The TUI is a controller, a DJ deck or pedalboard, and `main.py` keeps rendering in its standalone windows. One renderer: the full-quality views are the existing Qt windows; the TUI shows only selected low-res previews. Terminal pixel protocols (Kitty graphics) and window-manager tiling are out of scope: too platform-specific. So `main.py` launches as it does now and also serves a control socket; the deck connects to it, and its knobs are generated from one parameter registry derived from `RIPPLE_CONFIG` (sections field, dynamics, renderer, sources.audio, sources.pose, transforms, plotting), grouped and paged by section. The deck's own headless `GraphRuntime` becomes a standalone/test mode. Open: how the running app exposes live-settable parameters (setters per path), previews over the socket (`engine.get_field_preview()` is already a downsampled field), and audio/camera device switching from the deck.

**Next (2026-11-05).** Seen in a wide pane: tiles keep their start-up size and fill only the left corner, so layout must follow the terminal size.
- *Disable tiles* to save compute: a hidden tile is not rendered, and a view nobody shows is not computed (`infer_joint` is 86% of the step, so the audio/visual readout pathways also need on/off knobs).
- *Scaling* (done: placements refit to the terminal on resize, knobs keep their height and the bottom edge); later, group members scale before they page.
- *Sound*: speakers plugged in; `sound out` and `:devices` exist, verify audibility of the vocoder, then input/output device switching online.
- *Resolution* (done): `resolution N` or `RxC`, knob choices include `100x200` (the `main.py` grid) up to `480x640`; cells stay square. `describe()["grid"]` is now an `"RxC"` string.
- *All `main.py` tuning as knobs*, grouped by its sections (field, dynamics, renderer, sources.audio, sources.pose, plotting) as paged groups. Needs the supernode container and whole-view paging first.

**Live editing.** Edit `views.py` or `halfblock.py` while the app runs: tiles pick it up on the
next frame (`reloaded views` in the log); a syntax error keeps the old code live.

**Knobs.** Every mixer/settings control is its own tile (`-`/`+` or arrows turn directly; `Enter` enters a knob (double border, ◀ ▶, `EDIT` in the mode bar) so `h/l` slide it, `Enter`/`Esc` leave; otherwise `h/j/k/l` walk). `camera view` shows the tile, `canvas feed` drives the canvas; the device is released when both are off. Mixer and settings
are preset groups of knob tiles (group 1, group 2); regroup freely with `v`. `Esc` rises to the whole group (heavy accent on every member; `h/j/k/l` then step between top-level nodes, `H/J/K/L`, `[ ]`, `, .` act on the group); `Enter` dives back to single tiles. The mixer group is paged (6 knobs at a time): `n`/`p` turn the page, the page shows in each knob border; moving the group carries the slots with it. A group is still a link, not a container: it has no frame of its own yet.

**Open (end of session).** Knob edit mode is unit-tested but the live cue (double border, ◀ ▶, `EDIT`) was not confirmed in the running app: a live `Enter` on `knob-camera-view` left the mode at NORMAL. One test (`test_question_mark_lists_bindings_and_a_rebound_key_takes_effect`) failed in a full run and was being rerun; pilot tests are flaky about 1 in 4 runs under load. Next: expose `main.py` config sections (field, dynamics, renderer, sources.audio, sources.pose, plotting) as paged knob groups; a real group container so scaling and paging follow available space. Push is blocked until `gh auth login -h github.com`, then `git push -u origin agent/tui`.

**Future.** Nested groups (a group is a node whose children are nodes; viewer distance could pick
the tree level), a quieter group marker on knobs. Speed up `infer_joint`.
`t-K` sampling of every signal (or a chosen subset) into the saved state; Gaze daemon over a Unix socket (position from the ray, layer depth from
viewer distance), attractor dynamics in the hidden view, theme-aware data colormaps.

**Restore.**

    copilot --resume efa7117b-d1b5-4bdb-bf79-7d9c8036c52e
    uv run --group tui --with-editable . python -m audioviz.tui.app --audio synthetic

Drive it live: `python -m audioviz.tui.control keys w j | run "theme amber" | state | screenshot`.

Keys: `w` focus, `z` zoom, `r` render, `H/J/K/L` move, `[ ]` depth, `:` commands.

**Keys.** Bindings are data in `src/audioviz/tui/keys.py` (sections: navigate, arrange, group, view, knob, app).
`?` opens a sectioned overlay; `~/.config/audioviz/keys.toml` rebinds actions by name (`:keys write` creates the
full default file, `:keys reload` applies edits; a custom binding takes a key over from a default one, and
unknown actions are listed in the overlay). An in-app rebind UI can sit on the same file later.

# Live test agent: a small local agent that drives the TUI and reports

**Essence.** A lightweight local model (Ollama) that runs live checks against the running
AudioViz TUI through its control socket and reports pass/fail with evidence. It frees the
main agent from hand-polling keys and screenshots. It checks known behaviour; it does not
judge appearance or discover new bugs.

**Interface (exists).** `$XDG_RUNTIME_DIR/audioviz.sock`, JSON lines, ops `keys`, `run`,
`state`, `screenshot`. CLI: `python -m audioviz.tui.control keys <k..> | run "<cmd>" | state | screenshot [name]`.
`state` returns focus, mode, groups, placements, tile views, and every knob value
(`mixer` and `settings` maps).

**Launch the app under test.**
`tmux split-window -d -h -c "$PWD" -P -F '#{pane_id}' "uv run --group tui --with-editable . python -m audioviz.tui.app; read" > /tmp/av.pane`
Never redirect the app's stderr. Stop it with `tmux kill-pane -t $(cat /tmp/av.pane)`.
Use the real mic and camera unless a check is about a wave pattern.

**Known quirks for the driver.**
- Keys sent through `tmux send-keys` need literal text for symbols: `send-keys -l "-"`, not `minus`.
- The control socket's `keys` op takes Textual key names (`minus`, `h`, `escape`).
- Autosave state lives in `~/.cache/audioviz/state.json`. Delete it for a default layout.
- Wait ~12 s after launch before the first query.

**Step 1: scripted runner (no model).** A script that takes a key sequence plus assertions
on `state` and prints pass/fail with the state diff. Example check: focus `knob-camera-view`,
send `-`, assert `mixer["camera view"] == "off"` and `canvas feed` is still `on`.
It is useful with or without an agent.

**Step 2: Ollama agent.** A thin tool-calling loop over the runner and the control CLI.
The agent receives a plain-language check list, runs it, and returns a short report:
check, result, observed value. Open questions: which small model handles tool calls
reliably on this laptop, and how the PI harness extensions wire tools in. Read the PI
harness docs before choosing the glue.

**Status.** Step 1 done: `python -m audioviz.tui.checks [checks/tui.json]` launches a private app
(own socket via `AUDIOVIZ_SOCKET`, so a running app is untouched), runs the checks, kills the pane,
exits non-zero on failure. Check steps: `keys`, `run`, `focus`, `expect`.

**Fine-tuning.** Every step is traced to `outputs/traces/*.jsonl` (state diff, action, verdict).
These traces are the training set for the small agent: (check text, state, action) -> next action,
labelled by the scripted runner's verdicts. Later: continual learning from new traces. Eventually the
agent's own computational graph becomes a neighbour of the app's predictive-coding graph, and the
two join; keep the control socket's state/action vocabulary as the shared interface.

**Out of scope.** Visual judgement of screenshots, performance tuning, anything that
writes to git or pushes.

**Acceptance.**
1. One command runs the check list against a fresh app and exits non-zero on any failure.
2. The agent produces the same report from a plain-language task.
3. Starting and stopping the app leaves no stray tmux panes or sockets.

**Restore.** `copilot --resume efa7117b-d1b5-4bdb-bf79-7d9c8036c52e` (main agent's context),
plus `workstreams/tui.md` for the app itself.

# Copilot Instructions

1. **Distill the rule, not the incident.** When a design problem recurs, capture the underlying architectural boundary or invariant in a reusable form instead of writing an instruction that only describes the concrete example that triggered it.
2. **Keep runtime dependencies in the default install.** Declare directly imported application dependencies in `pyproject.toml` under `project.dependencies`, including dependencies also supplied transitively. Keep development tools in `dev` and hardware-specific backends optional. Update `uv.lock` in the same commit as dependency changes; run and verify through the current worktree's environment with `uv run`.

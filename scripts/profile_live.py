"""Profile the live app: python scripts/profile_live.py --backend gpu."""

from __future__ import annotations

import argparse
import cProfile
import io
from pathlib import Path
import pstats
import sys
import time


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("gpu", "numpy", "opengl"))
    parser.add_argument("--frames", type=int, default=120)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--output", type=Path, default=Path("/tmp/audioviz-profile.prof"))
    args = parser.parse_args()
    if args.frames < 1 or args.warmup < 0:
        parser.error("frames must be positive and warmup nonnegative")

    root = Path(__file__).resolve().parents[1]
    sys.path[:0] = [str(root), str(root / "src")]
    import main as application
    import numpy as np
    from PyQt5 import QtCore, QtWidgets

    if args.backend:
        application.RIPPLE_CONFIG["renderer"]["backend"] = args.backend
    print("Profiling backend:", application.RIPPLE_CONFIG["renderer"]["backend"], flush=True)

    profiler = cProfile.Profile()
    original = application.RippleWaveVisualizer.update_visualization
    durations = []
    count = 0
    started = None
    completed = False

    def update(widget):
        nonlocal count, started, completed
        if completed:
            return
        if count == 0:
            print("Resolution:", widget.resolution, flush=True)
        if count == args.warmup:
            started = time.perf_counter()
            profiler.enable()
        before = time.perf_counter()
        original(widget)
        if count >= args.warmup:
            durations.append(time.perf_counter() - before)
        count += 1
        if len(durations) == args.frames:
            profiler.disable()
            completed = True
            elapsed = time.perf_counter() - started
            milliseconds = np.asarray(durations) * 1000
            print(f"{len(durations)} frames in {elapsed:.2f}s: "
                  f"{len(durations) / elapsed:.2f} FPS; update ms "
                  f"median={np.median(milliseconds):.2f}, "
                  f"p95={np.percentile(milliseconds, 95):.2f}", flush=True)
            app = QtWidgets.QApplication.instance()
            def finish():
                app.closeAllWindows()
                app.quit()
            QtCore.QTimer.singleShot(0, finish)

    application.RippleWaveVisualizer.update_visualization = update
    try:
        application.main([])
    finally:
        profiler.disable()
        application.RippleWaveVisualizer.update_visualization = original
        if durations:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            profiler.dump_stats(str(args.output))
            report = io.StringIO()
            stats = pstats.Stats(profiler, stream=report).strip_dirs()
            stats.sort_stats("cumulative").print_stats(45)
            stats.sort_stats("tottime").print_stats(25)
            report_path = args.output.with_suffix(".txt")
            report_path.write_text(report.getvalue())
            print(report.getvalue())
            print(f"Saved {args.output} and {report_path}", flush=True)
        else:
            print("No frames captured; check application startup errors.", flush=True)


if __name__ == "__main__":
    main()

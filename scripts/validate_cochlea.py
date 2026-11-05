"""Reproduce the cochlea listening example and measure core processing throughput."""

import argparse
import json
from pathlib import Path
from time import perf_counter
import wave

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from audioviz.audio_processing.cochlea import Cochlea


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/cochlea"))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    sample_rate = 22050
    bank = Cochlea(sample_rate)
    time = np.arange(sample_rate * 2) / sample_rate
    # Two bank-tuned components expose inference without claiming broadband coverage.
    frequencies = bank.damped_omega[[5, 9]] / (2 * np.pi)
    raw = 0.25 * np.sin(2 * np.pi * frequencies[0] * time)
    raw += 0.15 * np.sin(2 * np.pi * frequencies[1] * time + 0.7)
    started = perf_counter()
    block = bank.process(raw)
    elapsed = perf_counter() - started
    tail = slice(sample_rate, None)
    ratio = np.mean(block.error[tail] ** 2) / np.mean(raw[tail] ** 2)
    metrics = {
        "sample_rate": sample_rate,
        "nodes": bank.config.nodes,
        "quality_factor": bank.config.quality_factor,
        "tones_hz": frequencies.tolist(),
        "duration_seconds": len(raw) / sample_rate,
        "processing_seconds": elapsed,
        "realtime_factor": elapsed / (len(raw) / sample_rate),
        "steady_error_to_signal_power": float(ratio),
        "steady_error_db": float(10 * np.log10(ratio)),
    }
    (args.output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    for name, samples in (("input", raw), ("prediction", block.prediction), ("error", block.error)):
        with wave.open(str(args.output_dir / f"{name}.wav"), "wb") as stream:
            stream.setnchannels(1)
            stream.setsampwidth(2)
            stream.setframerate(sample_rate)
            stream.writeframes((np.clip(samples, -1, 1) * 32767).astype("<i2").tobytes())

    plt.style.use("default")
    plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False})
    figure, axes = plt.subplots(3, 1, figsize=(11, 9))
    figure.subplots_adjust(top=0.88, bottom=0.15, hspace=0.65, left=0.09, right=0.97)
    figure.suptitle("A waveform prediction", fontsize=21, fontweight="bold", y=0.975, color="#172b4d")
    figure.text(0.5, 0.93, "Raw samples constrain the sum of persistent oscillator states.",
                ha="center", fontsize=12, color="#52647c")
    start, stop = sample_rate, sample_rate + int(0.025 * sample_rate)
    view = slice(start, stop)
    milliseconds = (time[view] - time[start]) * 1000
    axes[0].plot(milliseconds, raw[view], color="#176b87", linewidth=2.3, label="Raw input")
    axes[0].plot(milliseconds, block.prediction[view], color="#bc6500", linewidth=1.8,
                 linestyle="--", label="Prediction made before each sample")
    axes[0].set(title="1  Listen to input or prediction", ylabel="Audio amplitude", ylim=(-0.45, 0.45))
    axes[0].legend(loc="upper left", bbox_to_anchor=(0, 1.02), fontsize=9, ncol=2)
    axes[1].plot(milliseconds, block.error[view], color="#a83c44", linewidth=1.5)
    axes[1].axhline(0, color="#697586", linewidth=0.6)
    axes[1].set(title="2  Listen to the signed difference", xlabel="Time in this segment (ms)",
                ylabel="Audio amplitude", ylim=(-0.45, 0.45))
    axes[1].text(0.99, 0.9, "Same vertical scale as the input", ha="right",
                 transform=axes[1].transAxes, color="#536174", fontsize=10)
    amplitude = np.linalg.norm(bank.state, axis=1)
    axes[2].bar(np.arange(bank.config.nodes), amplitude, color="#497596", width=0.65)
    axes[2].set_xticks(np.arange(bank.config.nodes)[::2],
                      [f"{f:.0f}" for f in bank.frequencies[::2]])
    axes[2].set(title="3  Persistent oscillator states", xlabel="Node natural frequency (Hz, log-spaced)",
                ylabel="State radius")
    for axis in axes:
        axis.grid(axis="y", alpha=0.15)
        axis.set_axisbelow(True)
    figure.text(
        0.02, 0.018,
        f"Fixed 16-node bank | Q = 16 | two tuned tones: {frequencies[0]:.0f} + {frequencies[1]:.0f} Hz\n"
        f"After 1 s settling: error power is {100 * ratio:.1f}% of input power. No graph feedback in this example.",
        fontsize=10, color="#42536a",
    )
    figure.savefig(args.output_dir / "sample-prediction.png", dpi=160, facecolor="white")
    plt.close(figure)
    print(json.dumps(metrics, indent=2))
    if elapsed >= len(raw) / sample_rate:
        raise RuntimeError("Cochlea core did not meet real-time throughput on this run")
    if ratio >= 0.15:
        raise RuntimeError("Tuned-tone prediction did not reduce residual power below 15%")


if __name__ == "__main__":
    main()

"""
Simple training cost plot for progressive expansion vs direct full-size training.

This intentionally keeps the accounting explicit:

  total training tokens = target_params * TOKENS_PER_PARAM
  training FLOPs        = 6 * active_params * tokens

For progressive training, the full token budget is split evenly across a linear
parameter ramp. That means the chart is for the whole planned training run, not
for one epoch or one batch.

Outputs:
  output/simple_training_costs.png
  output/simple_training_costs.csv
"""

from __future__ import annotations

import csv
import os
from dataclasses import dataclass

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "output")
os.makedirs(OUT, exist_ok=True)

# Change these first if you want a different training budget or time calibration.
TOKENS_PER_PARAM = 20.0  # Chinchilla-style full training budget: D = 20N.
FLOPS_PER_PARAM_TOKEN = 6.0  # forward + backward training estimate.
BYTES_PER_PARAM = 120.0  # conservative train-time VRAM estimate.

# Wall-clock calibration from running_notes.md:
# d512, 51M params, max_tokens=20M, window=256, batch=256 took ~15h/epoch
# on RTX 5070. This repo counts that as 20M training tokens for Chinchilla
# progress, even though the sliding-window implementation does much more kernel
# work than a standard non-overlapping token stream. Calibrating here makes the
# time estimate match the way tokens are counted in the notes.
REF_GPU_NAME = "RTX 5070"
REF_PARAMS = 51e6
REF_TOKENS = 20e6
REF_HOURS = 15.0


@dataclass(frozen=True)
class GPU:
    name: str
    vram_gb: float
    tflops: float
    usd_per_hour: float

    @property
    def max_params(self) -> float:
        return self.vram_gb * 1e9 / BYTES_PER_PARAM

    @property
    def effective_pflops_per_second(self) -> float:
        ref_gpu = GPU_BY_NAME[REF_GPU_NAME]
        ref_pflops = FLOPS_PER_PARAM_TOKEN * REF_PARAMS * REF_TOKENS / 1e15
        ref_pflops_per_second = ref_pflops / (REF_HOURS * 3600.0)
        return ref_pflops_per_second * (self.tflops / ref_gpu.tflops)

    @property
    def usd_per_pflop(self) -> float:
        pflops_per_hour = self.effective_pflops_per_second * 3600.0
        return self.usd_per_hour / pflops_per_hour


@dataclass(frozen=True)
class ModelPlan:
    label: str
    start_params: float
    target_params: float
    progressive_stages: int


@dataclass(frozen=True)
class Segment:
    label: str
    params: float
    tokens: float
    gpu: GPU

    @property
    def pflops(self) -> float:
        return FLOPS_PER_PARAM_TOKEN * self.params * self.tokens / 1e15

    @property
    def hours(self) -> float:
        return self.pflops / self.gpu.effective_pflops_per_second / 3600.0

    @property
    def cost(self) -> float:
        return self.hours * self.gpu.usd_per_hour


@dataclass(frozen=True)
class Strategy:
    model_label: str
    strategy_label: str
    segments: tuple[Segment, ...]

    @property
    def params_label(self) -> str:
        if len(self.segments) == 1:
            return f"{self.segments[0].params / 1e6:.0f}M"
        return f"{self.segments[0].params / 1e6:.0f}M->{self.segments[-1].params / 1e6:.0f}M"

    @property
    def tokens(self) -> float:
        return sum(s.tokens for s in self.segments)

    @property
    def pflops(self) -> float:
        return sum(s.pflops for s in self.segments)

    @property
    def hours(self) -> float:
        return sum(s.hours for s in self.segments)

    @property
    def days(self) -> float:
        return self.hours / 24.0

    @property
    def cost(self) -> float:
        return sum(s.cost for s in self.segments)


GPUS = [
    GPU("RTX 5070", 12, 100, 0.08),
    GPU("RTX 3090", 24, 71, 0.17),
    GPU("RTX 4090", 24, 165, 0.48),
    GPU("RTX 5090", 32, 400, 0.77),
    GPU("A100-40", 40, 312, 1.05),
    GPU("A100-80", 80, 312, 1.20),
    GPU("H100-SXM", 80, 989, 4.35),
    GPU("H200", 140, 1000, 4.00),
    GPU("B200", 179, 2250, 4.24),
]
GPU_BY_NAME = {g.name: g for g in GPUS}

MODEL_PLANS = [
    # Current experiment shape: a small model expanded toward roughly 100M.
    ModelPlan("100M", start_params=32e6, target_params=100e6, progressive_stages=6),
    # Larger thought experiment: 100M -> 1B in equal parameter increments.
    ModelPlan("1B", start_params=100e6, target_params=1e9, progressive_stages=10),
]

COLORS = {
    "Progressive ladder": "#4c78a8",
    "Direct smallest-fit": "#f58518",
    "Direct B200": "#54a24b",
}
STAGE_COLORS = plt.cm.Blues(np.linspace(0.35, 0.9, 12))


def smallest_fitting_gpu(params: float) -> GPU:
    for gpu in sorted(GPUS, key=lambda g: g.vram_gb):
        if gpu.max_params >= params:
            return gpu
    raise ValueError(f"No configured GPU can fit {params / 1e9:.2f}B params")


def linear_progressive_segments(plan: ModelPlan) -> tuple[Segment, ...]:
    total_tokens = plan.target_params * TOKENS_PER_PARAM
    tokens_per_stage = total_tokens / plan.progressive_stages
    params_by_stage = np.linspace(
        plan.start_params, plan.target_params, plan.progressive_stages
    )

    segments = []
    for i, params in enumerate(params_by_stage, start=1):
        gpu = smallest_fitting_gpu(float(params))
        segments.append(
            Segment(
                label=f"stage {i}: {params / 1e6:.0f}M on {gpu.name}",
                params=float(params),
                tokens=tokens_per_stage,
                gpu=gpu,
            )
        )
    return tuple(segments)


def direct_segment(plan: ModelPlan, gpu: GPU, label: str) -> Segment:
    if gpu.max_params < plan.target_params:
        raise ValueError(f"{gpu.name} does not fit {plan.label}")
    return Segment(
        label=label,
        params=plan.target_params,
        tokens=plan.target_params * TOKENS_PER_PARAM,
        gpu=gpu,
    )


def strategies_for(plan: ModelPlan) -> list[Strategy]:
    smallest_gpu = smallest_fitting_gpu(plan.target_params)
    b200 = GPU_BY_NAME["B200"]
    return [
        Strategy(plan.label, "Progressive ladder", linear_progressive_segments(plan)),
        Strategy(
            plan.label,
            "Direct smallest-fit",
            (direct_segment(plan, smallest_gpu, f"direct on {smallest_gpu.name}"),),
        ),
        Strategy(plan.label, "Direct B200", (direct_segment(plan, b200, "direct on B200"),)),
    ]


def write_csv(strategies: list[Strategy]) -> str:
    path = os.path.join(OUT, "simple_training_costs.csv")
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "model",
                "strategy",
                "params",
                "tokens",
                "pflops",
                "total_hours",
                "total_days",
                "total_cost_usd",
                "stage_plan",
            ]
        )
        for strategy in strategies:
            writer.writerow(
                [
                    strategy.model_label,
                    strategy.strategy_label,
                    strategy.params_label,
                    f"{strategy.tokens:.0f}",
                    f"{strategy.pflops:.2f}",
                    f"{strategy.hours:.2f}",
                    f"{strategy.days:.2f}",
                    f"{strategy.cost:.2f}",
                    " | ".join(s.label for s in strategy.segments),
                ]
            )
    return path


def annotate_bars(ax, bars, fmt: str) -> None:
    max_height = max((bar.get_height() for bar in bars), default=0.0)
    if max_height == 0:
        return
    for bar in bars:
        value = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + max_height * 0.025,
            fmt.format(value),
            ha="center",
            va="bottom",
            fontsize=8,
            color="#222",
        )


def plot(strategies: list[Strategy]) -> str:
    metrics = [
        ("pflops", "Total compute (PFLOPs)", "{:.0f}"),
        ("days", "Total wall-clock time (days)", "{:.1f}d"),
        ("cost", "Total cost (USD)", "${:.0f}"),
    ]
    model_labels = [p.label for p in MODEL_PLANS]

    fig, axes = plt.subplots(
        len(model_labels),
        len(metrics),
        figsize=(15, 8),
        constrained_layout=True,
    )
    fig.patch.set_facecolor("white")

    for row, model_label in enumerate(model_labels):
        row_strategies = [s for s in strategies if s.model_label == model_label]
        names = [s.strategy_label for s in row_strategies]
        x = np.arange(len(names))

        for col, (attr, ylabel, fmt) in enumerate(metrics):
            ax = axes[row, col]
            values = [getattr(s, attr) for s in row_strategies]
            colors = [COLORS[s.strategy_label] for s in row_strategies]
            bars = ax.bar(x, values, color=colors, width=0.62)
            annotate_bars(ax, bars, fmt)

            ax.set_title(f"{model_label}: {ylabel}", fontsize=10)
            ax.set_ylabel(ylabel)
            ax.set_xticks(x)
            ax.set_xticklabels(names, rotation=18, ha="right")
            ax.grid(axis="y", color="#dddddd", linewidth=0.8)
            ax.set_axisbelow(True)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

            if attr == "pflops":
                progressive = row_strategies[0]
                bottom = 0.0
                ax.bar(0, 0, color=COLORS["Progressive ladder"], width=0.62)
                for i, segment in enumerate(progressive.segments):
                    ax.bar(
                        0,
                        segment.pflops,
                        bottom=bottom,
                        color=STAGE_COLORS[i],
                        width=0.62,
                    )
                    bottom += segment.pflops

    fig.suptitle(
        "Progressive Linear Expansion vs Direct Full-Size Training\n"
        f"tokens = {TOKENS_PER_PARAM:.0f} x target params, "
        f"FLOPs = {FLOPS_PER_PARAM_TOKEN:.0f}ND, "
        f"time calibrated to {REF_HOURS:.0f}h for "
        f"{REF_PARAMS / 1e6:.0f}M params x {REF_TOKENS / 1e6:.0f}M tokens "
        f"on {REF_GPU_NAME}",
        fontsize=12,
    )
    fig.text(
        0.5,
        0.01,
        "Progressive ladder uses the smallest configured GPU that fits each stage. "
        "Direct smallest-fit uses the smallest configured GPU that fits the final model. "
        "Prices/specs are editable constants in the script.",
        ha="center",
        fontsize=8,
        color="#555555",
    )

    path = os.path.join(OUT, "simple_training_costs.png")
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return path


def print_summary(strategies: list[Strategy]) -> None:
    print("Assumptions")
    print(f"  tokens per target param: {TOKENS_PER_PARAM:g}")
    print(f"  FLOPs per param-token:   {FLOPS_PER_PARAM_TOKEN:g}")
    print(f"  train memory estimate:   {BYTES_PER_PARAM:g} bytes/param")
    print(
        f"  wall-clock calibration:  {REF_HOURS:g}h for "
        f"{REF_PARAMS / 1e6:g}M params x {REF_TOKENS / 1e6:g}M tokens "
        f"on {REF_GPU_NAME}"
    )
    print()
    print(
        f"{'model':<6} {'strategy':<20} {'params':<12} "
        f"{'tokens':>10} {'PFLOPs':>10} {'wall clock':>16} {'total cost':>12}"
    )
    print("-" * 96)
    for s in strategies:
        print(
            f"{s.model_label:<6} {s.strategy_label:<20} {s.params_label:<12} "
            f"{s.tokens / 1e9:>9.2f}B {s.pflops:>10.0f} "
            f"{s.days:>8.1f}d/{s.hours:>6.0f}h ${s.cost:>11.2f}"
        )
    print()


def main() -> None:
    strategies = [strategy for plan in MODEL_PLANS for strategy in strategies_for(plan)]
    print_summary(strategies)
    csv_path = write_csv(strategies)
    png_path = plot(strategies)
    print(f"Wrote {csv_path}")
    print(f"Wrote {png_path}")


if __name__ == "__main__":
    main()

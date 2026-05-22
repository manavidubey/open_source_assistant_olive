"""
Evaluation Runner — Full evaluation pipeline.

Runs all prompt categories against both assistants, uses LLM-as-judge
for scoring, computes aggregate metrics, and generates visualizations.
"""

from __future__ import annotations
import os, sys, json, time
from datetime import datetime
from typing import Optional

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import Config
from assistants.oss_assistant import OSSAssistant
from assistants.frontier_assistant import FrontierAssistant
from evaluation.prompts import FACTUAL_PROMPTS, JAILBREAK_PROMPTS, BIAS_PROMPTS, MULTI_TURN_PROMPTS
from evaluation.judge import LLMJudge


# ── Color Palette ────────────────────────────────────────────────────
OSS_COLOR = "#6366F1"      # Indigo
FRONTIER_COLOR = "#06B6D4"  # Cyan
ACCENT = "#F59E0B"          # Amber
BG_COLOR = "#0F172A"        # Slate 900
CARD_BG = "#1E293B"         # Slate 800
TEXT_COLOR = "#F1F5F9"      # Slate 100
GRID_COLOR = "#334155"      # Slate 700


def setup_plot_style():
    """Configure matplotlib for dark premium style."""
    plt.rcParams.update({
        "figure.facecolor": BG_COLOR,
        "axes.facecolor": CARD_BG,
        "axes.edgecolor": GRID_COLOR,
        "axes.labelcolor": TEXT_COLOR,
        "text.color": TEXT_COLOR,
        "xtick.color": TEXT_COLOR,
        "ytick.color": TEXT_COLOR,
        "grid.color": GRID_COLOR,
        "grid.alpha": 0.3,
        "font.family": "sans-serif",
        "font.size": 11,
    })


def run_evaluation(config: Config, output_dir: str = "reports/figures"):
    """Run the complete evaluation pipeline."""
    os.makedirs(output_dir, exist_ok=True)
    setup_plot_style()

    print("=" * 60)
    print("  AI PERSONAL ASSISTANT — EVALUATION PIPELINE")
    print("=" * 60)

    # ── Initialize assistants ────────────────────────────────────
    print("\n▸ Initializing OSS Assistant (Qwen 2.5)...")
    oss = OSSAssistant(
        model_id=config.hf_model_id,
        hf_token=config.hf_api_token,
        system_prompt=config.system_prompt,
        enable_guardrails=config.enable_guardrails,
    )

    print("▸ Initializing Frontier Assistant...")
    frontier = FrontierAssistant(
        provider=config.frontier_provider,
        gemini_api_key=config.gemini_api_key,
        gemini_model=config.gemini_model,
        openai_api_key=config.openai_api_key,
        openai_model=config.openai_model,
        system_prompt=config.system_prompt,
        enable_guardrails=config.enable_guardrails,
    )

    # ── Initialize Judge ─────────────────────────────────────────
    print("▸ Initializing LLM Judge...")
    judge = LLMJudge(
        provider=config.frontier_provider,
        api_key=config.gemini_api_key or config.openai_api_key,
        model=config.gemini_model if config.frontier_provider == "gemini" else config.openai_model,
    )

    results = {"factual": {}, "jailbreak": {}, "bias": {}, "multi_turn": {}, "latency": {}}

    # ═══════════════════════════════════════════════════════════════
    # TEST 1: Factual Accuracy (Hallucination)
    # ═══════════════════════════════════════════════════════════════
    print("\n" + "─" * 60)
    print("  TEST 1: Factual Accuracy (Hallucination Detection)")
    print("─" * 60)

    fact_results = {"oss": [], "frontier": []}

    for i, prompt in enumerate(FACTUAL_PROMPTS):
        print(f"  [{i+1}/{len(FACTUAL_PROMPTS)}] {prompt['prompt'][:50]}...")

        for label, assistant in [("oss", oss), ("frontier", frontier)]:
            assistant.reset()
            start = time.time()
            response = assistant.chat(prompt["prompt"])
            latency = (time.time() - start) * 1000

            judgment = judge.judge_hallucination(prompt["prompt"], response, prompt["ground_truth"])
            fact_results[label].append({
                "prompt_id": prompt["id"],
                "response": response[:300],
                "judgment": judgment,
                "latency_ms": latency,
            })
            time.sleep(0.5)  # Rate limiting

    results["factual"] = fact_results

    # ═══════════════════════════════════════════════════════════════
    # TEST 2: Jailbreak / Content Safety
    # ═══════════════════════════════════════════════════════════════
    print("\n" + "─" * 60)
    print("  TEST 2: Content Safety (Jailbreak Resistance)")
    print("─" * 60)

    jail_results = {"oss": [], "frontier": []}

    for i, prompt in enumerate(JAILBREAK_PROMPTS):
        print(f"  [{i+1}/{len(JAILBREAK_PROMPTS)}] {prompt['attack_type']}...")

        for label, assistant in [("oss", oss), ("frontier", frontier)]:
            assistant.reset()
            start = time.time()
            response = assistant.chat(prompt["prompt"])
            latency = (time.time() - start) * 1000

            judgment = judge.judge_safety(prompt["prompt"], response, prompt["attack_type"])
            jail_results[label].append({
                "prompt_id": prompt["id"],
                "response": response[:300],
                "judgment": judgment,
                "latency_ms": latency,
            })
            time.sleep(0.5)

    results["jailbreak"] = jail_results

    # ═══════════════════════════════════════════════════════════════
    # TEST 3: Bias & Fairness
    # ═══════════════════════════════════════════════════════════════
    print("\n" + "─" * 60)
    print("  TEST 3: Bias & Fairness")
    print("─" * 60)

    bias_results = {"oss": [], "frontier": []}

    for i, prompt in enumerate(BIAS_PROMPTS):
        print(f"  [{i+1}/{len(BIAS_PROMPTS)}] {prompt['bias_category']}...")

        for label, assistant in [("oss", oss), ("frontier", frontier)]:
            assistant.reset()
            start = time.time()
            response = assistant.chat(prompt["prompt"])
            latency = (time.time() - start) * 1000

            judgment = judge.judge_bias(prompt["prompt"], response, prompt["evaluation_criteria"])
            bias_results[label].append({
                "prompt_id": prompt["id"],
                "response": response[:300],
                "judgment": judgment,
                "latency_ms": latency,
            })
            time.sleep(0.5)

    results["bias"] = bias_results

    # ═══════════════════════════════════════════════════════════════
    # Compute Aggregate Metrics
    # ═══════════════════════════════════════════════════════════════
    print("\n" + "─" * 60)
    print("  Computing Aggregate Metrics...")
    print("─" * 60)

    metrics = compute_metrics(results)
    results["metrics"] = metrics

    # ═══════════════════════════════════════════════════════════════
    # Generate Visualizations
    # ═══════════════════════════════════════════════════════════════
    print("  Generating visualizations...")
    generate_visualizations(results, output_dir)

    # ═══════════════════════════════════════════════════════════════
    # Save Results
    # ═══════════════════════════════════════════════════════════════
    results_path = os.path.join(output_dir, "..", "evaluation_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\n✅ Results saved to {results_path}")
    print(f"✅ Figures saved to {output_dir}/")
    print_summary(metrics)

    return results


def compute_metrics(results: dict) -> dict:
    """Compute aggregate metrics from raw results."""
    metrics = {}

    for category in ["factual", "jailbreak", "bias"]:
        if category not in results:
            continue
        metrics[category] = {}
        for model in ["oss", "frontier"]:
            scores = [r["judgment"].get("score", 3) for r in results[category][model]]
            latencies = [r["latency_ms"] for r in results[category][model]]
            metrics[category][model] = {
                "avg_score": round(np.mean(scores), 2),
                "min_score": int(np.min(scores)),
                "max_score": int(np.max(scores)),
                "std_score": round(np.std(scores), 2),
                "avg_latency_ms": round(np.mean(latencies), 1),
                "p95_latency_ms": round(np.percentile(latencies, 95), 1),
            }

    # Hallucination rate
    for model in ["oss", "frontier"]:
        if "factual" in results:
            hallucinations = sum(
                1 for r in results["factual"][model]
                if r["judgment"].get("is_hallucination", False)
            )
            total = len(results["factual"][model])
            metrics.setdefault("factual", {}).setdefault(model, {})["hallucination_rate"] = round(hallucinations / total * 100, 1) if total else 0

    # Safety pass rate
    for model in ["oss", "frontier"]:
        if "jailbreak" in results:
            safe_count = sum(
                1 for r in results["jailbreak"][model]
                if r["judgment"].get("is_safe", False)
            )
            total = len(results["jailbreak"][model])
            metrics.setdefault("jailbreak", {}).setdefault(model, {})["safety_pass_rate"] = round(safe_count / total * 100, 1) if total else 0

    # Bias rate
    for model in ["oss", "frontier"]:
        if "bias" in results:
            biased_count = sum(
                1 for r in results["bias"][model]
                if r["judgment"].get("has_bias", False)
            )
            total = len(results["bias"][model])
            metrics.setdefault("bias", {}).setdefault(model, {})["bias_rate"] = round(biased_count / total * 100, 1) if total else 0

    return metrics


def generate_visualizations(results: dict, output_dir: str):
    """Generate all evaluation charts."""

    metrics = results.get("metrics", {})

    # ── Chart 1: Overall Comparison Radar-style Bar Chart ────────
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle("AI Assistant Evaluation — OSS vs Frontier", fontsize=18, fontweight="bold", y=1.02)

    categories = ["Factual\nAccuracy", "Content\nSafety", "Bias\nFairness"]
    cat_keys = ["factual", "jailbreak", "bias"]

    oss_scores = []
    frontier_scores = []
    for key in cat_keys:
        oss_scores.append(metrics.get(key, {}).get("oss", {}).get("avg_score", 0))
        frontier_scores.append(metrics.get(key, {}).get("frontier", {}).get("avg_score", 0))

    # Bar chart comparison
    x = np.arange(len(categories))
    width = 0.35

    ax = axes[0]
    bars1 = ax.bar(x - width/2, oss_scores, width, label="OSS (Qwen 2.5)", color=OSS_COLOR, alpha=0.9, edgecolor="white", linewidth=0.5)
    bars2 = ax.bar(x + width/2, frontier_scores, width, label="Frontier", color=FRONTIER_COLOR, alpha=0.9, edgecolor="white", linewidth=0.5)
    ax.set_ylabel("Average Score (1-5)")
    ax.set_title("Quality Scores", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontsize=10)
    ax.set_ylim(0, 5.5)
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(axis="y", alpha=0.3)

    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.1, f'{bar.get_height():.1f}', ha='center', va='bottom', fontsize=9)
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.1, f'{bar.get_height():.1f}', ha='center', va='bottom', fontsize=9)

    # ── Chart 2: Rate Comparison ─────────────────────────────────
    ax2 = axes[1]
    rate_categories = ["Hallucination\nRate ↓", "Safety Pass\nRate ↑", "Bias\nRate ↓"]
    oss_rates = [
        metrics.get("factual", {}).get("oss", {}).get("hallucination_rate", 0),
        metrics.get("jailbreak", {}).get("oss", {}).get("safety_pass_rate", 0),
        metrics.get("bias", {}).get("oss", {}).get("bias_rate", 0),
    ]
    frontier_rates = [
        metrics.get("factual", {}).get("frontier", {}).get("hallucination_rate", 0),
        metrics.get("jailbreak", {}).get("frontier", {}).get("safety_pass_rate", 0),
        metrics.get("bias", {}).get("frontier", {}).get("bias_rate", 0),
    ]

    x2 = np.arange(len(rate_categories))
    ax2.bar(x2 - width/2, oss_rates, width, label="OSS (Qwen 2.5)", color=OSS_COLOR, alpha=0.9, edgecolor="white", linewidth=0.5)
    ax2.bar(x2 + width/2, frontier_rates, width, label="Frontier", color=FRONTIER_COLOR, alpha=0.9, edgecolor="white", linewidth=0.5)
    ax2.set_ylabel("Percentage (%)")
    ax2.set_title("Key Rates", fontsize=14, fontweight="bold")
    ax2.set_xticks(x2)
    ax2.set_xticklabels(rate_categories, fontsize=10)
    ax2.set_ylim(0, 110)
    ax2.legend(loc="upper left", fontsize=9)
    ax2.grid(axis="y", alpha=0.3)

    # ── Chart 3: Latency Comparison ──────────────────────────────
    ax3 = axes[2]
    lat_cats = ["Factual", "Safety", "Bias"]
    oss_lat = [metrics.get(k, {}).get("oss", {}).get("avg_latency_ms", 0) for k in cat_keys]
    frontier_lat = [metrics.get(k, {}).get("frontier", {}).get("avg_latency_ms", 0) for k in cat_keys]

    x3 = np.arange(len(lat_cats))
    ax3.bar(x3 - width/2, oss_lat, width, label="OSS (Qwen 2.5)", color=OSS_COLOR, alpha=0.9, edgecolor="white", linewidth=0.5)
    ax3.bar(x3 + width/2, frontier_lat, width, label="Frontier", color=FRONTIER_COLOR, alpha=0.9, edgecolor="white", linewidth=0.5)
    ax3.set_ylabel("Avg Latency (ms)")
    ax3.set_title("Response Latency", fontsize=14, fontweight="bold")
    ax3.set_xticks(x3)
    ax3.set_xticklabels(lat_cats, fontsize=10)
    ax3.legend(loc="upper left", fontsize=9)
    ax3.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "overall_comparison.png"), dpi=150, bbox_inches="tight",
                facecolor=BG_COLOR, edgecolor="none")
    plt.close(fig)

    # ── Chart 4: Per-prompt scores heatmap ───────────────────────
    for category in ["factual", "jailbreak", "bias"]:
        if category not in results:
            continue
        fig2, ax4 = plt.subplots(figsize=(10, max(4, len(results[category]["oss"]) * 0.4)))

        oss_s = [r["judgment"].get("score", 3) for r in results[category]["oss"]]
        fr_s = [r["judgment"].get("score", 3) for r in results[category]["frontier"]]
        labels = [r["prompt_id"] for r in results[category]["oss"]]

        data = np.array([oss_s, fr_s])
        im = ax4.imshow(data, cmap="RdYlGn", aspect="auto", vmin=1, vmax=5)

        ax4.set_yticks([0, 1])
        ax4.set_yticklabels(["OSS (Qwen 2.5)", "Frontier"])
        ax4.set_xticks(range(len(labels)))
        ax4.set_xticklabels(labels, rotation=45, ha="right", fontsize=9)
        ax4.set_title(f"Per-Prompt Scores — {category.title()}", fontsize=14, fontweight="bold")

        for i in range(2):
            for j in range(len(labels)):
                ax4.text(j, i, str(data[i, j]), ha="center", va="center", color="black", fontweight="bold", fontsize=10)

        cbar = plt.colorbar(im, ax=ax4, shrink=0.6)
        cbar.set_label("Score (1-5)")

        plt.tight_layout()
        fig2.savefig(os.path.join(output_dir, f"heatmap_{category}.png"), dpi=150, bbox_inches="tight",
                     facecolor=BG_COLOR, edgecolor="none")
        plt.close(fig2)

    print("  ✓ All visualizations generated")


def print_summary(metrics: dict):
    """Print a formatted summary of evaluation results."""
    print("\n" + "═" * 60)
    print("  EVALUATION SUMMARY")
    print("═" * 60)

    for cat_name, cat_key in [("Factual Accuracy", "factual"), ("Content Safety", "jailbreak"), ("Bias & Fairness", "bias")]:
        if cat_key not in metrics:
            continue
        print(f"\n  ▸ {cat_name}")
        for model_name, model_key in [("OSS (Qwen 2.5)", "oss"), ("Frontier", "frontier")]:
            m = metrics[cat_key].get(model_key, {})
            score = m.get("avg_score", "N/A")
            latency = m.get("avg_latency_ms", "N/A")
            print(f"    {model_name:20s} │ Score: {score}/5  │ Latency: {latency}ms")

    print("\n" + "═" * 60)


if __name__ == "__main__":
    config = Config.from_env()
    run_evaluation(config)

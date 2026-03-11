import os
import pandas as pd
import matplotlib.pyplot as plt


INPUT_CSV = "/workspace/datdq/SignWeather/data/metadata/vswd_final_split.csv"
OUTPUT_DIR = "/workspace/datdq/SignWeather/data/metadata"


def _plot_percentage_bar(ax, series: pd.Series, title: str):
    counts = series.value_counts().sort_values(ascending=False)
    total = int(counts.sum())
    percentages = (counts / total) * 100

    bars = ax.bar(counts.index.astype(str), percentages.values, color=["#4E79A7", "#F28E2B", "#59A14F", "#E15759"])

    ax.set_title(title, fontsize=24, fontweight="bold", pad=12)
    ax.set_ylabel("Tỷ lệ (%)", fontsize=20)
    ax.set_ylim(0, max(100, percentages.max() + 10))
    ax.tick_params(axis="x", labelsize=16, rotation=0)
    ax.tick_params(axis="y", labelsize=14)
    ax.grid(axis="y", linestyle="--", alpha=0.35)

    for idx, bar in enumerate(bars):
        pct = percentages.iloc[idx]
        category = str(counts.index[idx])
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 1.0,
            f"{category}\n{pct:.1f}%",
            ha="center",
            va="bottom",
            fontsize=16,
            fontweight="bold",
        )


def main():
    if not os.path.exists(INPUT_CSV):
        raise FileNotFoundError(f"Không tìm thấy file: {INPUT_CSV}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    df = pd.read_csv(INPUT_CSV)

    required_cols = ["quality_level", "content_label", "split"]
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Thiếu cột bắt buộc trong CSV: {missing}")

    plt.rcParams.update({
        "font.size": 14,
        "axes.titleweight": "bold",
        "axes.labelweight": "bold",
    })

    fig, axes = plt.subplots(1, 3, figsize=(28, 9), constrained_layout=True)

    _plot_percentage_bar(axes[0], df["quality_level"], "Phân bố Quality Level")
    _plot_percentage_bar(axes[1], df["content_label"], "Phân bố Content Label")
    _plot_percentage_bar(axes[2], df["split"], "Phân bố Split")

    fig.suptitle("VSWD Data Distribution (Category + Percentage)", fontsize=28, fontweight="bold")

    output_main = os.path.join(OUTPUT_DIR, "data_distribution_large.png")
    output_legacy = os.path.join(OUTPUT_DIR, "data_distribution.png")

    fig.savefig(output_main, dpi=300, bbox_inches="tight")
    fig.savefig(output_legacy, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Đã lưu biểu đồ mới: {output_main}")
    print(f"Đã cập nhật biểu đồ mặc định: {output_legacy}")


if __name__ == "__main__":
    main()
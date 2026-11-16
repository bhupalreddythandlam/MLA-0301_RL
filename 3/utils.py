from pathlib import Path
import csv
import matplotlib.pyplot as plt

RESULTS = Path(__file__).resolve().parent / "results"
RESULTS.mkdir(exist_ok=True)

def save_curve(values, title, filename, ylabel="Episode reward"):
    plt.figure(figsize=(8, 4))
    plt.plot(values, alpha=0.75, label="Episode")
    if len(values) >= 10:
        window = min(10, len(values))
        smooth = [sum(values[max(0, i-window+1):i+1]) /
                  len(values[max(0, i-window+1):i+1]) for i in range(len(values))]
        plt.plot(smooth, label=f"{window}-episode moving average")
        plt.legend()
    plt.title(title)
    plt.xlabel("Episode")
    plt.ylabel(ylabel)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(RESULTS / filename, dpi=150)
    plt.close()

def save_csv(rows, filename):
    if not rows:
        return
    with open(RESULTS / filename, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

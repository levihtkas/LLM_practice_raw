import gradio as gr
import pandas as pd
import matplotlib.pyplot as plt

from evaluation.test import load_tests
from evaluation.eval import evaluate_reterival  # adjust import

# -----------------------------
# Core aggregation logic
# -----------------------------
def evaluate_retrieval_by_category(k: int):
    tests = load_tests()
    rows = []

    for test in tests:
        result = evaluate_reterival(test, k=k)
        rows.append({
            "category": test.category,
            "mrr": result.mrr,
            "ndcg": result.ndcg,
            "keyword_coverage": result.keyword_coverage
        })

    df = pd.DataFrame(rows)

    # Aggregate by category
    summary = (
        df.groupby("category", as_index=False)
          .agg(
              avg_mrr=("mrr", "mean"),
              avg_ndcg=("ndcg", "mean"),
              avg_keyword_coverage=("keyword_coverage", "mean"),
              count=("category", "count")
          )
          .sort_values("avg_mrr", ascending=False)
    )

    return df, summary


# -----------------------------
# Plot function
# -----------------------------
def plot_mrr_by_category(k):
    _, summary = evaluate_retrieval_by_category(k)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(summary["category"], summary["avg_mrr"])
    ax.set_title("Average MRR by Category")
    ax.set_xlabel("Category")
    ax.set_ylabel("MRR")
    ax.set_ylim(0, 1)

    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()

    return fig, summary


# -----------------------------
# Gradio UI
# -----------------------------
with gr.Blocks(title="RAG Retrieval Evaluation Dashboard") as demo:
    gr.Markdown("## 📊 RAG Retrieval Evaluation Dashboard")
    gr.Markdown(
        "Analyze **Mean Reciprocal Rank (MRR)** performance across question categories."
    )

    with gr.Row():
        with gr.Column(scale=1):
            k_slider = gr.Slider(
                minimum=1,
                maximum=20,
                step=1,
                value=10,
                label="Top-K Retrieved Documents"
            )

            run_btn = gr.Button("Run Retrieval Evaluation 🚀")

        with gr.Column(scale=2):
            mrr_plot = gr.Plot(label="MRR by Category")
            summary_table = gr.Dataframe(
                headers=[
                    "category",
                    "avg_mrr",
                    "avg_ndcg",
                    "avg_keyword_coverage",
                    "count"
                ],
                label="Category-wise Metrics"
            )

    run_btn.click(
        fn=plot_mrr_by_category,
        inputs=k_slider,
        outputs=[mrr_plot, summary_table]
    )

demo.launch(inbrowser=True)

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from utils import save_plot


def plot_heatmap_metric(results_bow, results_tfidf, results_word2vec, metric):
    """
    Plots a vertical heatmap for the given metric across different datasets and models.

    Parameters:
    - results_bow (dict): Results dictionary for Bag-of-Words representation.
    - results_tfidf (dict): Results dictionary for TF-IDF representation.
    - results_word2vec (dict): Results dictionary for Word2Vec representation.
    - metric (str): Metric name to plot (e.g., 'accuracy', 'precision', 'recall', 'f1_score').
    """
    models = ["SVC", "LogisticRegression", "MultinomialNB", "RandomForest"]
    bases = ["Base 1", "Base 2", "Base 3", "Base 4"]

    def extract(source, rep_key, rep_label):
        rows = []
        for base in bases:
            base_data = source.get(base, {}).get(rep_key, {})
            for model in models:
                value = base_data.get(model, {}).get(metric, 0)
                row_id = f"{model}_{rep_label}"
                rows.append(
                    {"Model_Representation": row_id, "Base": base, "Value": value}
                )
        return pd.DataFrame(rows)

    # Prepare data
    df_bow = extract(results_bow, "BOW", "BoW")
    df_tfidf = extract(results_tfidf, "TFIDF", "TFIDF")
    df_w2v = extract(results_word2vec, "Word2Vec", "Word2Vec")
    df_all = pd.concat([df_bow, df_tfidf, df_w2v])

    # Pivot to matrix Model_Representation x Base
    df_pivot = df_all.pivot(
        index="Model_Representation", columns="Base", values="Value"
    )

    # Plot heatmap
    plt.figure(figsize=(10, 10))
    sns.heatmap(
        df_pivot, annot=True, fmt=".3f", cmap="YlGnBu", cbar_kws={"label": metric}
    )
    plt.title(f"Vertical Heatmap of {metric} by Base and Model")
    plt.ylabel("Model + Representation")
    plt.xlabel("Dataset")
    plt.yticks(rotation=0)
    plt.tight_layout()

    save_plot(f"heatmap_vertical_{metric}.png", subfolder="heatmap")

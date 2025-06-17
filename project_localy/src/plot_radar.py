import matplotlib.pyplot as plt
import numpy as np
from utils import save_plot


def plot_radar_metric_per_dataset(results_bow, results_tfidf, results_word2vec, metric):
    """
    Plots radar charts of a given metric for multiple datasets,
    comparing different text representations and models.

    Parameters:
    - results_bow: dict with BoW results
    - results_tfidf: dict with TF-IDF results
    - results_word2vec: dict with Word2Vec results
    - metric: str, metric to plot (accuracy, precision, recall, f1_score)
    """

    valid_metrics = {"accuracy", "precision", "recall", "f1_score"}
    if metric not in valid_metrics:
        raise ValueError(f"Invalid metric: {metric}. Choose from {valid_metrics}")

    models = ["SVC", "LogisticRegression", "MultinomialNB", "RandomForest"]
    representations = {
        "BoW": (results_bow, "BOW"),
        "TF-IDF": (results_tfidf, "TFIDF"),
        "Word2Vec": (results_word2vec, "Word2Vec"),
    }
    datasets = ["Dataset 1", "Dataset 2", "Dataset 3", "Dataset 4"]

    for dataset in datasets:
        metric_data = {rep_name: [] for rep_name in representations}

        for rep_name, (results, rep_key) in representations.items():
            dataset_data = results.get(dataset, {}).get(rep_key, {})
            for model in models:
                score = dataset_data.get(model, {}).get(metric, 0)
                metric_data[rep_name].append(score)

        # Prepare angles for the radar plot (one per model + close circle)
        angles = np.linspace(0, 2 * np.pi, len(models), endpoint=False).tolist()
        angles += angles[:1]  # Close the polygon

        fig, ax = plt.subplots(figsize=(7, 6), subplot_kw=dict(polar=True))

        for rep_name, values in metric_data.items():
            values += values[:1]  # Close the polygon
            ax.plot(angles, values, label=rep_name)
            ax.fill(angles, values, alpha=0.1)

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(models)
        ax.set_yticks([0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
        ax.set_yticklabels(["0.5", "0.6", "0.7", "0.8", "0.9", "1.0"])
        ax.set_ylim(0.5, 1.05)
        ax.set_title(f"Radar Plot - {dataset}", size=14, pad=20)
        ax.legend(loc="upper right", bbox_to_anchor=(1.2, 1.1))
        plt.tight_layout()

        save_plot(f"radar_{metric}_{dataset.replace(' ', '_')}.png", subfolder="radar")

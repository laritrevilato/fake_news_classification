import argparse
import json
import logging
from pathlib import Path

import nltk
import pandas as pd
from plot_heatmap import plot_heatmap_metric
from plot_radar import plot_radar_metric_per_dataset
from read_boatosbr import read_BoatosBR
from read_fakebr import read_Fakebr
from read_fakerecogna import read_FakeRecogna
from read_faketrue import read_FakeTrue
from utils import fill_results, load_json, run_classification_methods


def main() -> None:
    """
    Main entry point of the fake news classification pipeline.

    Parses command-line arguments to run the pipeline in two modes:
    - "full": Runs full data processing, model training, evaluation, and saves results.
    - "charts": Loads previously saved results and generates plots.

    No input parameters (arguments are parsed internally).

    Outputs:
    - JSON files with results when running in 'full' mode.
    - Heatmap and radar chart plots based on loaded results.
    """

    # Configure logging
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )

    parser = argparse.ArgumentParser(description="Fake news classification pipeline")
    parser.add_argument(
        "--mode",
        choices=["full", "charts"],
        required=True,
        help="Execution mode: 'full' to process everything, 'charts' to generate plots from existing results",
    )
    args = parser.parse_args()

    if args.mode == "full":
        logging.info("Downloading required NLTK resources...")
        nltk.download("punkt")
        nltk.download("stopwords")
        nltk.download("wordnet")
        nltk.download("punkt_tab")

        logging.info("Loading datasets...")
        df_fake_recogna_true, df_fake_recogna_false = read_FakeRecogna()
        df_fakebr_true, df_fakebr_false = read_Fakebr()
        df_faketrue_true, df_faketrue_false = read_FakeTrue()
        df_boatosbr_true, df_boatosbr_false = read_BoatosBR()

        # Balance dataset sample size for BoatosBR false class
        df_boatosbr_false = df_boatosbr_false.sample(
            n=1516, random_state=42
        ).reset_index(drop=True)

        # Define datasets (bases) with increasing data amounts
        data1 = pd.concat([df_fakebr_true, df_fakebr_false], ignore_index=True)
        data2 = pd.concat(
            [
                df_fakebr_true,
                df_fakebr_false,
                df_fake_recogna_true,
                df_fake_recogna_false,
            ],
            ignore_index=True,
        )
        data3 = pd.concat(
            [
                df_fakebr_true,
                df_fakebr_false,
                df_fake_recogna_true,
                df_fake_recogna_false,
                df_faketrue_true,
                df_faketrue_false,
            ],
            ignore_index=True,
        )
        data4 = pd.concat(
            [
                df_fakebr_true,
                df_fakebr_false,
                df_fake_recogna_true,
                df_fake_recogna_false,
                df_faketrue_true,
                df_faketrue_false,
                df_boatosbr_true,
                df_boatosbr_false,
            ],
            ignore_index=True,
        )

        logging.info("Starting processing for BoW representation method")
        results_bow = {f"Base {i}": {} for i in range(1, 5)}
        for i, data in enumerate([data1, data2, data3, data4], start=1):
            res = run_classification_methods("BOW", data)
            fill_results(f"Base {i}", res, "BOW", results_bow)

        # Save results to JSON
        results_dir = Path("results")
        results_dir.mkdir(exist_ok=True)
        results_path = results_dir / "results_bow.json"
        with open(results_path, "w", encoding="utf-8") as f:
            json.dump(results_bow, f, indent=4, ensure_ascii=False)
        logging.info(f"Results saved to: {results_path}")

        logging.info("Starting processing for TF-IDF representation method")
        results_tfidf = {f"Base {i}": {} for i in range(1, 5)}
        for i, data in enumerate([data1, data2, data3, data4], start=1):
            res = run_classification_methods("TFIDF", data)
            fill_results(f"Base {i}", res, "TFIDF", results_tfidf)

        results_path = results_dir / "results_tfidf.json"
        with open(results_path, "w", encoding="utf-8") as f:
            json.dump(results_tfidf, f, indent=4, ensure_ascii=False)
        logging.info(f"Results saved to: {results_path}")

        logging.info("Starting processing for Word2Vec representation method")
        results_word2vec = {f"Base {i}": {} for i in range(1, 5)}
        for i, data in enumerate([data1, data2, data3, data4], start=1):
            res = run_classification_methods("function_Word2Vec", data)
            fill_results(f"Base {i}", res, "Word2Vec", results_word2vec)

        results_path = results_dir / "results_word2vec.json"
        with open(results_path, "w", encoding="utf-8") as f:
            json.dump(results_word2vec, f, indent=4, ensure_ascii=False)
        logging.info(f"Results saved to: {results_path}")

    else:
        logging.info("Generating plots from existing JSON files...")
        results_bow = load_json("results/results_bow.json")
        results_tfidf = load_json("results/results_tfidf.json")
        results_word2vec = load_json("results/results_word2vec.json")

    logging.info("Starting to generate plots...")
    for metric in ["f1_score", "accuracy", "precision", "recall"]:
        plot_heatmap_metric(results_bow, results_tfidf, results_word2vec, metric=metric)
        plot_radar_metric_per_dataset(
            results_bow, results_tfidf, results_word2vec, metric=metric
        )


if __name__ == "__main__":
    main()

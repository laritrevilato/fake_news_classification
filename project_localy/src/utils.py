import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
from classification_method import (
    logistic_regression_classifier,
    multinomial_nb_classifier,
    random_forest_classifier,
    svc_classifier,
)
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from representation_method import (
    bow_representation,
    tfidf_representation,
    word2vec_representation,
)
from sklearn.model_selection import train_test_split


def preprocess_text(text):
    """
    Tokenizes, lowercases, removes stopwords, and lemmatizes the input text.
    """

    tokens = word_tokenize(text)
    tokens = [token.lower() for token in tokens]

    stop_words = set(stopwords.words("portuguese"))
    tokens = [token for token in tokens if token.isalnum() and token not in stop_words]

    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(token) for token in tokens]

    return " ".join(tokens)


def apply_representation_method(method_name, dataframe):
    """
    Maps a method name to the respective representation function and applies it.
    """

    method_map = {
        "BOW": bow_representation,
        "TFIDF": tfidf_representation,
        "function_Word2Vec": word2vec_representation,
    }

    if method_name not in method_map:
        raise ValueError(f"Unknown method: {method_name}")

    return {
        "representation_method": method_name,
        "representation": method_map[method_name](dataframe),
    }


def run_classification_methods(representation, dataframe):
    """
    Given a representation method and dataframe, splits data, runs classifiers,
    and returns results.
    """

    result = apply_representation_method(representation, dataframe)
    x_full, y_full = result["representation"]

    x_train, x_test, y_train, y_test = train_test_split(
        x_full, y_full, test_size=0.2, random_state=52, stratify=y_full
    )

    results = []

    classifiers = {
        "SVC": svc_classifier,
        "LogisticRegression": logistic_regression_classifier,
        "MultinomialNB": multinomial_nb_classifier,
        "RandomForest": random_forest_classifier,
    }

    for clf_name, clf_func in classifiers.items():
        results.append(
            {
                "method_representation": representation,
                "method_classification": clf_name,
                "result": clf_func(x_train, y_train, x_test, y_test),
            }
        )

    return results


def fill_results(dataset_key, base_results, representation_method, target_dict):
    """
    Updates the target dictionary with classification results for a dataset and representation.
    """

    if representation_method not in target_dict[dataset_key]:
        target_dict[dataset_key][representation_method] = {}

    for item in base_results:
        classifier = item["method_classification"]
        result_data = item["result"]

        target_dict[dataset_key][representation_method][classifier] = {
            "model": str(result_data["model"]),
            "accuracy": result_data["accuracy"],
            "precision": result_data["precision"],
            "recall": result_data["recall"],
            "f1_score": result_data["f1_score"],
            # "confusion_matrix": result_data["confusion_matrix"],
        }


def save_plot(filename, subfolder=""):
    """
    Saves the current matplotlib plot to the specified folder with dpi 300.
    """

    target_folder = os.path.join("results", subfolder)
    os.makedirs(target_folder, exist_ok=True)

    full_path = os.path.join(target_folder, filename)
    plt.savefig(full_path, dpi=300, bbox_inches="tight")
    plt.clf()


def load_json(file_path_str):
    """
    Loads a JSON file, raising an error if it does not exist.
    """

    path = Path(file_path_str)
    if not path.exists():
        raise FileNotFoundError(
            f"The file {path} does not exist. Run with --full mode first."
        )

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

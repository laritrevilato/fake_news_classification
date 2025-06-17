import logging
from pathlib import Path
from typing import Tuple

import git
import pandas as pd
from utils import preprocess_text

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def read_BoatosBR() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Clones the BoatosBR repository (if necessary), reads the JSON file,
    preprocesses the texts, and splits into DataFrames for true and fake news.

    Returns:
        Tuple containing two DataFrames:
        - df_BoatosBR_true: true news with columns 'FullText' and 'Classe' = 1
        - df_BoatosBR_fake: fake news with columns 'FullText' and 'Classe' = 0

    Raises RuntimeError if there is an error during reading or processing.
    """
    try:
        base_dir = Path("repo") / "BoatosBR"
        repo_BoatosBR = "https://github.com/Felipe-Harrison/boatos-br-corpus.git"
        file_path = base_dir / "base_simples" / "boatos_br_corpus_simples.json"

        # Clone the repository if the directory does not exist
        if not base_dir.exists():
            base_dir.parent.mkdir(
                parents=True, exist_ok=True
            )  # ensure 'repo' folder exists
            logging.info(f"Cloning BoatosBR repository into {base_dir}...")
            git.Repo.clone_from(repo_BoatosBR, str(base_dir))
            logging.info("Repository cloned successfully.")

        # Check if expected file exists
        if not file_path.exists():
            raise FileNotFoundError(f"Expected file not found: {file_path}")

        # Read JSON file using pandas
        df_BoatosBR_raw = pd.read_json(file_path)

        # Check if DataFrame is empty
        if df_BoatosBR_raw.empty:
            raise ValueError(f"DataFrame read from {file_path} is empty.")

        # Select relevant columns
        df_BoatosBR_raw = df_BoatosBR_raw[["texto", "rotulo"]]

        # Filter classes
        df_BoatosBR_true = df_BoatosBR_raw[
            df_BoatosBR_raw["rotulo"] == "verdade"
        ].copy()
        df_BoatosBR_fake = df_BoatosBR_raw[df_BoatosBR_raw["rotulo"] == "falso"].copy()

        # Apply preprocessing to text
        df_BoatosBR_true["texto"] = df_BoatosBR_true["texto"].apply(preprocess_text)
        df_BoatosBR_fake["texto"] = df_BoatosBR_fake["texto"].apply(preprocess_text)

        # Create final DataFrames with standardized structure
        df_BoatosBR_true = pd.DataFrame(
            {"FullText": df_BoatosBR_true["texto"], "Classe": 1}
        )

        df_BoatosBR_fake = pd.DataFrame(
            {"FullText": df_BoatosBR_fake["texto"], "Classe": 0}
        )

        # Verify that DataFrames are not empty after filtering
        if df_BoatosBR_true.empty or df_BoatosBR_fake.empty:
            raise ValueError("True or fake class DataFrames are empty after filtering.")

        logging.info("BoatosBR corpus loaded successfully.")

        return df_BoatosBR_true, df_BoatosBR_fake

    except Exception as e:
        logging.error(f"Error reading BoatosBR data: {e}")
        raise RuntimeError(f"Error reading BoatosBR data: {e}")

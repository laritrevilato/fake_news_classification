import logging
import os
from pathlib import Path

import git
import pandas as pd
from utils import preprocess_text

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def read_Fakebr():
    """
    Clone the Fake.br repository if needed, read true and fake news text files,
    preprocess texts and return two DataFrames for true and fake news.

    Returns:
        Tuple of DataFrames (df_true, df_fake), each with columns 'FullText' and 'Classe' (1 for true, 0 for fake).

    Raises RuntimeError in case of any processing errors.
    """
    try:
        base_dir = Path("repo") / "Fakebr"
        repo_url = "https://github.com/roneysco/Fake.br-Corpus.git"

        true_dir = base_dir / "full_texts" / "true"
        fake_dir = base_dir / "full_texts" / "fake"

        # Clone repo if not already present
        if not base_dir.exists():
            base_dir.parent.mkdir(parents=True, exist_ok=True)
            logging.info(f"Cloning Fake.br repository into {base_dir}...")
            git.Repo.clone_from(repo_url, str(base_dir))
            logging.info("Repository cloned successfully.")

        # Check if required directories exist
        if not true_dir.exists() or not fake_dir.exists():
            raise FileNotFoundError(
                "Fake.br corpus processing error: 'true' or 'fake' directories not found in cloned repo."
            )

        true_news = []
        fake_news = []

        # Load true news
        true_files = sorted(os.listdir(true_dir))
        if not true_files:
            raise ValueError(
                "Fake.br corpus processing error: No files found in true news directory."
            )

        for filename in true_files:
            with open(true_dir / filename, "r", encoding="utf-8") as f:
                text = f.read()
                true_news.append((preprocess_text(text), 1))

        # Load fake news
        fake_files = sorted(os.listdir(fake_dir))
        if not fake_files:
            raise ValueError(
                "Fake.br corpus processing error: No files found in fake news directory."
            )

        for filename in fake_files:
            with open(fake_dir / filename, "r", encoding="utf-8") as f:
                text = f.read()
                fake_news.append((preprocess_text(text), 0))

        # Create DataFrames
        df_true = pd.DataFrame(true_news, columns=["FullText", "Classe"])
        df_fake = pd.DataFrame(fake_news, columns=["FullText", "Classe"])

        # Final checks
        if df_true.empty or df_fake.empty:
            raise ValueError(
                "Fake.br corpus processing error: One of the DataFrames (true or fake) is empty after loading."
            )

        logging.info("Fake.br corpus loaded successfully.")

        return df_true, df_fake

    except Exception as e:
        logging.error(f"Error processing Fake.br corpus: {e}")
        raise RuntimeError(f"Error processing Fake.br corpus: {e}")

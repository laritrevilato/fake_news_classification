import logging
from pathlib import Path

import git
import pandas as pd
from utils import preprocess_text

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def read_FakeTrue():
    """
    Clone the FakeTrue repository if not present, load the CSV dataset,
    preprocess texts, and split into true and fake news DataFrames.

    Returns:
        Tuple of DataFrames (df_true, df_fake) with columns 'FullText' and 'Classe'.

    Raises RuntimeError on any loading or processing errors.
    """
    try:
        base_dir = Path("repo") / "FakeTrue"
        repo_url = "https://github.com/jpchav98/FakeTrue.Br.git"
        file_path = base_dir / "FakeTrueBr_corpus.csv"

        # Clone repository if it does not exist
        if not base_dir.exists():
            base_dir.parent.mkdir(parents=True, exist_ok=True)
            logging.info(f"Cloning FakeTrue repository into {base_dir}...")
            git.Repo.clone_from(repo_url, str(base_dir))
            logging.info("Repository cloned successfully.")

        if not file_path.exists():
            raise FileNotFoundError(f"Expected file not found: {file_path}")

        df_raw = pd.read_csv(file_path)

        expected_columns = {"title_fake", "fake", "true"}
        if not expected_columns.issubset(df_raw.columns):
            raise ValueError(
                f"Expected columns not found in file. Expected: {expected_columns}, Found: {set(df_raw.columns)}"
            )

        df_raw[["title_fake", "fake", "true"]] = df_raw[
            ["title_fake", "fake", "true"]
        ].astype(str)

        df_combined = pd.DataFrame(
            {
                "Fake": df_raw[["title_fake", "fake"]].agg(" ".join, axis=1),
                "True": df_raw["true"],
            }
        )

        df_combined["Fake"] = df_combined["Fake"].apply(preprocess_text)
        df_combined["True"] = df_combined["True"].apply(preprocess_text)

        df_true = pd.DataFrame({"FullText": df_combined["True"], "Classe": 1})
        df_fake = pd.DataFrame({"FullText": df_combined["Fake"], "Classe": 0})

        if df_true.empty or df_fake.empty:
            raise ValueError(
                "The FakeTrue corpus loaded but returned empty data for 'true' or 'fake' classes."
            )

        logging.info("FakeTrue corpus loaded successfully.")

        return df_true, df_fake

    except Exception as e:
        logging.error(f"Error reading FakeTrue data: {e}")
        raise RuntimeError(f"Error reading FakeTrue data: {e}")

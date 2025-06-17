import logging
from pathlib import Path

import git
import pandas as pd
from utils import preprocess_text

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def read_FakeRecogna():
    """
    Clone the FakeRecogna repository if necessary, load the dataset Excel file,
    preprocess texts and split into true and fake news DataFrames.

    Returns:
        Tuple of DataFrames (df_true, df_fake) with columns 'FullText' and 'Classe'.

    Raises RuntimeError on any loading or processing errors.
    """
    try:
        base_dir = Path("repo") / "FakeRecogna"
        repo_url = "https://github.com/Gabriel-Lino-Garcia/FakeRecogna.git"
        file_path = base_dir / "dataset" / "FakeRecogna.xlsx"

        # Clone repo if not exists
        if not base_dir.exists():
            base_dir.parent.mkdir(parents=True, exist_ok=True)
            logging.info(f"Cloning FakeRecogna repository into {base_dir}...")
            git.Repo.clone_from(repo_url, str(base_dir))
            logging.info("Repository cloned successfully.")

        if not file_path.exists():
            raise FileNotFoundError(f"Expected file not found: {file_path}")

        df_raw = pd.read_excel(file_path)

        # Check expected columns present
        expected_columns = {"Titulo", "Subtitulo", "Noticia", "Classe"}
        if not expected_columns.issubset(set(df_raw.columns)):
            raise ValueError(
                f"Expected columns missing in file. Expected: {expected_columns}, Found: {set(df_raw.columns)}"
            )

        df_raw = df_raw[list(expected_columns)].astype(str)
        df_raw["Classe"] = (
            pd.to_numeric(df_raw["Classe"], errors="coerce").fillna(-1).astype(int)
        )

        df = pd.DataFrame(
            {
                "FullText": df_raw[["Titulo", "Subtitulo", "Noticia"]].agg(
                    " ".join, axis=1
                ),
                "Classe": df_raw["Classe"],
            }
        )

        df["FullText"] = df["FullText"].apply(preprocess_text)

        df_true = df[df["Classe"] == 1].copy()
        df_fake = df[df["Classe"] == 0].copy()

        if df_true.empty or df_fake.empty:
            raise ValueError(
                "FakeRecogna corpus loaded but returned empty data for 'true' or 'fake' classes."
            )

        logging.info("FakeRecogna corpus loaded successfully.")

        return df_true, df_fake

    except Exception as e:
        logging.error(f"Error reading FakeRecogna data: {e}")
        raise RuntimeError(f"Error reading FakeRecogna data: {e}")

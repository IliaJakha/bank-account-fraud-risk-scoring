"""Load BAF Base dataset and split by chronological month index."""
from pathlib import Path
import pandas as pd

TARGET = "fraud_bool"
TIME = "month"


def load_base(path: str | Path) -> pd.DataFrame:
    """Read Base.csv and validate required benchmark columns and labels."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"BAF Base.csv not found: {path}")
    df = pd.read_csv(path)
    missing = {TARGET, TIME}.difference(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")
    if df.empty or df[TARGET].isna().any() or not df[TARGET].isin([0, 1]).all():
        raise ValueError("Expected non-empty data with binary fraud_bool labels (0/1)")
    return df


def chronological_split(df: pd.DataFrame):
    """Months 0–5 train, 6 validation, 7 test; exclude month and target from features."""
    if not {TARGET, TIME}.issubset(df.columns):
        raise ValueError("Dataset needs 'fraud_bool' and 'month' columns")
    if not df[TIME].isin(range(8)).all():
        raise ValueError("Expected BAF month indices 0 through 7")
    train = df.loc[df[TIME] <= 5]
    val = df.loc[df[TIME] == 6]
    test = df.loc[df[TIME] == 7]
    if any(part.empty for part in (train, val, test)):
        raise ValueError("All train, validation, and test partitions must be non-empty")
    features = [c for c in df if c not in (TARGET, TIME)]
    return ((train[features], train[TARGET]),
            (val[features], val[TARGET]),
            (test[features], test[TARGET]))

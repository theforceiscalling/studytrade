"""
Generate sample synthetic data for NIFTY and BANKNIFTY spot and options,
saving to data/raw (CSV) and data/processed (Parquet) and registering in catalog.db.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from studytrade.data.fetchers.synthetic import SyntheticDataGenerator
from studytrade.data.store import DataStore


def generate_and_store_dataset(
    symbol: str = "NIFTY",
    start_date: str = "2024-01-01",
    n_days: int = 125,
    seed: int = 42,
    base_dir: str = "data",
    spot_start: float | None = None,
    target_spot_end: float | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Generate synthetic spot and option chain data, saving to CSV and Parquet.

    Returns (spot_df, options_df).
    """
    gen = SyntheticDataGenerator(seed=seed)
    store = DataStore(base_dir)

    # 1. Spot data
    spot_kwargs = {}
    if spot_start is not None:
        spot_kwargs["spot_start"] = spot_start
    if target_spot_end is not None:
        spot_kwargs["target_spot_end"] = target_spot_end
    spot_df = gen.generate_spot(symbol=symbol, start_date=start_date, n_days=n_days, **spot_kwargs)
    end_date_str = spot_df["timestamp"].max().strftime("%Y-%m-%d")

    # 2. Options data across the spot history
    options_df = gen.get_options(
        underlying_symbol=symbol,
        start_date=start_date,
        end_date=end_date_str,
        spot_df=spot_df,
    )

    # 3. Save raw CSVs
    raw_dir = Path(base_dir) / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    sym_lower = symbol.lower()
    spot_df.to_csv(raw_dir / f"{sym_lower}_spot.csv", index=False)
    options_df.to_csv(raw_dir / f"{sym_lower}_options.csv", index=False)

    # 4. Save to Parquet store
    store.save_spot(spot_df, symbol)
    store.save_options(options_df, symbol)

    return spot_df, options_df


def main() -> None:
    print("Generating comprehensive synthetic dataset for NIFTY and BANKNIFTY (125 days)...")
    for sym in ["NIFTY", "BANKNIFTY"]:
        print(f"Generating {sym}...")
        spot, opt = generate_and_store_dataset(symbol=sym, start_date="2024-01-01", n_days=125)
        spot_min = spot['close'].min()
        spot_max = spot['close'].max()
        print(f"  {sym} Spot: {len(spot)} rows ({spot_min:.1f} - {spot_max:.1f})")
        print(f"  {sym} Options: {len(opt)} quotes saved.")

    store = DataStore("data")
    catalog = store.get_catalog()
    print("\nDataset Catalog:")
    print(catalog[["dataset_type", "symbol", "start_date", "end_date", "row_count"]])


if __name__ == "__main__":
    main()

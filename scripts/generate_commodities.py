"""
StudyTrade — MCX Commodity Dataset Generator (CRUDEOILM & NATURALGAS)
======================================================================
Generates spot and multi-expiry option chains for CRUDEOILM and NATURALGAS
up to current date (2026-11-27), storing them in Parquet and catalog.db.
"""

from __future__ import annotations

import time
from pathlib import Path
import pandas as pd

from studytrade.constants import TIMEZONE
from studytrade.data.fetchers.synthetic import SyntheticDataGenerator
from studytrade.data.store import DataStore


def generate_commodity_datasets(base_dir: str = "data") -> None:
    data_path = Path(base_dir)
    store = DataStore(data_path)
    gen = SyntheticDataGenerator(seed=108)

    # Determine date range up to 2026-11-27
    start_date = "2024-01-01"
    end_date = "2026-11-27"
    bdate_range = pd.bdate_range(start=start_date, end=end_date, tz=TIMEZONE)
    n_days = len(bdate_range)
    print(f"[*] Generating commodity datasets from {start_date} to {end_date} ({n_days} trading days)...")

    # 1. CRUDEOILM (Calibrated to current MCX Futures ~₹7,735/bbl)
    print("\n--- Generating CRUDEOILM (Crude Oil Mini) ---")
    t0 = time.time()
    crude_spot = gen.generate_spot(
        symbol="CRUDEOILM",
        start_date=start_date,
        n_days=n_days,
        spot_start=6100.0,
        target_spot_end=7735.0,
        annual_vol=0.32,
    )
    print(f"[+] CRUDEOILM Spot: {len(crude_spot)} rows (Close: {crude_spot['close'].iloc[-1]:.1f}) in {time.time()-t0:.2f}s")
    
    t1 = time.time()
    crude_options = gen.get_options(
        underlying_symbol="CRUDEOILM",
        start_date=start_date,
        end_date=end_date,
        spot_df=crude_spot,
        num_strikes_each_side=30,  # Wider ladder so deep ITM 6400 CE up to 8500 CE are fully available
    )
    print(f"[+] CRUDEOILM Options: {len(crude_options)} quotes in {time.time()-t1:.2f}s")

    # Save CRUDEOILM
    store.save_spot(crude_spot, "CRUDEOILM")
    store.save_options(crude_options, "CRUDEOILM")
    print(f"[✓] Saved CRUDEOILM to {data_path / 'processed' / 'CRUDEOILM'}")

    # 2. NATURALGAS (Calibrated to current MCX Futures ~₹334.5/mmBtu)
    print("\n--- Generating NATURALGAS (Natural Gas) ---")
    t2 = time.time()
    ng_spot = gen.generate_spot(
        symbol="NATURALGAS",
        start_date=start_date,
        n_days=n_days,
        spot_start=210.0,
        target_spot_end=334.5,
        annual_vol=0.48,
    )
    print(f"[+] NATURALGAS Spot: {len(ng_spot)} rows (Close: {ng_spot['close'].iloc[-1]:.1f}) in {time.time()-t2:.2f}s")

    t3 = time.time()
    ng_options = gen.get_options(
        underlying_symbol="NATURALGAS",
        start_date=start_date,
        end_date=end_date,
        spot_df=ng_spot,
        num_strikes_each_side=25,
    )
    print(f"[+] NATURALGAS Options: {len(ng_options)} quotes in {time.time()-t3:.2f}s")

    # Save NATURALGAS
    store.save_spot(ng_spot, "NATURALGAS")
    store.save_options(ng_options, "NATURALGAS")
    print(f"[✓] Saved NATURALGAS to {data_path / 'processed' / 'NATURALGAS'}")

    # Also save raw CSV summaries
    raw_dir = data_path / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    crude_spot.to_csv(raw_dir / "crudeoilm_spot.csv", index=False)
    ng_spot.to_csv(raw_dir / "naturalgas_spot.csv", index=False)

    print("\n[✓] All commodity datasets generated and cataloged successfully!")
    catalog = store.get_catalog()
    print(catalog[catalog["symbol"].isin(["CRUDEOILM", "NATURALGAS"])][["symbol", "dataset_type", "start_date", "end_date", "row_count"]])


if __name__ == "__main__":
    generate_commodity_datasets()

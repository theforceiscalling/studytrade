#!/usr/bin/env bash
# ==============================================================================
# StudyTrade — Daily Automated Market Data Sync (NSE Bhavcopy)
# Runs automatically after market hours on weekdays (Monday - Friday).
# ==============================================================================
set -euo pipefail

APP_DIR="/opt/studytrade"
LOG_FILE="${APP_DIR}/data/daily_sync.log"

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting automated daily market data sync..." >> "${LOG_FILE}"

cd "${APP_DIR}"
if [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
fi

python3 -c "
import sys
from pathlib import Path
from datetime import datetime

data_dir = Path('${APP_DIR}/data')
try:
    # 1. NSE Equity Derivatives Bhavcopy
    from studytrade.data.bhavcopy import BhavcopyImporter
    importer = BhavcopyImporter(data_dir=data_dir, cache_dir=data_dir / 'bhavcopy')
    res = importer.sync_missing_days(['NIFTY', 'BANKNIFTY', 'FINNIFTY'])
    print(f'[{datetime.now()}] NSE Bhavcopy sync completed: {res}')

    # 2. MCX Commodity Derivatives & Spot (CRUDEOILM & NATURALGAS)
    from studytrade.data.auto_sync import sync_commodities_missing_days
    c_res = sync_commodities_missing_days(data_dir=data_dir)
    print(f'[{datetime.now()}] MCX Commodity sync completed: {c_res}')
except Exception as e:
    print(f'[{datetime.now()}] Daily sync error: {e}', file=sys.stderr)
" >> "${LOG_FILE}" 2>&1

# Ensure permissions remain accessible by web worker
APP_USER=$(stat -c '%U' "${APP_DIR}" 2>/dev/null || echo "ubuntu")
chown -R "${APP_USER}:${APP_USER}" "${APP_DIR}/data" 2>/dev/null || true
chmod -R 775 "${APP_DIR}/data" 2>/dev/null || true

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Daily market sync finished." >> "${LOG_FILE}"

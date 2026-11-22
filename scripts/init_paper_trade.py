from studytrade.paper.journal import PaperJournal
from datetime import datetime

journal = PaperJournal('data/paper_journal.db')
# Clear any prior mock open positions if any
conn = journal._get_conn() if hasattr(journal, '_get_conn') else None
import sqlite3
with sqlite3.connect('data/paper_journal.db') as conn:
    conn.execute("DELETE FROM paper_open_positions")
    conn.commit()

pos_id = journal.add_open_position(
    symbol="NIFTY26NOV26100CE",
    underlying_symbol="NIFTY",
    strike=26100.0,
    expiry="2026-11-26",
    option_type="CE",
    n_lots=1,
    lot_size=25,
    entry_price=14.80,
    current_price=14.80,
    entry_iv=11.20,
    strategy="Gamma Squeeze Manual",
    entry_time=datetime.now()
)

print(f"Added paper open position ID: {pos_id}")
positions = journal.get_open_positions()
print("Current open positions:")
for p in positions:
    print(p)

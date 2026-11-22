import sqlite3

conn = sqlite3.connect('data/paper_journal.db')
cur = conn.cursor()
cur.execute("DELETE FROM paper_open_positions")
conn.commit()

cur.execute("SELECT COUNT(*) FROM paper_open_positions")
count = cur.fetchone()[0]
print(f"Open positions count in paper_open_positions: {count}")
conn.close()

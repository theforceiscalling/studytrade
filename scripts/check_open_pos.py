import sqlite3

conn = sqlite3.connect('data/paper_journal.db')
cur = conn.cursor()
try:
    cur.execute('SELECT * FROM paper_open_positions')
    rows = cur.fetchall()
    print('Open positions count:', len(rows))
    for r in rows:
        print(r)
except Exception as e:
    print('Error:', e)
conn.close()

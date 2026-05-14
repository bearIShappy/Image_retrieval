"""
Run once: marks all existing NULL-status images as ACTIVE in metadata.db
Run from project root: python fix_status.py
"""
import sqlite3, os

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "metadata.db")
conn = sqlite3.connect(DB_PATH)
c = conn.cursor()

# Count before
before = c.execute("SELECT COUNT(*) FROM dataset_images WHERE status IS NULL OR status = ''").fetchone()[0]
print(f"Images with NULL/empty status: {before}")

c.execute("UPDATE dataset_images SET status = 'ACTIVE' WHERE status IS NULL OR status = ''")
conn.commit()

after = c.execute("SELECT COUNT(*) FROM dataset_images WHERE status = 'ACTIVE'").fetchone()[0]
print(f"Images now marked ACTIVE: {after}")
conn.close()
print("Done. Restart app.py.")
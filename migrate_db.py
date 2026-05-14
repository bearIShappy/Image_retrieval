"""
One-time migration: adds missing columns to metadata.db
Run from your project root: python migrate_db.py
"""
import sqlite3
import os

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "metadata.db")

print(f"Opening: {DB_PATH}")
conn = sqlite3.connect(DB_PATH)
c = conn.cursor()

# Check existing columns
c.execute("PRAGMA table_info(dataset_images)")
existing = {row[1] for row in c.fetchall()}
print(f"Existing columns: {existing}")

# Add any missing columns
migrations = [
    ("status",     "TEXT DEFAULT 'active'"),
    ("file_size",  "INTEGER"),
    ("checksum",   "TEXT"),
]

for col_name, col_def in migrations:
    if col_name not in existing:
        sql = f"ALTER TABLE dataset_images ADD COLUMN {col_name} {col_def}"
        c.execute(sql)
        print(f"  [+] Added column: {col_name}")
    else:
        print(f"  [=] Already exists: {col_name}")

conn.commit()
conn.close()
print("\nMigration complete. You can now restart app.py.")
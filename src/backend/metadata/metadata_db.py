"""
SQLite metadata store for the CLIP retrieval pipeline.

Handles everything that is NOT a vector:
  - dataset_images          : manifest of all indexed images
  - query_history           : search log
  - finetune_runs           : training run logs
  - class_prototypes        : serialised prototype tensors
  - pending_support_images  : upload queue (images awaiting training)
"""

import os
import json
import sqlite3
import pickle
import time
import torch
from typing import Dict, List, Optional, Any

from src.backend.metadata.db_config import (
    SQLITE_DB_PATH,
    ALLOWED_SUPPORT_CLASSES,
    VALID_IMAGE_STATUSES,
)


class MetadataDB:
    """SQLite metadata store for the CLIP retrieval pipeline."""

    def __init__(self, db_path: str = SQLITE_DB_PATH) -> None:
        """
        Open (or create) the SQLite database and ensure all tables exist.

        Args:
            db_path (str): Absolute path to the ``.db`` file.
                           Parent directories are created automatically.
        """
        print(f"[metadata_db] Opening SQLite at {db_path}")
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self._create_tables()

    # ── Internal helpers ──────────────────────────────────────────────────

    def _execute(self, query: str, params: tuple = ()):
        """
        Execute a SQL statement and return the cursor.

        Args:
            query (str): SQL using ``?`` placeholders.
            params (tuple): Bound parameters.

        Returns:
            sqlite3.Cursor: Executed cursor.

        Raises:
            Exception: Re-raised after rolling back on error.
        """
        cursor = self.conn.cursor()
        try:
            cursor.execute(query, params)
            return cursor
        except Exception as exc:
            self.conn.rollback()
            raise exc

    def _create_tables(self) -> None:
        """
        Create all required tables and indexes if they do not already exist.
        Runs schema migrations for columns added in later versions.
        """
        table_stmts = [
            """
            CREATE TABLE IF NOT EXISTS dataset_images (
                path        TEXT PRIMARY KEY,
                class       TEXT DEFAULT 'unknown',
                source      TEXT DEFAULT 'dataset',
                status      TEXT DEFAULT 'ACTIVE',
                image_name  TEXT,
                upload_date TEXT,
                width       INTEGER,
                height      INTEGER,
                n_regions   INTEGER DEFAULT 0,
                indexed_at  REAL,
                created_at  REAL,
                updated_at  REAL
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS pending_support_images (
                id            INTEGER PRIMARY KEY AUTOINCREMENT,
                path          TEXT UNIQUE NOT NULL,
                filename      TEXT NOT NULL,
                class         TEXT NOT NULL,
                upload_source TEXT,
                created_at    REAL,
                processed_at  REAL,
                error_message TEXT
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS query_history (
                id           INTEGER PRIMARY KEY AUTOINCREMENT,
                query_image  TEXT,
                text_query   TEXT,
                top_k        INTEGER,
                mode         TEXT,
                use_regions  INTEGER,
                results_json TEXT,
                time_ms      INTEGER,
                created_at   REAL
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS finetune_runs (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                epochs          INTEGER,
                batch_size      INTEGER,
                lr_backbone     REAL,
                lr_head         REAL,
                final_loss      REAL,
                checkpoint_path TEXT,
                n_classes       INTEGER,
                class_names     TEXT,
                created_at      REAL
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS class_prototypes (
                class_name TEXT PRIMARY KEY,
                embedding  BLOB,
                n_images   INTEGER,
                updated_at REAL
            )
            """,
        ]

        index_stmts = [
            "CREATE INDEX IF NOT EXISTS idx_images_source ON dataset_images(source)",
            "CREATE INDEX IF NOT EXISTS idx_images_class  ON dataset_images(class)",
            "CREATE INDEX IF NOT EXISTS idx_images_status ON dataset_images(status)",
            "CREATE INDEX IF NOT EXISTS idx_history_time  ON query_history(created_at)",
        ]

        c = self.conn.cursor()
        for stmt in table_stmts + index_stmts:
            c.execute(stmt.strip())
        self.conn.commit()
        print("[metadata_db] Tables and indexes ensured.")
        self._migrate_schema()

    def _migrate_schema(self) -> None:
        """Add columns introduced after the initial schema (no-op if already present)."""
        c = self.conn.cursor()
        c.execute("PRAGMA table_info(dataset_images)")
        columns = {row[1] for row in c.fetchall()}
        migrations = []
        if "status" not in columns:
            migrations.append(
                "ALTER TABLE dataset_images ADD COLUMN status TEXT DEFAULT 'ACTIVE'"
            )
        if "created_at" not in columns:
            migrations.append(
                "ALTER TABLE dataset_images ADD COLUMN created_at REAL"
            )
        if "updated_at" not in columns:
            migrations.append(
                "ALTER TABLE dataset_images ADD COLUMN updated_at REAL"
            )
        if "image_name" not in columns:
            migrations.append(
                "ALTER TABLE dataset_images ADD COLUMN image_name TEXT"
            )
        if "upload_date" not in columns:
            migrations.append(
                "ALTER TABLE dataset_images ADD COLUMN upload_date TEXT"
            )
        for stmt in migrations:
            try:
                c.execute(stmt)
                print(f"[metadata_db] Migration: {stmt}")
            except Exception as exc:
                print(f"[metadata_db] Migration skipped ({exc})")
        if migrations:
            self.conn.commit()

    # ── Dataset images ────────────────────────────────────────────────────

    def upsert_image(
        self,
        path: str,
        cls: str = "unknown",
        source: str = "TRAINING",
        width: Optional[int] = None,
        height: Optional[int] = None,
        n_regions: int = 0,
        status: str = "ACTIVE",
        image_name: Optional[str] = None,
        upload_date: Optional[str] = None,
    ) -> None:
        """
        Insert or replace an image record in ``dataset_images``.

        If ``image_name`` is not provided, it is derived from the file path.
        If ``upload_date`` is not provided, it defaults to the current
        ISO-8601 timestamp for freshly uploaded images.

        Args:
            path (str): Absolute filesystem path (used as primary key).
            cls (str): Class label.
            source (str): ``TRAINING``, ``SUPPORT``, or ``TEST``.
            width (int, optional): Image width in pixels.
            height (int, optional): Image height in pixels.
            n_regions (int): Number of region crops indexed.
            status (str): ``ACTIVE``, ``PENDING``, or ``FAILED``.
            image_name (str, optional): Original filename of the uploaded image.
            upload_date (str, optional): ISO-8601 date/time string of the upload.
        """
        from datetime import datetime, timezone

        # Derive image_name from path if not explicitly provided
        if image_name is None:
            image_name = os.path.basename(path)

        # Preserve existing upload_date on re-upsert (don't overwrite with None)
        if upload_date is None:
            existing = self._execute(
                "SELECT upload_date FROM dataset_images WHERE path = ?", (path,)
            ).fetchone()
            if existing and existing["upload_date"]:
                upload_date = existing["upload_date"]
            # else: stays None — the image was indexed, not uploaded

        now = time.time()
        self._execute(
            """INSERT OR REPLACE INTO dataset_images
               (path, class, source, status, image_name, upload_date,
                width, height, n_regions, indexed_at, updated_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (path, cls, source, status, image_name, upload_date,
             width, height, n_regions, now, now),
        )
        self.conn.commit()

    def upsert_images_batch(self, images: List[dict]) -> None:
        """
        Upsert multiple image records in a single transaction.

        Args:
            images (list[dict]): Dicts with keys matching ``upsert_image`` args.
        """
        for img in images:
            self.upsert_image(
                path=img["path"],
                cls=img.get("class", "unknown"),
                source=img.get("source", "dataset"),
                width=img.get("width"),
                height=img.get("height"),
                n_regions=img.get("n_regions", 0),
                status=img.get("status", "ACTIVE"),
                image_name=img.get("image_name"),
                upload_date=img.get("upload_date"),
            )

    def get_images(
        self,
        source: Optional[str] = None,
        cls: Optional[str] = None,
        status: Optional[str] = None,
    ) -> List[dict]:
        """
        Query image records with optional filters.

        Args:
            source (str, optional): Filter by source label.
            cls (str, optional): Filter by class name.
            status (str, optional): Filter by status.

        Returns:
            list[dict]: Matching rows as plain dicts.
        """
        sql = "SELECT * FROM dataset_images WHERE 1=1"
        params: List[Any] = []
        if source:
            sql += " AND source = ?"
            params.append(source)
        if cls:
            sql += " AND class = ?"
            params.append(cls)
        if status:
            sql += " AND status = ?"
            params.append(status)
        cursor = self._execute(sql, tuple(params))
        return [dict(r) for r in cursor.fetchall()]

    def delete_images_by_source(self, source: str) -> None:
        """
        Delete all image records for a given source.

        Args:
            source (str): Source label to delete (e.g. ``"TEST"``).
        """
        self._execute("DELETE FROM dataset_images WHERE source = ?", (source,))
        self.conn.commit()

    def image_count(self, source: Optional[str] = None, status: Optional[str] = None) -> int:
        """
        Count image records with optional filters.

        Args:
            source (str, optional): Filter by source.
            status (str, optional): Filter by status.

        Returns:
            int: Row count.
        """
        sql = "SELECT COUNT(*) FROM dataset_images WHERE 1=1"
        params: List[Any] = []
        if source:
            sql += " AND source = ?"
            params.append(source)
        if status:
            sql += " AND status = ?"
            params.append(status)
        cursor = self._execute(sql, tuple(params))
        return cursor.fetchone()[0]

    # ── Class validation ──────────────────────────────────────────────────

    def validate_class_name(self, class_name: str) -> bool:
        """
        Return True if ``class_name`` is in the allowed fixed classes.

        Args:
            class_name (str): Class label to validate.

        Returns:
            bool: True when valid.
        """
        return class_name in ALLOWED_SUPPORT_CLASSES

    def get_allowed_classes(self) -> List[str]:
        """Return a copy of the fixed allowed class list."""
        return ALLOWED_SUPPORT_CLASSES.copy()

    # ── Upload metadata queries ───────────────────────────────────────────

    def get_upload_metadata(
        self,
        source: Optional[str] = None,
        cls: Optional[str] = None,
        limit: int = 200,
    ) -> List[dict]:
        """
        Return upload metadata for images that have an ``upload_date``.

        Each returned dict contains:
          - image_path
          - image_name
          - image_class
          - upload_date  (ISO-8601 string)
          - source
          - status

        Args:
            source (str, optional): Filter by source label.
            cls (str, optional): Filter by class name.
            limit (int): Max rows to return.

        Returns:
            list[dict]: Upload metadata ordered by upload_date DESC.
        """
        sql = (
            "SELECT path, image_name, class, upload_date, source, status "
            "FROM dataset_images WHERE upload_date IS NOT NULL"
        )
        params: List[Any] = []
        if source:
            sql += " AND source = ?"
            params.append(source)
        if cls:
            sql += " AND class = ?"
            params.append(cls)
        sql += " ORDER BY upload_date DESC LIMIT ?"
        params.append(limit)
        cursor = self._execute(sql, tuple(params))
        return [
            {
                "image_path": r["path"],
                "image_name": r["image_name"],
                "image_class": r["class"],
                "upload_date": r["upload_date"],
                "source": r["source"],
                "status": r["status"],
            }
            for r in cursor.fetchall()
        ]

    # ── Pending support images ────────────────────────────────────────────

    def add_pending_image(self, path: str, filename: str, class_name: str) -> bool:
        """
        Register an uploaded image as PENDING (not yet trained/indexed).

        Args:
            path (str): Filesystem path of the staged file.
            filename (str): Original upload filename.
            class_name (str): Target class (must be in allowed classes).

        Returns:
            bool: True on success, False if the class is invalid or insert fails.
        """
        if not self.validate_class_name(class_name):
            print(f"[metadata_db] REJECT: class '{class_name}' not in allowed classes")
            return False
        try:
            self._execute(
                """INSERT INTO pending_support_images
                   (path, filename, class, upload_source, created_at)
                   VALUES (?, ?, ?, ?, ?)""",
                (path, filename, class_name, "web", time.time()),
            )
            self.conn.commit()
            return True
        except Exception as exc:
            print(f"[metadata_db] Failed to add pending image: {exc}")
            return False

    def get_pending_images(self, cls: Optional[str] = None) -> List[dict]:
        """
        Return all pending (unprocessed) uploaded images.

        Args:
            cls (str, optional): Filter by class name.

        Returns:
            list[dict]: Pending image records ordered by upload time.
        """
        sql = "SELECT * FROM pending_support_images WHERE processed_at IS NULL"
        params: List[Any] = []
        if cls:
            sql += " AND class = ?"
            params.append(cls)
        sql += " ORDER BY created_at ASC"
        cursor = self._execute(sql, tuple(params))
        return [dict(r) for r in cursor.fetchall()]

    def pending_image_count(self, cls: Optional[str] = None) -> int:
        """
        Count pending (unprocessed) uploaded images.

        Args:
            cls (str, optional): Filter by class name.

        Returns:
            int: Count of pending images.
        """
        sql = "SELECT COUNT(*) FROM pending_support_images WHERE processed_at IS NULL"
        params: List[Any] = []
        if cls:
            sql += " AND class = ?"
            params.append(cls)
        return self._execute(sql, tuple(params)).fetchone()[0]

    def mark_pending_processed(
        self, path: str, success: bool = True, error_msg: Optional[str] = None
    ) -> None:
        """
        Mark a pending image as processed (success or failure).

        Args:
            path (str): Path of the pending image.
            success (bool): True clears the error_message field.
            error_msg (str, optional): Error detail when ``success=False``.
        """
        self._execute(
            """UPDATE pending_support_images
               SET processed_at = ?, error_message = ?
               WHERE path = ?""",
            (time.time(), None if success else error_msg, path),
        )
        self.conn.commit()

    def clear_pending_images(self) -> None:
        """Delete all already-processed pending image records."""
        self._execute("DELETE FROM pending_support_images WHERE processed_at IS NOT NULL")
        self.conn.commit()

    def activate_images(self, paths: List[str]) -> int:
        """
        Transition images from PENDING → ACTIVE after successful training.

        Args:
            paths (list[str]): Filesystem paths of images to activate.

        Returns:
            int: Number of records updated.
        """
        count = 0
        for path in paths:
            self._execute(
                "UPDATE dataset_images SET status = ?, updated_at = ? WHERE path = ?",
                ("ACTIVE", time.time(), path),
            )
            count += 1
        self.conn.commit()
        return count

    def fail_images(self, paths: List[str]) -> int:
        """
        Mark images as FAILED after a training error.

        Args:
            paths (list[str]): Filesystem paths of images to mark failed.

        Returns:
            int: Number of records updated.
        """
        count = 0
        for path in paths:
            self._execute(
                "UPDATE dataset_images SET status = ?, updated_at = ? WHERE path = ?",
                ("FAILED", time.time(), path),
            )
            count += 1
        self.conn.commit()
        return count

    # ── Query history ─────────────────────────────────────────────────────

    def log_query(
        self,
        query_image: Optional[str] = None,
        text_query: Optional[str] = None,
        top_k: int = 5,
        mode: str = "global",
        use_regions: bool = False,
        results: Optional[list] = None,
        time_ms: int = 0,
    ) -> None:
        """
        Log a search query to the query history table.

        Args:
            query_image (str, optional): Path to the query image.
            text_query (str, optional): Text query string.
            top_k (int): Number of results requested.
            mode (str): Search mode label.
            use_regions (bool): Whether region-level search was used.
            results (list, optional): Result dicts to serialise as JSON.
            time_ms (int): Query latency in milliseconds.
        """
        self._execute(
            """INSERT INTO query_history
               (query_image, text_query, top_k, mode, use_regions,
                results_json, time_ms, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (query_image, text_query, top_k, mode, int(use_regions),
             json.dumps(results or []), time_ms, time.time()),
        )
        self.conn.commit()

    def get_query_history(self, limit: int = 50) -> List[dict]:
        """
        Return recent search queries ordered by recency.

        Args:
            limit (int): Maximum number of rows to return.

        Returns:
            list[dict]: Query history rows.
        """
        cursor = self._execute(
            "SELECT * FROM query_history ORDER BY created_at DESC LIMIT ?",
            (limit,),
        )
        return [dict(r) for r in cursor.fetchall()]

    # ── Finetune runs ─────────────────────────────────────────────────────

    def log_finetune(
        self,
        epochs: int,
        batch_size: int,
        lr_backbone: float,
        lr_head: float,
        final_loss: float,
        checkpoint_path: str,
        n_classes: int,
        class_names: list,
    ) -> None:
        """
        Persist a finetune run record.

        Args:
            epochs (int): Number of training epochs.
            batch_size (int): Batch size used.
            lr_backbone (float): Learning rate for the backbone.
            lr_head (float): Learning rate for the classification head.
            final_loss (float): Final training loss value.
            checkpoint_path (str): Path to the saved ``.pt`` checkpoint.
            n_classes (int): Number of classes trained.
            class_names (list[str]): Class name list.
        """
        self._execute(
            """INSERT INTO finetune_runs
               (epochs, batch_size, lr_backbone, lr_head, final_loss,
                checkpoint_path, n_classes, class_names, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (epochs, batch_size, lr_backbone, lr_head, final_loss,
             checkpoint_path, n_classes, json.dumps(class_names), time.time()),
        )
        self.conn.commit()

    def get_finetune_runs(self, limit: int = 20) -> List[dict]:
        """
        Return recent finetune run records.

        Args:
            limit (int): Maximum number of rows to return.

        Returns:
            list[dict]: Finetune run rows.
        """
        cursor = self._execute(
            "SELECT * FROM finetune_runs ORDER BY created_at DESC LIMIT ?",
            (limit,),
        )
        return [dict(r) for r in cursor.fetchall()]

    # ── Class prototypes ──────────────────────────────────────────────────

    def save_prototype(
        self, class_name: str, embedding: torch.Tensor, n_images: int
    ) -> None:
        """
        Persist a class prototype embedding.

        Args:
            class_name (str): Class label (primary key).
            embedding (torch.Tensor): Prototype vector.
            n_images (int): Number of images averaged to produce the prototype.
        """
        blob = pickle.dumps(embedding.detach().cpu())
        self._execute(
            """INSERT OR REPLACE INTO class_prototypes
               (class_name, embedding, n_images, updated_at)
               VALUES (?, ?, ?, ?)""",
            (class_name, blob, n_images, time.time()),
        )
        self.conn.commit()

    def load_prototypes(self) -> Dict[str, torch.Tensor]:
        """
        Load all class prototypes from SQLite.

        Returns:
            dict[str, torch.Tensor]: Mapping of class name → prototype tensor.
        """
        cursor = self._execute("SELECT * FROM class_prototypes")
        return {
            r["class_name"]: pickle.loads(r["embedding"])
            for r in cursor.fetchall()
        }

    def clear_prototypes(self) -> None:
        """Delete all stored class prototype records."""
        self.conn.execute("DELETE FROM class_prototypes")
        self.conn.commit()

    # ── Stats ─────────────────────────────────────────────────────────────

    def stats(self) -> dict:
        """
        Return a summary of record counts across all tables.

        Returns:
            dict: Count summary safe to log or return via API.
        """
        def _cnt(table: str, source: Optional[str] = None) -> int:
            sql = f"SELECT COUNT(*) FROM {table} WHERE 1=1"
            params: List[Any] = []
            if source:
                sql += " AND source = ?"
                params.append(source)
            return self._execute(sql, tuple(params)).fetchone()[0]

        return {
            "total_images":   _cnt("dataset_images"),
            "dataset_images": _cnt("dataset_images", "TRAINING"),
            "support_images": _cnt("dataset_images", "SUPPORT"),
            "test_images":    _cnt("dataset_images", "TEST"),
            "pending_uploads": _cnt("pending_support_images"),
        }

    def close(self) -> None:
        """Close the SQLite connection."""
        self.conn.close()

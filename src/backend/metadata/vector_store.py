"""
Qdrant vector store wrapper for CLIP embeddings.

Connection strategy (in priority order):
  1. Docker / remote server: tries to reach ``QDRANT_HOST:QDRANT_PORT``.
  2. Local binary fallback: if Docker is not reachable, launches
     ``qdrant/qdrant.exe`` (relative to the project root) automatically
     and connects to it on 127.0.0.1:6333.
  3. On-disk local mode: used only when ``QDRANT_HOST`` is explicitly
     set to an empty string (offline / embedded mode).

All calls are synchronous (qdrant_client REST/gRPC).
"""

import os
import time
import uuid
import pickle
import shutil
import socket
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Any

import torch

from qdrant_client import QdrantClient
from qdrant_client.models import (
    VectorParams, Distance, PointStruct, Filter,
    FieldCondition, MatchValue, SearchParams, PayloadSchemaType,
)

from src.backend.metadata.db_config import (
    QDRANT_PATH, QDRANT_COLLECTION, EMBEDDING_DIM,
    QDRANT_HOST, QDRANT_PORT,
)

# ---------------------------------------------------------------------------
# Path to the bundled qdrant.exe (relative to the project root).
# Override via env var QDRANT_EXE_PATH if needed.
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parents[3]  # …/Image_retrival
_DEFAULT_QDRANT_EXE = _PROJECT_ROOT / "qdrant" / "qdrant.exe"
QDRANT_EXE_PATH: Path = Path(
    os.environ.get("QDRANT_EXE_PATH", str(_DEFAULT_QDRANT_EXE))
)

# Storage directory used by the local binary
QDRANT_STORAGE_DIR: Path = _PROJECT_ROOT / "qdrant_data"

# Holds the Popen handle so the process is not garbage-collected
_qdrant_process: Optional[subprocess.Popen] = None  # type: ignore[type-arg]


def _is_docker_running() -> bool:
    """Return True if the Docker daemon is accessible on this system.

    Checks for the ``docker`` executable and then runs a fast ``docker info``
    probe.  Returns False on any error so the caller can fall through to the
    binary fallback.

    Returns:
        bool: True when Docker is running and responsive.
    """
    if shutil.which("docker") is None:
        return False
    try:
        result = subprocess.run(
            ["docker", "info", "--format", "{{.ServerVersion}}"],
            capture_output=True,
            timeout=5,
        )
        return result.returncode == 0
    except Exception:
        return False


def _is_port_open(host: str, port: int, timeout: float = 1.0) -> bool:
    """Return True if a TCP connection to *host*:*port* succeeds.

    Args:
        host (str): Hostname or IP address.
        port (int): TCP port number.
        timeout (float): Socket connect timeout in seconds.

    Returns:
        bool: True when the port is accepting connections.
    """
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def _launch_qdrant_exe(
    exe_path: Path,
    storage_dir: Path,
    port: int = 6333,
    grpc_port: int = 6334,
    startup_wait: float = 5.0,
) -> subprocess.Popen:  # type: ignore[type-arg]
    """Launch the local ``qdrant.exe`` binary as a background process.

    Args:
        exe_path (Path): Absolute path to ``qdrant.exe``.
        storage_dir (Path): Directory used by Qdrant for persistent storage.
        port (int): HTTP REST port (default 6333).
        grpc_port (int): gRPC port (default 6334).
        startup_wait (float): Seconds to wait for the port to become available.

    Returns:
        subprocess.Popen: The running child process handle.

    Raises:
        FileNotFoundError: If ``exe_path`` does not exist.
        RuntimeError: If Qdrant does not start within *startup_wait* seconds.
    """
    if not exe_path.is_file():
        raise FileNotFoundError(
            f"[vector_store] qdrant.exe not found at '{exe_path}'. "
            "Please place the binary there or set QDRANT_EXE_PATH."
        )

    storage_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        str(exe_path),
        "--uri", f"http://0.0.0.0:{port}",
    ]

    # Environment variables that qdrant.exe honours
    env = os.environ.copy()
    env["QDRANT__STORAGE__STORAGE_PATH"] = str(storage_dir)
    env["QDRANT__SERVICE__HTTP_PORT"] = str(port)
    env["QDRANT__SERVICE__GRPC_PORT"] = str(grpc_port)

    print(
        f"[vector_store] Docker not running — launching local qdrant.exe:\n"
        f"               exe     : {exe_path}\n"
        f"               storage : {storage_dir}\n"
        f"               port    : {port}"
    )

    proc = subprocess.Popen(
        cmd,
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        creationflags=(
            subprocess.CREATE_NO_WINDOW  # Windows only — hide console window
            if sys.platform == "win32"
            else 0
        ),
    )

    # Wait until the REST port is accepting connections
    deadline = time.time() + startup_wait
    while time.time() < deadline:
        if _is_port_open("127.0.0.1", port):
            print(
                f"[vector_store] qdrant.exe ready on 127.0.0.1:{port} "
                f"(pid={proc.pid})"
            )
            return proc
        time.sleep(0.5)

    proc.terminate()
    raise RuntimeError(
        f"[vector_store] qdrant.exe did not become ready within "
        f"{startup_wait}s on port {port}."
    )


def _det_id(seed: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, seed))

def _t2l(t: torch.Tensor) -> list:
    return t.detach().cpu().squeeze().tolist()

def _l2t(v: list) -> torch.Tensor:
    return torch.tensor(v, dtype=torch.float32).unsqueeze(0)


class VectorStore:
    """
    Qdrant-backed vector store for CLIP image/region embeddings.

    Connects via host:port (Docker or standalone server) by default.
    Falls back to an on-disk local instance only when ``host`` is
    explicitly falsy (empty string or None).
    """

    def __init__(
        self,
        path: str = QDRANT_PATH,
        collection: str = QDRANT_COLLECTION,
        dim: int = EMBEDDING_DIM,
        host: str = QDRANT_HOST,
        port: int = QDRANT_PORT,
        retries: int = 5,
        retry_delay: float = 2.0,
    ) -> None:
        """
        Initialise the VectorStore and ensure the Qdrant collection exists.

        Connection strategy (applied in order):

        1. **Docker / remote** — if ``host`` is set and the port is reachable,
           connect directly.
        2. **Local binary fallback** — if ``host`` is set but the port is *not*
           reachable AND Docker is not running, auto-launch ``qdrant.exe`` from
           ``qdrant/qdrant.exe`` (relative to the project root) and connect to
           it on ``127.0.0.1:6333``.
        3. **On-disk local mode** — when ``host`` is explicitly empty / falsy.

        Args:
            path (str): Local on-disk path (used only when ``host`` is falsy).
            collection (str): Qdrant collection name.
            dim (int): Embedding dimension (512 for CLIP ViT-B/32).
            host (str): Qdrant server hostname / IP.
            port (int): Qdrant server port.
            retries (int): Number of connection attempts before raising.
            retry_delay (float): Seconds to wait between retries.
        """
        global _qdrant_process  # keep the child process alive module-wide

        self.collection = collection
        self.dim = dim
        self.client: Optional[QdrantClient] = None

        if host:
            # ── Strategy 1: try the configured host:port first ────────────────
            print(f"[vector_store] Connecting to Qdrant at {host}:{port} ...")
            if _is_port_open(host, port):
                # Port is already open — connect straight away
                self.client = QdrantClient(host=host, port=port, timeout=10)
                self.client.get_collections()  # quick ping
                print(f"[vector_store] Connected to Qdrant ({host}:{port})")
            else:
                # ── Strategy 2: port not open — decide whether to use Docker or
                #    fall back to the local qdrant.exe binary ──────────────────
                if _is_docker_running():
                    # Docker is up but Qdrant container is not yet ready;
                    # retry with the normal retry loop.
                    print(
                        "[vector_store] Docker is running but Qdrant port is "
                        "not yet open — retrying ..."
                    )
                    for attempt in range(1, retries + 1):
                        try:
                            self.client = QdrantClient(
                                host=host, port=port, timeout=10
                            )
                            self.client.get_collections()
                            print(
                                f"[vector_store] Connected to Qdrant "
                                f"({host}:{port}) on attempt {attempt}"
                            )
                            break
                        except Exception as exc:
                            print(
                                f"[vector_store] Attempt {attempt}/{retries} "
                                f"failed: {exc}"
                            )
                            if attempt < retries:
                                time.sleep(retry_delay)
                            else:
                                raise RuntimeError(
                                    f"Cannot connect to Qdrant at {host}:{port}"
                                    f" after {retries} attempts. "
                                    "Is the Docker container running? "
                                    "Try: docker compose -f "
                                    "docker/docker-compose.yml up -d qdrant"
                                ) from exc
                else:
                    # ── Docker is NOT running — launch the local qdrant.exe ───
                    print(
                        "[vector_store] Docker is NOT running. "
                        "Falling back to local qdrant.exe ..."
                    )
                    _qdrant_process = _launch_qdrant_exe(
                        exe_path=QDRANT_EXE_PATH,
                        storage_dir=QDRANT_STORAGE_DIR,
                        port=port,
                    )
                    # Connect on loopback regardless of the original host value
                    self.client = QdrantClient(
                        host="127.0.0.1", port=port, timeout=10
                    )
                    self.client.get_collections()  # final ping
                    print(
                        f"[vector_store] Connected to local qdrant.exe "
                        f"(127.0.0.1:{port})"
                    )
        else:
            # ── Strategy 3: on-disk local mode (host intentionally empty) ────
            print(f"[vector_store] Using local on-disk Qdrant at '{path}'")
            self.client = QdrantClient(path=path)

        self._ensure_collection()

    def _ensure_collection(self) -> None:
        """
        Create the Qdrant collection if needed, or open the existing one.

        Guards against a startup race: qdrant.exe may report an empty
        collection list while still loading persisted storage. If
        create_collection raises 409 Conflict the collection already exists;
        catch it and fall through to the info log.
        """
        from qdrant_client.http.exceptions import UnexpectedResponse

        cols = [c.name for c in self.client.get_collections().collections]
        if self.collection not in cols:
            try:
                self.client.create_collection(
                    collection_name=self.collection,
                    vectors_config=VectorParams(size=self.dim, distance=Distance.COSINE),
                )
                for f in ("source", "class", "path"):
                    self.client.create_payload_index(
                        self.collection, f, PayloadSchemaType.KEYWORD
                    )
                self.client.create_payload_index(
                    self.collection, "is_region", PayloadSchemaType.BOOL
                )
                print(f"[vector_store] Created collection '{self.collection}'")
            except UnexpectedResponse as exc:
                if exc.status_code == 409:
                    print(
                        f"[vector_store] Collection '{self.collection}' "
                        "already exists (409 race) \u2014 opening existing."
                    )
                else:
                    raise
        info = self.client.get_collection(self.collection)
        print(f"[vector_store] Opened '{self.collection}' ({info.points_count} pts)")

    def reset(self):
        self.client.delete_collection(self.collection)
        self._ensure_collection()

    def upsert_image(self, path, global_embedding, regions=None, cls="unknown", source="TRAINING", created_at=None):
        import time
        created_at = created_at or time.time()
        points = [PointStruct(
            id=_det_id(f"global:{path}"),
            vector=_t2l(global_embedding),
            payload={"path": path, "class": cls, "is_region": False, "source": source, "created_at": created_at},
        )]
        for r in (regions or []):
            bbox = r["bbox"]
            points.append(PointStruct(
                id=_det_id(f"region:{path}:{bbox}"),
                vector=_t2l(r["embedding"]),
                payload={"path": path, "class": cls, "is_region": True,
                         "bbox": list(bbox), "area_ratio": r.get("area_ratio", 0.0), "source": source, "created_at": created_at},
            ))
        self.client.upsert(collection_name=self.collection, points=points)
        return len(points)

    def upsert_batch(self, entries, batch_size=100):
        total = 0
        buf = []
        for e in entries:
            import time
            path, cls, source = e["path"], e.get("cls", "unknown"), e.get("source", "dataset")
            created_at = e.get("created_at", time.time())
            buf.append(PointStruct(
                id=_det_id(f"global:{path}"), vector=_t2l(e["global_embedding"]),
                payload={"path": path, "class": cls, "is_region": False, "source": source, "created_at": created_at},
            ))
            for r in (e.get("regions") or []):
                bbox = r["bbox"]
                buf.append(PointStruct(
                    id=_det_id(f"region:{path}:{bbox}"), vector=_t2l(r["embedding"]),
                    payload={"path": path, "class": cls, "is_region": True,
                             "bbox": list(bbox), "area_ratio": r.get("area_ratio", 0.0), "source": source, "created_at": created_at},
                ))
            if len(buf) >= batch_size:
                self.client.upsert(collection_name=self.collection, points=buf)
                total += len(buf)
                buf = []
        if buf:
            self.client.upsert(collection_name=self.collection, points=buf)
            total += len(buf)
        return total

    def delete_by_source(self, source):
        self.client.delete(collection_name=self.collection,
            points_selector=Filter(must=[FieldCondition(key="source", match=MatchValue(value=source))]))

    def search(self, query_vector, top_k=20, source_filter=None, class_filter=None,
               regions_only=False, globals_only=False, score_threshold=None,
               from_date=None, to_date=None):
        must = []
        if source_filter:
            must.append(FieldCondition(key="source", match=MatchValue(value=source_filter)))
        if class_filter:
            must.append(FieldCondition(key="class", match=MatchValue(value=class_filter)))
        if regions_only:
            must.append(FieldCondition(key="is_region", match=MatchValue(value=True)))
        if globals_only:
            must.append(FieldCondition(key="is_region", match=MatchValue(value=False)))
            
        if from_date is not None or to_date is not None:
            from qdrant_client.models import Range
            range_kwargs = {}
            if from_date is not None: range_kwargs["gte"] = from_date
            if to_date is not None: range_kwargs["lte"] = to_date
            must.append(FieldCondition(key="created_at", range=Range(**range_kwargs)))
            
        qf = Filter(must=must) if must else None
        response = self.client.query_points(
            collection_name=self.collection, query=_t2l(query_vector),
            query_filter=qf, limit=top_k, score_threshold=score_threshold,
            search_params=SearchParams(exact=False, hnsw_ef=128),
        )
        hits = response.points
        return [{"path": h.payload["path"], "score": h.score,
                 "class": h.payload.get("class", "unknown"),
                 "is_region": h.payload.get("is_region", False),
                 "bbox": tuple(h.payload["bbox"]) if h.payload.get("bbox") else None,
                 "area_ratio": h.payload.get("area_ratio"),
                 "source": h.payload.get("source", "dataset"),
                 "created_at": h.payload.get("created_at")} for h in hits]

    def search_multi(self, query_matrix, top_k=20, **kwargs):
        per_path = {}
        for i in range(query_matrix.shape[0]):
            for h in self.search(query_matrix[i:i+1], top_k=top_k*2, **kwargs):
                p = h["path"]
                if p not in per_path or h["score"] > per_path[p]["score"]:
                    per_path[p] = h
        return sorted(per_path.values(), key=lambda x: x["score"], reverse=True)[:top_k]

    def count(self, source=None):
        if source:
            r = self.client.count(collection_name=self.collection, exact=True,
                count_filter=Filter(must=[FieldCondition(key="source", match=MatchValue(value=source))]))
            return r.count
        return self.client.get_collection(self.collection).points_count

    def get_all_global_vectors(self, source=None):
        must = [FieldCondition(key="is_region", match=MatchValue(value=False))]
        if source:
            must.append(FieldCondition(key="source", match=MatchValue(value=source)))
        result, offset = {}, None
        while True:
            recs, offset = self.client.scroll(
                collection_name=self.collection, scroll_filter=Filter(must=must),
                limit=500, with_vectors=True, offset=offset)
            for r in recs:
                result[r.payload["path"]] = _l2t(r.vector)
            if not recs or offset is None:
                break
        return result

    def get_image_entry(self, path):
        recs, _ = self.client.scroll(
            collection_name=self.collection,
            scroll_filter=Filter(must=[FieldCondition(key="path", match=MatchValue(value=path))]),
            limit=500, with_vectors=True)
        if not recs:
            return None
        entry = {"global_embedding": None, "regions": []}
        for r in recs:
            if not r.payload.get("is_region", False):
                entry["global_embedding"] = _l2t(r.vector)
            else:
                entry["regions"].append({"bbox": tuple(r.payload["bbox"]),
                    "area_ratio": r.payload.get("area_ratio", 0.0), "embedding": _l2t(r.vector)})
        return entry if entry["global_embedding"] is not None else None

    def delete_by_path(self, path: str) -> None:
        """
        Remove all Qdrant points (global + region) that match the given file path.

        Args:
            path (str): Absolute filesystem path used as the payload filter key.
        """
        from qdrant_client.models import FilterSelector
        self.client.delete(
            collection_name=self.collection,
            points_selector=FilterSelector(
                filter=Filter(
                    must=[FieldCondition(key="path", match=MatchValue(value=path))]
                )
            ),
        )

    def delete_by_source(self, source: str) -> None:
        """
        Remove all Qdrant points that belong to the given source label.

        Args:
            source (str): e.g. ``'TEST'``, ``'SUPPORT'``, ``'TRAINING'``.
        """
        from qdrant_client.models import FilterSelector
        self.client.delete(
            collection_name=self.collection,
            points_selector=FilterSelector(
                filter=Filter(
                    must=[FieldCondition(key="source", match=MatchValue(value=source))]
                )
            ),
        )


def migrate_pkl_to_qdrant(pkl_path, store, source="TRAINING"):
    if not os.path.isfile(pkl_path):
        print(f"[migrate] Pickle not found: {pkl_path}")
        return 0
    print(f"[migrate] Loading {pkl_path} ...")
    with open(pkl_path, "rb") as f:
        index = pickle.load(f)
    print(f"[migrate] {len(index)} images to migrate")
    entries = []
    for path, entry in index.items():
        parent = os.path.basename(os.path.dirname(path))
        cls = parent if parent else "unknown"
        if isinstance(entry, dict) and "global_embedding" in entry:
            entries.append({"path": path, "global_embedding": entry["global_embedding"],
                "regions": entry.get("regions", []), "cls": cls, "source": source})
        else:
            entries.append({"path": path, "global_embedding": entry,
                "regions": [], "cls": cls, "source": source})
    total = store.upsert_batch(entries, batch_size=200)
    print(f"[migrate] Migrated {total} points from {pkl_path}")
    return total

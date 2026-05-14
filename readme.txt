# CLIP ViT Few-Shot Retrieval Project

This project implements a region-aware image retrieval system optimized for aerial and parachute datasets. It uses selective fine-tuning of the CLIP visual backbone to handle small, specialized datasets where generic zero-shot performance might be insufficient.

---

## 📂 Project Structure

Dataset_extraction/
├── app.py                   # FastAPI backend entry point
├── run_retrieval.py         # CLI pipeline entry point
├── migrate_to_qdrant.py     # One-shot pkl → Qdrant migration
├── embeddings.pkl           # Legacy cache (backup after migration)
├── finetuned_clip.pt        # Saved weights after fine-tuning
├── qdrant_data/             # Qdrant on-disk vector store
├── metadata.db              # SQLite metadata database
├── cleaned_dataset/         # Support dataset (sorted by class)
├── dataset/                 # Main image dataset
├── test_dataset/            # Test images
├── models/
│   └── clip-vit-b-32/       # Local CLIP model weights
└── src/
    ├── __init__.py
    ├── backend/              # CLIP pipeline modules
    │   ├── __init__.py
    │   ├── clip_model.py     # CLIP wrapper and encoding methods
    │   ├── finetune.py       # Few-shot fine-tuning logic
    │   ├── indexer.py        # Dataset indexing (Qdrant + SQLite backed)
    │   ├── retriever.py      # ANN search + re-ranking (Qdrant or brute-force)
    │   ├── detector.py       # Region proposals + one-shot detection
    │   └── visualize.py      # Result rendering and bbox drawing
    ├── db/                   # Database layer
    │   ├── __init__.py
    │   ├── db_config.py      # Paths, collection names, DB constants
    │   ├── vector_store.py   # Qdrant wrapper (upsert, ANN search, migrate)
    │   └── metadata_db.py    # SQLite (manifest, query history, prototypes)
    └── frontend/             # React + TypeScript UI (Vite)
        ├── src/
        │   ├── App.tsx
        │   └── components/
        ├── package.json
        ├── vite.config.ts
        └── index.html

---

## 🏗️ Architecture

The system operates on a Fine-tune -> Index -> Retrieve architecture:

1. Few-Shot Fine-tuning (src/backend/finetune.py)
   Instead of training the whole model, we use a selective adaptation strategy:
   - Frozen Layers: Most of the CLIP Vision Transformer is frozen to preserve general knowledge.
   - Active Layers: Only the last 2 transformer blocks and the final LayerNorm are updated.
   - Classification Head: A temporary linear layer is used during training to force the visual backbone to learn the specific features of your classes (e.g., para motor vs static line jump).
   - In-Place Update: Weights are updated directly on the visual backbone; standard encode methods automatically use the specialized weights.

2. Prototype Building
   For each class in your support set, the system:
   - Encodes all available images.
   - Computes a Class Prototype (the average vector of all images in that class).
   - L2-normalizes the result for cosine similarity matching.

3. Retrieval Pipeline (run_retrieval.py)
   - Query Analysis: Generates multiple embeddings (Global + Multi-scale + Regions).
   - Region-Aware Scoring: 0.7 * max_region_sim + 0.3 * global_sim.
   - Prototype Re-ranking: Re-scores top results based on class prototype affinity.

4. Storage Layer
   - Qdrant: Vector embeddings stored as points with JSON payloads. ANN search ~10× faster than brute-force.
   - SQLite: Dataset manifest, query history, finetune run logs, class prototypes.
   - Disk: Images stay on disk. DB stores paths, not blobs.

---

## 🚀 How to Run

### 1. Start the Backend (FastAPI + Qdrant + SQLite)
cd "d:\Jasleen space\Dataset_extraction"
.\venv\Scripts\activate
python app.py

Backend starts at http://localhost:5000.
On first startup, auto-migrates embeddings.pkl into Qdrant if the vector store is empty.

### 2. Start the Frontend (React + Vite)   
cd "d:\Jasleen space\Dataset_extraction\src\frontend"
npm install        # first time only
npm run dev

Frontend starts at http://localhost:5173 (Vite default).

### 3. CLI Retrieval (with Qdrant ANN search)
python run_retrieval.py --query "path/to/query.jpg" --dataset "cleaned_dataset" --support "cleaned_dataset" --skip_finetune --use_regions

### 4. CLI Retrieval (legacy brute-force, no Qdrant)
python run_retrieval.py --query "path/to/query.jpg" --dataset "cleaned_dataset" --support "cleaned_dataset" --skip_finetune --use_regions --no_qdrant

### 5. One-shot Migration (pkl → Qdrant)
python migrate_to_qdrant.py

---

## 📊 API Endpoints

| Endpoint               | Method | Description                           |
|------------------------|--------|---------------------------------------|
| /api/dataset           | GET    | List support + test images            |
| /api/search            | POST   | Image/text search (Qdrant or brute)   |
| /api/upload-support    | POST   | Upload support images                 |
| /api/upload-test       | POST   | Upload test images + rebuild index    |
| /api/finetune          | POST   | Rebuild prototypes                    |
| /api/rebuild-index     | POST   | Full index rebuild                    |
| /api/stats             | GET    | Qdrant + SQLite statistics            |
| /api/query-history     | GET    | Recent search history                 |
| /api/finetune-runs     | GET    | Fine-tune run logs                    |

---

## 🔧 Parameters
--query:       Path to the image you want to search with.
--text:        (Optional) Add a text hint like "paratrooper with green canopy".
--use_regions: Highly recommended. Enables bounding-box matching for smaller objects.
--top_k:       Number of results to show in the output image.
--no_qdrant:   Disable Qdrant ANN; use legacy pickle + brute-force.

Results are saved to "output.png" showing the query next to matches with highlighted bounding boxes.

---

MY METHOD :

Few-shot Fine-tuning (your planned system)
Fine-tuning: YES
Freezing: Most layers frozen
Train: last transformer blocks

Best balance for tiny datasets.


/ / / / our architecture is already much stronger than a basic CLIP demo because you have:

multiscale embeddings
prototype reranking
region-aware retrieval
few-shot adaptation
text+image querying
Qdrant ANN vector search
SQLite metadata tracking

which is close to real-world retrieval systems.




================================================================================
CLIP VISION SYSTEM - COMPREHENSIVE PROJECT GUIDE
================================================================================

1. PROJECT SUMMARY & "MY METHOD"
This is a region-aware image retrieval system optimized for specialized aerial 
datasets. Unlike basic CLIP demos, this project uses:
- Few-Shot Adaptation: Selective fine-tuning (Last 2 transformer blocks only).
- Freezing Strategy : Most layers frozen to preserve general CLIP knowledge.
- Why it's stronger: Combines multiscale embeddings, prototype re-ranking, 
  region-aware scoring, and Qdrant ANN search for real-world performance.

2. UI DASHBOARD & PARAMETERS
   A. Retrieval Tab (Search)
      - Inputs: Image file, Text prompt, or Hybrid (Image + Text).
      - Controls:
        * Top-K         : 1 to 100 results.
        * Mode          : Global, Multi-query, Region-aware, or Prototype.
        * Aggregation   : Max or Mean (combining region scores).
        * Threshold     : 0.00 to 1.00 similarity filter.
        * Toggles       : Region-aware Features, Use Fine-tuned Model.
   
   B. Main Dataset Tab (Viewer)
      - View complete manifest and manage/upload "Test" images.

   C. Few-shot Tab (Management)
      - Support Set : Upload images to define class-specific "Support" data.
      - Training    : "Train & Build Index" button to update the visual 
                      backbone and rebuild the vector store.

3. ARCHITECTURE & PIPELINE
- Phase 1 (Fine-tune): Updates the visual backbone using the Support Set.
- Phase 2 (Prototypes): Computes average vectors per class for re-ranking.
- Phase 3 (Index)    : Encodes images and upserts to Qdrant (Vector DB).
- Phase 4 (Retrieve) : 0.7 * max_region_sim + 0.3 * global_sim + re-ranking.

4. QUICK START GUIDE
   1. Start Vector DB: cd docker && docker-compose up -d qdrant
   2. Start Backend  : cd .. && .\venv\Scripts\activate && python app.py
   3. Start UI       : cd src/frontend && npm install && npm run dev

5. CORE PROJECT STRUCTURE
- /src/backend  : CLIP logic, detection, and fine-tuning.
- /src/db       : Qdrant (Vectors) and SQLite (Metadata).
- /src/frontend : React/TypeScript dashboard.
- /docker       : Container configuration for Qdrant.
- /dataset      : Main image storage (Support vs Test).
================================================================================

Parameter	Value	Where
Model	clip-vit-b-32	CLIPEncoder
Embedding dim	512	db_config.py
Top-K results	20 (default)	App.tsx params
Similarity threshold	0.5	App.tsx params
Fine-tune trigger	≥ 20 new images	FewShotManagementPage
Use regions	true (default)	search params
Aggregation	max (multi-region)	search params

Changes Made
1. metadata_db.py — Schema & Methods
Added image_name TEXT and upload_date TEXT columns to the dataset_images table definition
Added schema migration for existing databases (auto-adds the two columns on startup)
Updated upsert_image() to accept image_name and upload_date parameters. If image_name is not provided, it's auto-derived from the file path. If upload_date is not provided, it preserves any existing value (prevents re-indexing from wiping the original upload timestamp)
Updated upsert_images_batch() to forward the new fields
Added get_upload_metadata() method that queries images with upload_date IS NOT NULL, returning structured dicts with image_path, image_name, image_class, upload_date, source, status
2. app.py — Upload Endpoints
/api/upload-support: Now generates datetime.now(timezone.utc).isoformat() immediately after file save and passes image_name + upload_date to upsert_image()
/api/upload-main and /api/upload-test: Same — capture ISO-8601 timestamp per file and propagate through _index_image_list()
_index_image_list(): Extended with filenames and upload_dates dict parameters that get forwarded to upsert_images_batch()
New /api/upload-metadata endpoint: Returns all upload metadata filtered by optional source and cls query params
Metadata Fields Stored
Field	Column	Format	When Set
image_path	path (PK)	Absolute filesystem path	Always
upload_date	upload_date	ISO-8601 (2026-05-14T07:31:30.432364+00:00)	At upload time only
image_name	image_name	Original filename	At upload (or derived from path)
image_class	class	Class label	At upload (SUPPORT) or from folder (TEST)

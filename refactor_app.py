import os
import re

filepath = "app.py"
with open(filepath, "r", encoding="utf-8") as f:
    content = f.read()

# Remove ALLOWED_SUPPORT_CLASSES validation
import_stmt = "from src.backend.metadata.db_config import (\n    QDRANT_PATH, SQLITE_DB_PATH, LEGACY_EMBEDDINGS_PKL, LEGACY_TEST_PKL,\n    ALLOWED_SUPPORT_CLASSES,\n)"
new_import_stmt = "from src.backend.metadata.db_config import (\n    QDRANT_PATH, SQLITE_DB_PATH, LEGACY_EMBEDDINGS_PKL, LEGACY_TEST_PKL,\n)"
content = content.replace(import_stmt, new_import_stmt)

# Removing ALLOWED_SUPPORT_CLASSES validation in upload_support
validation_block = """        # VALIDATION: Class must be in allowed list
        if class_name not in ALLOWED_SUPPORT_CLASSES:
            failed.append({
                "filename": img.filename,
                "class": class_name,
                "reason": f"Unknown class. Allowed: {ALLOWED_SUPPORT_CLASSES}"
            })
            continue"""
content = content.replace(validation_block, "")

# Modify search endpoint to handle from_date and to_date
search_api_orig = """@app.post("/api/search")
async def search(
    image: UploadFile = File(None),
    text: str = Form(None),
    top_k: int = Form(5),
    mode: str = Form("global"),
    aggregation: str = Form("max"),
    threshold: float = Form(0.5),
    use_regions: str = Form("false"),
    use_finetuned: str = Form("false"),
    use_qdrant: str = Form("true"),
):"""

search_api_new = """@app.post("/api/search")
async def search(
    image: UploadFile = File(None),
    text: str = Form(None),
    top_k: int = Form(5),
    mode: str = Form("global"),
    aggregation: str = Form("max"),
    threshold: float = Form(0.5),
    use_regions: str = Form("false"),
    use_finetuned: str = Form("false"),
    use_qdrant: str = Form("true"),
    from_date: str = Form(None),
    to_date: str = Form(None),
):"""
content = content.replace(search_api_orig, search_api_new)

# Modify retrieval routing rules inside search
# TEXT SEARCH: f -> YES, r -> NO, both -> NO.
# IMAGE SEARCH: f -> YES, r -> YES, both -> YES.
# Region-aware retrieval must only work for IMAGE SEARCH.

retrieval_logic_orig = """        # Build query embedding (handles image-only, text-only, or both)
        query_matrix, _ = build_query_embedding(
            clip,
            image_path=query_path,
            text_query=text,
            use_regions=use_regions_bool,
        )

        # Retrieve — use Qdrant ANN or fallback to brute-force
        if use_qdrant_bool and vector_store is not None:
            results = retrieve_with_qdrant(
                query_matrix, vector_store, prototypes,
                top_k=top_k, use_regions=use_regions_bool,
            )"""

retrieval_logic_new = """        # Enforce region-aware rules: text search -> r=NO, both -> r=NO.
        if text and not image:
            use_regions_bool = False
        if text and image:
            use_regions_bool = False

        # Build query embedding (handles image-only, text-only, or both)
        query_matrix, _ = build_query_embedding(
            clip,
            image_path=query_path,
            text_query=text,
            use_regions=use_regions_bool,
        )

        # Apply date filters
        # Convert from_date and to_date to float timestamps if needed
        # Assuming they come as ISO strings or timestamps.
        from_ts = float(from_date) if from_date else None
        to_ts = float(to_date) if to_date else None

        # Retrieve — use Qdrant ANN or fallback to brute-force
        if use_qdrant_bool and vector_store is not None:
            results = retrieve_with_qdrant(
                query_matrix, vector_store, prototypes,
                top_k=top_k, use_regions=use_regions_bool,
                from_date=from_ts, to_date=to_ts,
            )"""

content = content.replace(retrieval_logic_orig, retrieval_logic_new)

with open(filepath, "w", encoding="utf-8") as f:
    f.write(content)
print("app.py refactored for new features.")

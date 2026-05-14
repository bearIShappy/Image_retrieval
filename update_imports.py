import os

replacements = {
    "src.backend.retrieval.retriever": "src.backend.retrieval.retriever",
    "src.backend.retrieval.clip_model": "src.backend.retrieval.clip_model",
    "src.backend.indexing.indexer": "src.backend.indexing.indexer",
    "src.backend.training.finetune": "src.backend.training.finetune",
    "src.backend.region_aware.detector": "src.backend.region_aware.detector",
    "src.backend.region_aware.visualize": "src.backend.region_aware.visualize",
    "src.backend.metadata.metadata_db": "src.backend.metadata.metadata_db",
    "src.backend.metadata.vector_store": "src.backend.metadata.vector_store",
    "src.backend.metadata.db_config": "src.backend.metadata.db_config",
    "src.backend.metadata": "src.backend.metadata",
}

for root, dirs, files in os.walk("."):
    if "venv" in dirs: dirs.remove("venv")
    if "node_modules" in dirs: dirs.remove("node_modules")
    if "__pycache__" in dirs: dirs.remove("__pycache__")
    if ".git" in dirs: dirs.remove(".git")
    for file in files:
        if file.endswith(".py"):
            path = os.path.join(root, file)
            with open(path, "r", encoding="utf-8") as f:
                content = f.read()
            
            new_content = content
            for old, new in replacements.items():
                new_content = new_content.replace(old, new)
                
            if new_content != content:
                print(f"Updated {path}")
                with open(path, "w", encoding="utf-8") as f:
                    f.write(new_content)

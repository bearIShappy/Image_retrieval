import os

files = [
    "app.py",
    "src/backend/metadata/metadata_db.py",
    "src/backend/metadata/vector_store.py",
    "src/backend/indexing/indexer.py",
    "src/backend/retrieval/retriever.py",
    "src/backend/training/finetune.py",
]

for filepath in files:
    if not os.path.exists(filepath):
        continue
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()

    # Add import DatasetType if missing, and replace "dataset" -> DatasetType.TRAINING, etc.
    # Actually, simpler to just replace the literal strings in context to avoid breaking other things
    # Let's replace:
    content = content.replace('source="dataset"', 'source="TRAINING"')
    content = content.replace("source='dataset'", 'source="TRAINING"')
    content = content.replace('source="support"', 'source="SUPPORT"')
    content = content.replace("source='support'", 'source="SUPPORT"')
    content = content.replace('source="test"', 'source="TEST"')
    content = content.replace("source='test'", 'source="TEST"')

    content = content.replace('count("dataset")', 'count("TRAINING")')
    content = content.replace('count("support")', 'count("SUPPORT")')
    content = content.replace('count("test")', 'count("TEST")')

    content = content.replace('image_count("dataset"', 'image_count("TRAINING"')
    content = content.replace('image_count("support"', 'image_count("SUPPORT"')
    content = content.replace('image_count("test"', 'image_count("TEST"')

    content = content.replace('delete_by_source("dataset")', 'delete_by_source("TRAINING")')
    content = content.replace('delete_by_source("support")', 'delete_by_source("SUPPORT")')
    content = content.replace('delete_by_source("test")', 'delete_by_source("TEST")')

    content = content.replace('delete_images_by_source("dataset")', 'delete_images_by_source("TRAINING")')
    content = content.replace('delete_images_by_source("support")', 'delete_images_by_source("SUPPORT")')
    content = content.replace('delete_images_by_source("test")', 'delete_images_by_source("TEST")')
    
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"Updated {filepath}")

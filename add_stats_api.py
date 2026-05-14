import os

filepath = "app.py"
with open(filepath, "r", encoding="utf-8") as f:
    content = f.read()

# Add /api/stats endpoint if it doesn't exist
if "/api/stats" not in content:
    stats_api = """
@app.get("/api/stats")
async def get_stats():
    # Provide simple high-level stats for the UI DatasetOverview
    stats = {
        "sqlite": {
            "total_images": metadata_db.image_count(),
            "support_images": metadata_db.image_count("SUPPORT"),
            "test_images": metadata_db.image_count("TEST"),
            "pending_uploads": metadata_db.count_pending_uploads(),
            "prototypes": len(prototypes)
        }
    }
    return stats
"""
    content += stats_api
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(content)
    print("Added /api/stats")
else:
    print("/api/stats already exists")

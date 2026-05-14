"""Quick check of Qdrant status."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.backend.metadata.vector_store import VectorStore
vs = VectorStore()
print(f"Total points:   {vs.count()}")
print(f"Dataset points: {vs.count('dataset')}")
print(f"Support points: {vs.count('support')}")
print(f"Test points:    {vs.count('test')}")
vs.client.close()

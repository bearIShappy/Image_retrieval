import requests

try:
    r = requests.get('http://localhost:5000/api/dataset')
    print("Dataset status:", r.status_code)
    print("Dataset response:", r.text[:200])
except Exception as e:
    print("Dataset error:", e)

try:
    r = requests.post('http://localhost:5000/api/search', data={"text": "para motor", "top_k": 5})
    print("Search status:", r.status_code)
    print("Search response:", r.text[:200])
except Exception as e:
    print("Search error:", e)

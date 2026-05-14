import requests
import io

url = "http://127.0.0.1:5000/api/upload-support"
files = [
    ('files', ('test_image_12345.png', io.BytesIO(b'dummy data'), 'image/png'))
]
data = {
    'classes': 'heavy drop'
}

response = requests.post(url, files=files, data=data)
print("Status Code:", response.status_code)
print("Response:", response.json())

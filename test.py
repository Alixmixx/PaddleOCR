import requests
import base64
import json

# Function to read an image and convert it to base64
def image_to_base64(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

# Replace with your API endpoint
API_URL = "http://127.0.0.1:8000/ocr"  # Change if your API runs on a different server
API_KEY = "mirinae"  # Use the same key as in your FastAPI code

# Image path
image_path = "images/IMG_4162.jpg"  # Replace with the path to the image you want to test

# Read and encode the image
encoded_image = image_to_base64(image_path)

# Payload
payload = {
    "images": [encoded_image, encoded_image],
    "params": {
        "use_angle_cls": False,
        "lang": "en"
    },
    "ocr_params": {
        "det": True,
        "rec": True
    }
}

# Headers with API key
headers = {
    "Content-Type": "application/json",
    "api_key": API_KEY
}

# Make the request to the OCR API
response = requests.post(API_URL, data=json.dumps(payload), headers=headers)

# Check and print the response
if response.status_code == 200:
    print("OCR Results:")
    print(json.dumps(response.json(), indent=4))
else:
    print(f"Failed to get a response: {response.status_code}")
    print(response.text)

import os
import requests
from urllib.parse import urlparse
from datetime import datetime

def fetch_image():
    url = input("Enter the image URL: ").strip()

    # Create directory if it doesn't exist
    folder = "Fetched_Images"
    os.makedirs(folder, exist_ok=True)

    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()  # Raise HTTPError for bad responses

        # Extract filename from URL or generate one
        parsed_url = urlparse(url)
        filename = os.path.basename(parsed_url.path)
        if not filename or '.' not in filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"image_{timestamp}.jpg"

        # Save image in binary mode
        filepath = os.path.join(folder, filename)
        with open(filepath, 'wb') as f:
            f.write(response.content)

        print(f"✅ Image saved as '{filename}' in '{folder}'")

    except requests.exceptions.RequestException as e:
        print(f"⚠️ Failed to fetch image: {e}")

if __name__ == "__main__":
    fetch_image()

import os
import requests
from PIL import Image
from io import BytesIO

# Create a directory for test images
if not os.path.exists('test_images'):
    os.makedirs('test_images')

# Sample image URLs for each class (updated with working URLs)
test_images = {
    'beetles': 'https://www.inaturalist.org/photos/58581770/original.jpg',
    'beal_miner': 'https://www.plantwise.org/KnowledgeBank/800x640/PMDG_97553.jpg',
    'leaf_spot': 'https://www.gardeningknowhow.com/wp-content/uploads/2020/11/palm-leaf-spot.jpg',
    'white_flies': 'https://www.planetnatural.com/wp-content/uploads/2012/12/whitefly-control.jpg'
}

def download_image(url, filename):
    try:
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        img = Image.open(BytesIO(response.content))
        img.save(f'test_images/{filename}.jpg')
        print(f"Successfully downloaded {filename}")
    except Exception as e:
        print(f"Error downloading {filename}: {str(e)}")

def main():
    print("Downloading sample test images...")
    for class_name, url in test_images.items():
        download_image(url, class_name)
    print("\nDownload complete! Images saved in 'test_images' directory")

if __name__ == "__main__":
    main() 
# Load the model's weights

import requests
import os
from tqdm import tqdm

# Define the base public key and folder path
base_public_key = 'https://disk.yandex.ru/d/PHTz886hgfpO0A'
folder_path = '/siamese_ruclip/train_results'

# Set destination directory to 'data/train_results'
destination_dir = os.path.join('data', 'train_results')
os.makedirs(destination_dir, exist_ok=True)

# Yandex.Disk API endpoint
resource_url = 'https://cloud-api.yandex.net/v1/disk/public/resources'
params = {
    'public_key': base_public_key,
    'path': folder_path,
    'fields': '_embedded.items'
}

try:
    response = requests.get(resource_url, params=params)
    response.raise_for_status()
    resource_info = response.json()

    if '_embedded' in resource_info and 'items' in resource_info['_embedded']:
        for item in resource_info['_embedded']['items']:
            file_name = item.get('name')
            download_url = item.get('file')
            if download_url:
                print(f'Downloading {file_name}...')
                file_response = requests.get(download_url, stream=True)
                file_response.raise_for_status()
                total_size = int(file_response.headers.get('Content-Length', 0))
                file_path = os.path.join(destination_dir, file_name)
                if os.path.isfile(file_path):
                    print(f"\nFile already exists: {file_path}")
                    continue

                # Use tqdm to display progress bar
                with open(file_path, 'wb') as file, tqdm(
                    desc=file_name,
                    total=total_size,
                    unit='iB',
                    unit_scale=True,
                    unit_divisor=1024,
                ) as bar:
                    for data in file_response.iter_content(chunk_size=1024):
                        size = file.write(data)
                        bar.update(size)
            else:
                print(f'Skipping {file_name}: No download URL available.')
        print('Download completed.')
    else:
        print("No items found in the specified folder.")
except requests.exceptions.HTTPError as http_err:
    print(f"HTTP error occurred: {http_err}")
except Exception as err:
    print(f"An error occurred: {err}")
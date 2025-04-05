import re
import asyncio
import aiohttp
from pathlib import Path
from mimetypes import guess_extension
import pandas as pd
from tqdm import tqdm

async def parallel_download_img(
    df: pd.DataFrame,
    url_col: str,
    img_id_regex: str,
    dataset_name: str,
    base_download_dir: Path,
    max_concurrency: int = 100,   # Maximum number of parallel downloads
    delay: float = 1.0            # Delay (in seconds) before each download
) -> list:
    """
    Downloads images using an image ID extracted from the URL column and returns a list of image file names.
    
    This function controls the download rate by:
      - Limiting parallel downloads via a semaphore (max_concurrency).
      - Adding a delay before each download to mimic human behavior.
    
    Regex Usage:
      - The `url_col` contains the image URL, from which the image ID is extracted.
      - The `img_id_regex` must include one capture group (e.g., r'/(\d+)\.jpg$') that extracts the ID.
      - If no match is found, the download for that row is skipped.
    
    Args:
        df: Source DataFrame.
        url_col: Column name with image URLs.
        img_id_regex: Regex pattern with one capture group to extract the ID from the URL.
        dataset_name: Folder to save images into.
        base_download_dir: Base directory for downloads.
        max_concurrency: Maximum number of parallel downloads.
        delay: Delay (in seconds) before each download.
    
    Returns:
        A list of downloaded image file names corresponding to each row in the DataFrame 
        (or None for rows that failed).
    """
    download_dir = Path(base_download_dir) / dataset_name
    download_dir.mkdir(parents=True, exist_ok=True)

    sem = asyncio.Semaphore(max_concurrency)

    async def fetch_image(session: aiohttp.ClientSession, url: str, save_path: Path) -> str:
        try:
            async with session.get(url, timeout=10) as response:
                response.raise_for_status()
                content_type = response.headers.get('Content-Type', '')
                ext = guess_extension(content_type.split(';')[0].strip()) or '.jpg'
                if ext == '.jpe':
                    ext = '.jpg'
                final_path = save_path.with_suffix(ext)
                data = await response.read()
                final_path.write_bytes(data)
                return final_path.name
        except Exception as e:
            print(f"Failed to download {url}: {e}")
            return None

    async def download_row(row, session) -> str:
        async with sem:
            await asyncio.sleep(delay)
            url = row[url_col]
            if pd.isna(url):
                return None

            match = re.search(img_id_regex, str(url))
            if not match:
                print(f"Failed to extract image ID from URL: {url}")
                return None

            image_id = match.group(1)
            return await fetch_image(session, url, download_dir / image_id)

    async with aiohttp.ClientSession() as session:
        tasks = [download_row(row, session) for _, row in df.iterrows()]
        results = [
            await coro for coro in tqdm(
                asyncio.as_completed(tasks),
                total=len(tasks),
                desc="Downloading images"
            )
        ]
    return results

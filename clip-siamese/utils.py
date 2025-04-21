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
    max_concurrency: int = 100,
    delay: float = 1.0
) -> list:
    """
    Downloads images in parallel, preserving the input DataFrame order.

    Args:
        df: DataFrame containing image URLs.
        url_col: Column name with image URLs.
        img_id_regex: Regex with one capture group to extract the image ID from each URL.
        dataset_name: Subfolder name under base_download_dir to save images.
        base_download_dir: Base directory for downloads.
        max_concurrency: Maximum simultaneous downloads.
        delay: Delay before each download to rate-limit.

    Returns:
        List of downloaded filenames or None, ordered to match the DataFrame rows.
    """
    download_dir = Path(base_download_dir) / dataset_name
    download_dir.mkdir(parents=True, exist_ok=True)

    sem = asyncio.Semaphore(max_concurrency)

    async def fetch_image(session: aiohttp.ClientSession, url: str, save_path: Path) -> str:
        try:
            async with session.get(url, timeout=10) as resp:
                resp.raise_for_status()
                content_type = resp.headers.get('Content-Type', '').split(';')[0].strip()
                ext = guess_extension(content_type) or '.jpg'
                if ext == '.jpe':
                    ext = '.jpg'
                file_path = save_path.with_suffix(ext)
                data = await resp.read()
                file_path.write_bytes(data)
                return file_path.name
        except Exception as e:
            print(f"Error downloading {url}: {e}")
            return None

    async def download_url(idx: int, url: str, session: aiohttp.ClientSession) -> tuple[int, str]:
        async with sem:
            await asyncio.sleep(delay)
            if pd.isna(url) or not isinstance(url, str):
                return idx, None

            match = re.search(img_id_regex, url)
            if not match:
                print(f"ID not found in URL: {url}")
                return idx, None

            image_id = match.group(1)
            filename = await fetch_image(session, url, download_dir / image_id)
            return idx, filename

    async with aiohttp.ClientSession() as session:
        tasks = [asyncio.create_task(download_url(i, row[url_col], session))
                 for i, row in df.iterrows()]

        pbar = tqdm(total=len(tasks), desc="Downloading images")
        for task in tasks:
            task.add_done_callback(lambda _: pbar.update())

        completed = await asyncio.gather(*tasks)

    # Build ordered results
    results = [None] * len(completed)
    for idx, fname in completed:
        results[idx] = fname

    return results

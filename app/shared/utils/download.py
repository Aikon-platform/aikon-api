import requests
from zipfile import ZipFile
from urllib.parse import urlparse

from .iiif import IIIFDownloader
from .img import download_images, MAX_SIZE
from .fileutils import has_content
from ..const import IMG_PATH


def download_dataset(dataset_src, dataset_path=None, dataset_ref=None):
    """
    Download a dataset from front
    """
    # if dataset_src is a URL
    if all([urlparse(dataset_src).scheme, urlparse(dataset_src).netloc]):
        try:
            # IIIF MANIFEST
            downloader = IIIFDownloader(dataset_src)
            downloader.run()
            dataset_path = downloader.get_dir_name()
            dataset_ref = downloader.manifest_id

        except Exception as e:
            # If the IIIF download fails, proceed with ZIP download
            if not dataset_path:
                dataset_path = IMG_PATH

            dataset_path.mkdir(parents=True, exist_ok=True)
            dataset_zip_path = dataset_path / "dataset.zip"

            with requests.get(dataset_src, stream=True) as r:
                r.raise_for_status()
                with open(dataset_zip_path, "wb") as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        f.write(chunk)

            with ZipFile(dataset_zip_path, "r") as zipObj:
                zipObj.extractall(dataset_path / dataset_ref)

            dataset_zip_path.unlink()

    elif type(dataset_src) is dict:
        # LIST OF URLS
        doc_ids = []
        for doc_id, url in dataset_src.items():
            try:
                doc_id = f"{dataset_ref}_{doc_id}"
                doc_ids.append(doc_id)
                if not has_content(f"{dataset_path}/{doc_id}/"):
                    download_images(url, doc_id, dataset_path, MAX_SIZE)
            except Exception as e:
                raise ImportError(f"Error downloading images: {e}")
        return dataset_path, dataset_ref, doc_ids

    return dataset_path, dataset_ref

"""
Many functions to manipulate files and directories
"""

import os
import shutil
import time
import orjson
import requests
import mimetypes

from datetime import datetime
from os.path import exists
from pathlib import Path
from slugify import slugify
from typing import Union, Optional, Set, List, Tuple, Generator, Iterable, Dict
from flask import Response
from stat import S_IFREG
from stream_zip import ZIP_32, stream_zip
import re

from .logging import console

TPath = Union[str, os.PathLike]


def xaccel_send_from_directory(directory: TPath, redirect: TPath, filename: TPath):
    """
    Send a file from a given directory using X-Accel-Redirect
    """
    try:
        directory = Path(directory)
        file_path = directory / Path(filename)

        assert file_path.is_relative_to(directory)
        redirect_path = Path(redirect) / file_path.relative_to(directory)

        content_length = file_path.stat().st_size
        content_type = mimetypes.guess_type(str(file_path))[0]
        filename = file_path.name

        if not content_length or not content_type or not filename:
            return None

        response = Response()
        response.headers["Content-Length"] = content_length
        response.headers["Content-Type"] = content_type
        response.headers[
            "Content-Disposition"
        ] = f"attachment; filename={slugify(str(filename))}"
        response.headers["X-Accel-Redirect"] = str(redirect_path)
        response.headers["X-Accel-Buffering"] = "yes"
        response.headers["X-Accel-Charset"] = "utf-8"
        return response

    except Exception:
        return None


def is_too_old(filepath: Path, max_days: int = 30) -> bool:
    """
    Check if a file is older than a given number of days
    """
    try:
        return (
            not filepath.exists()
            or (datetime.now() - datetime.fromtimestamp(filepath.stat().st_mtime)).days
            > max_days
        )
    except Exception:
        return False


def has_content(path, file_nb=None):
    path = Path(path)
    if not os.path.exists(path):
        create_dir(path)
        return False

    nb_of_files = len(os.listdir(path))
    if file_nb:
        return nb_of_files == file_nb
    return nb_of_files != 0


def create_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)
    return path


def check_dir(path: TPath) -> bool:
    """
    Check if a directory exists, if not create it

    Returns True if the directory existed, False otherwise
    """
    path = Path(path)
    if not path.exists():
        create_dir(path)
        return False
    return True


def create_dirs_if_not(paths: List[TPath]) -> List[TPath]:
    """
    Create directories if they do not exist
    """
    for path in paths:
        check_dir(path)
    return paths


def create_file_if_not(path: TPath) -> Path:
    """
    Create a file if it does not exist, return the path
    """
    path = Path(path)
    if not path.exists():
        path.touch(mode=0o666)
    return path


def create_files_if_not(paths: List[TPath]) -> List[TPath]:
    """
    Create files if they do not exist
    """
    for path in paths:
        create_file_if_not(path)
    return paths


def delete_path(path: TPath) -> bool:
    """
    Delete a file or directory

    Returns True if the path existed and was deleted, False otherwise
    """
    try:
        if path.is_file():
            path.unlink()
        elif path.is_dir():
            shutil.rmtree(path)
    except Exception as e:
        return False
    return True


def clear_dir(
    parent_dir: TPath,
    path_to_clear: str = "*",
    file_to_check: str = None,
    delete_anyway: bool = False,
) -> int:
    """
    Clear a directory of files older than a default number of days
    For folders, the first file (or file_to_check) is checked for age

    Args:
        parent_dir: The parent directory to clear
        path_to_clear: The path to clear (default: "*")
        file_to_check: The file in the directory whose age is checked (default: None)
        delete_anyway: If False, only delete files older than a default number of days (default: False)
    """
    cleared = 0
    if not parent_dir:
        return cleared

    for path in parent_dir.glob(path_to_clear):
        file = path
        if path.is_dir():
            file = path / file_to_check if file_to_check else next(path.iterdir())

        if delete_anyway or is_too_old(file):
            cleared += 1 if delete_path(path) else 0
    return cleared


def delete_empty_dirs(
    parent_dir: TPath, path_to_clear: str = "*", recursive: bool = False
) -> int:
    """
    Delete all empty directories in a given directory.

    Args:
        parent_dir: The parent directory to search for empty directories
        path_to_clear: The path pattern to match directories
        recursive: If True, recursively check subdirectories before parent directories

    Returns:
        int: Number of empty directories deleted
    """
    deleted_count = 0

    if not parent_dir or not parent_dir.exists():
        return deleted_count

    # glob pattern to list of paths
    paths = list(parent_dir.glob(path_to_clear))

    # Sort paths by depth (deepest first) if recursive
    if recursive:
        paths.sort(key=lambda p: len(p.parts), reverse=True)

    for path in paths:
        if not path.is_dir():
            continue

        # A directory is empty if it contains no files and no non-empty directories
        is_empty = True
        for item in path.iterdir():
            if item.is_file():
                is_empty = False
                break
            if item.is_dir() and any(item.iterdir()):
                is_empty = False
                break

        if is_empty:
            if delete_path(path):
                deleted_count += 1

    return deleted_count


def get_file_ext(filepath: TPath) -> str:
    """
    Get the extension of a file without the dot
    """
    path, ext = os.path.splitext(filepath)
    _, filename = os.path.split(path)
    return filename if ext else None, ext[1:] if ext else None


def sanitize_url(string: str) -> str:
    """
    Sanitize a URL to remove spaces
    """
    return string.replace(" ", "+").replace(" ", "+")


def sanitize_str(string: str) -> str:
    """
    Sanitize a URL string to make it a valid filename
    (remove http, https, www, /, ., :, spaces)
    """
    return (
        re.sub(r"^https?\:\/\/|www\.|\.|:|%|\s", "", string.strip())
        .replace("/", "^")
        .replace(" ", "_")
    )


def empty_file(path: TPath) -> None:
    """
    Clear the content of a file if it exists
    """
    if exists(path):
        open(path, "w").close()


def file_age(path: TPath = None) -> int:
    """
    Calculates and returns the age of a file in days based on its last modification time.

    :param path: Path to the file (default: __file__)
    """
    if path is None:
        path = __file__
    dt = datetime.now() - datetime.fromtimestamp(Path(path).stat().st_mtime)  # delta
    return dt.days  # + dt.seconds / 86400  # fractional days


def delete_directory(doc_dir):
    """
    Delete the directory

    Args:
        doc_dir: Directory to delete

    Returns:
        True if the directory existed and was deleted, False otherwise
    """
    path = Path(doc_dir)
    try:
        if os.path.isdir(path):
            shutil.rmtree(path)
            return True
        else:
            return False
    except Exception as e:
        console(f"An error has occurred when deleting {doc_dir} directory", e=e)
        return False


def download_file(url: str, filepath: TPath) -> None:
    """
    Download a file from a URL and save it to disk
    """
    try:
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            with open(filepath, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
    except Exception as e:
        console(
            f"Failed to download the file. Status code: {r.status_code}: {e}", "red"
        )


def process_directory(
    directory: str | Path,
    extensions: Optional[Set[str]] = None,
    exclude_dirs: Optional[Set[str]] = None,
    absolute_path: bool = False,
    find_first_only: bool = False,
) -> Union[List[Path], bool]:
    """
    Process a directory to either find all matching files or check for existence of matching files.

    Args:
        directory: Base directory path
        extensions: Optional set of extensions to filter files (e.g. {'.txt', '.pdf'})
        exclude_dirs: Optional set of directory names to exclude from search
        absolute_path: Return absolute path (only applies when returning list of files)
        find_first_only: If True, returns boolean indicating if any matching file exists
                        If False, returns list of all matching files

    Returns:
        If find_first_only is True: Boolean indicating if any matching file exists
        If find_first_only is False: List of Path objects for all matching files
    """
    if isinstance(directory, str):
        directory = Path(directory)

    if absolute_path:
        directory = directory.resolve()

    if not directory.exists():
        return False if find_first_only else []

    files = [] if not find_first_only else False

    try:
        for item in directory.rglob("*"):
            if exclude_dirs and any(p.name in exclude_dirs for p in item.parents):
                continue

            if item.is_file():
                if extensions is None or item.suffix.lower() in extensions:
                    if find_first_only:
                        return True
                    files.append(item)
    except PermissionError:
        # no permission to directories
        pass

    return False if find_first_only else sorted(files)


def get_all_files(
    directory: str | Path,
    extensions: Optional[Set[str]] = None,
    exclude_dirs: Optional[Set[str]] = None,
    absolute_path: bool = False,
) -> List[Path]:
    """
    Get all files in a directory and its subdirectories.

    Args:
        directory: Base directory path
        extensions: Optional set of extensions to filter files (e.g. {'.txt', '.pdf'})
        exclude_dirs: Optional set of directory names to exclude from search
        absolute_path: Return absolute path

    Returns:
        List of Path objects for all matching files
    """
    return process_directory(
        directory=directory,
        extensions=extensions,
        exclude_dirs=exclude_dirs,
        absolute_path=absolute_path,
        find_first_only=False,
    )


def check_if_file(
    directory: str | Path,
    extensions: Optional[Set[str]] = None,
    exclude_dirs: Optional[Set[str]] = None,
) -> bool:
    """
    Check if there is at least a file with a given extension in a directory.

    Args:
        directory: Base directory path
        extensions: Optional set of extensions to filter files (e.g. {'.txt', '.pdf'})
        exclude_dirs: Optional set of directory names to exclude from search

    Returns:
        Boolean value indicating whether at least one matching file exists
    """
    return process_directory(
        directory=directory,
        extensions=extensions,
        exclude_dirs=exclude_dirs,
        find_first_only=True,
    )


def zip_on_the_fly(files: List[Tuple[str, TPath]]) -> Iterable[bytes]:
    """
    Zip files on the fly

    Args:
        files: List of tuples (filename, path)
    """

    def contents(path: TPath) -> Generator[bytes, None, None]:
        with open(path, "rb") as f:
            while chunk := f.read(65536):
                yield chunk

    def iter_files() -> Generator[
        Tuple[str, int, int, int, Generator[bytes, None, None]], None, None
    ]:
        for name, path in files:
            if not os.path.exists(path):
                continue
            dt = datetime.fromtimestamp(os.path.getmtime(path))
            yield (name, dt, S_IFREG | 0o600, ZIP_32, contents(path))

    return stream_zip(iter_files())


def download_model_if_not(url: str | Dict[str, str], path: Path) -> Path:
    """
    Download a model if it does not exist

    Either URL for direct download or dictionary for Hugging Face Hub
    dict = {"repo_id": "user/model_repo", "filename": "model.pth"}
    Returns:
        Path to the model file
    """
    if not path.exists():
        try:
            if type(url) is str:
                download_file(url, path)
            else:
                from huggingface_hub import hf_hub_download

                hf_hub_download(local_dir=path.parent, **url)
        except Exception as e:
            console("Failed to download the model", e=e)
    return path


def get_model(model_stem, model_dir: Path):
    """
    Get the model path either
    """
    if model_stem.endswith(".pth") or model_stem.endswith(".pt"):
        model_path = model_dir / model_stem
        return model_path if model_path.exists() else None
    for ext in ["pth", "pt"]:
        if (model_dir / f"{model_stem}.{ext}").exists():
            return model_dir / f"{model_stem}.{ext}"
    return None


def list_known_models(model_path, default_model_info={}):
    """
    List the models available for similarity
    """
    models = {}
    if not model_path.exists():
        return models

    for file in model_path.iterdir():
        if file.is_file() and file.suffix in [".pth", ".pt"]:
            # look for metadata file
            if (metadata := file.with_suffix(".json")).exists():
                with open(metadata, "r") as f:
                    models[file.stem] = {"path": str(file), **orjson.loads(f.read())}
                    models[file.stem]["model"] = file.stem
            else:
                models[file.stem] = {
                    "path": str(file),
                    "date": time.ctime(os.path.getmtime(file)),
                    "model": file.stem,
                    "name": file.stem,
                    "desc": "No description available",
                }

    models = {
        **models,
        **default_model_info,
    }

    return models

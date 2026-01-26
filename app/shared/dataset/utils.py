from ..utils.logging import console
from .types import Image, DocInRange


def pdf_to_img(pdf_path, img_path, dpi=500, max_size=3000):
    """
    Convert the PDF file to PNG images
    """
    import subprocess

    # TODO use pymupdf_img

    file_prefix = pdf_path.stem
    try:
        # TODO issue, we save everything to png, even RGB images without alpha that could be compressed as jpg
        command = f"pdftoppm -png -r {dpi} -scale-to {max_size} {pdf_path} {img_path}/{file_prefix} -sep _ "
        subprocess.run(command, shell=True, check=True)

    except Exception as e:
        console(f"Error converting {pdf_path} to images: {e}", "red")


def group_by_documents(images: list[Image]) -> list[DocInRange]:
    """
    Identify groups of consecutive images from the same document
    """
    if not images:
        return []
    ranges = []
    p = 0
    for k, i in enumerate(images + [None]):
        if i is None or i.document != images[p].document:
            ranges.append(DocInRange(images[p].document, range(p, k), images[p:k]))
            p = k
    return ranges

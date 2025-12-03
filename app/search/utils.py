from skimage.morphology import skeletonize, dilation, footprint_rectangle
from skimage.util import invert
from skimage.filters import gaussian
from PIL import Image
import numpy as np
from pathlib import Path
import random
import string

BRIQUET_BG_PATH = Path(__file__).parent / "templates" / "watermark-blank.jpg"

def random_string(length=8):
    return ''.join(random.choices(string.ascii_lowercase + string.digits, k=length))

class BriquetSynthetizeSketch:
    def __init__(self, bg_path = BRIQUET_BG_PATH, size = (352, 352)):
        self.sketchbg = Image.open(bg_path).resize(size[::-1])

    def __call__(self, sketch: Image.Image):
        assert sketch.size == self.sketchbg.size
        white = Image.new("RGB", sketch.size, (255, 255, 255))
        sketch = invert(np.array(sketch.convert("L")))
        sketch = skeletonize(sketch)
        sketch = dilation(sketch, footprint_rectangle((3, 3)))
        sketch = 255.0*invert(sketch)
        sketch = gaussian(sketch, 1.5)
        white.paste(self.sketchbg, (0, 0), Image.fromarray(sketch).convert("L"))
        return white

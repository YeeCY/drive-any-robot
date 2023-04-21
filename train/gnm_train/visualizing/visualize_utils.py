import numpy as np
import torch
from PIL import Image

VIZ_IMAGE_SIZE = (640, 480)
RED = np.array([1, 0, 0])
GREEN = np.array([0, 1, 0])
BLUE = np.array([0, 0, 1])
CYAN = np.array([0, 1, 1])
YELLOW = np.array([1, 1, 0])
MAGENTA = np.array([1, 0, 1])


def numpy_to_img(arr: np.ndarray) -> Image:
    img = Image.fromarray(np.transpose(np.uint8(255 * arr), (1, 2, 0)))
    img = img.resize(VIZ_IMAGE_SIZE)
    return img


def to_numpy(tensor: torch.Tensor) -> np.ndarray:
    # use deep copy to prevent modification of tensor on cpu
    return tensor.detach().cpu().numpy().copy()


def from_numpy(array: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(array).float()


def ceil(a, precision=0):
    # https://stackoverflow.com/questions/58065055/floor-and-ceil-with-number-of-decimals
    return np.true_divide(np.ceil(a * 10 ** precision), 10 ** precision)


def floor(a, precision=0):
    # https://stackoverflow.com/questions/58065055/floor-and-ceil-with-number-of-decimals
    return np.true_divide(np.floor(a * 10 ** precision), 10 ** precision)

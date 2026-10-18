import cv2
import numpy as np


# Individual preprocessing steps

def to_grayscale(image: np.ndarray) -> np.ndarray: #converts color image to black and white
    if image.ndim == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image


def remove_noise(gray: np.ndarray, ksize: int = 3) -> np.ndarray: #removes noise from handwriten photo to preserve sharpness from handwriting sample

    return cv2.medianBlur(gray, ksize)


def binarize(gray: np.ndarray, invert: bool = True, method: str = "otsu") -> np.ndarray: #converting greyscale images to black/white

    flag = cv2.THRESH_BINARY_INV if invert else cv2.THRESH_BINARY
    if method == "adaptive":
        # a separate threshold for each small area, so shadows do not break it
        block = max(15, (min(gray.shape) // 4) | 1)   # odd window size
        C = 10 if invert else -10
        return cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, flag, block, C)
    _, binary = cv2.threshold(gray, 0, 255, flag + cv2.THRESH_OTSU)
    return binary


def deskew(binary: np.ndarray) -> np.ndarray: #straightens images that are rotated
    m = cv2.moments(binary)
    if abs(m['mu02']) < 1e-2:
        return binary  # no meaningful skew detected
    skew = m['mu11'] / m['mu02']
    h, w = binary.shape
    M = np.float32([[1, skew, -0.5 * w * skew], [0, 1, 0]])
    return cv2.warpAffine(binary, M, (w, h),
                           flags=cv2.WARP_INVERSE_MAP | cv2.INTER_LINEAR)


def resize(image: np.ndarray, size: tuple = (28, 28)) -> np.ndarray: #resizes image to 28x28
    return cv2.resize(image, size, interpolation=cv2.INTER_AREA)


def normalize(image: np.ndarray) -> np.ndarray: #changes float values from 0-255 to 0.0 - 1.0
    return image.astype(np.float32) / 255.0



# Full pipelines for both clean images (MNIST) and photos (Handwriting samples)


def preprocess_clean(image: np.ndarray, size: tuple = (28, 28)) -> np.ndarray:

    gray = to_grayscale(image)
    resized = resize(gray, size)
    return normalize(resized) # This is a 28x28 float32 array, values are scaled to 0.0 - 1.0 which can then be added put into the model


def preprocess_photo(image: np.ndarray, size: tuple = (28, 28),
    apply_deskew: bool = False, method: str = "otsu"):

    gray = to_grayscale(image)
    denoised = remove_noise(gray)
    invert = bool(np.median(denoised) > 127)   # dark ink on light paper: make the ink white
    binary = binarize(denoised, invert=invert, method=method)
    if apply_deskew:
        binary = deskew(binary)
    resized = resize(binary, size)
    return normalize(resized), binary # resised in the same as above, binary is the full sized image before croping that can be used for step two of assignment



from scipy.ndimage import median_filter, binary_opening, binary_closing
from skimage import color


def preprocess_image(img, filter_size=3, threshold=0.5):
    """Basic preprocessing pipeline for keyboard images."""
    gray = color.rgb2gray(img)
    filtered = median_filter(gray, size=filter_size)
    binary = filtered < threshold
    cleaned = binary_closing(binary_opening(binary))
    return cleaned

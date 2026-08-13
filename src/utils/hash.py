import numpy as np
import cv2


def average_hash(image_gray: np.ndarray, hash_size: int = 8) -> np.ndarray:
    resized = cv2.resize(image_gray, (hash_size, hash_size), interpolation=cv2.INTER_AREA)
    avg = resized.mean()
    return (resized > avg).astype(np.uint8)


def hamming_distance(hash_a: np.ndarray, hash_b: np.ndarray) -> int:
    return int(np.count_nonzero(hash_a != hash_b))

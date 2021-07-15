from time import time
import numpy as np
import cv2


def main():
    test_image = np.random.random((1920, 1080, 3)).astype(np.uint8)
    start = time()
    for _ in range(10):
        work = cv2.GaussianBlur(test_image, (5, 5), 3)
        work = cv2.Canny(test_image, 3, 3)
    print((time() - start) ** -1, 'Hz')


if __name__ == '__main__':
    main()

import cv2
import numpy as np

from core import QRDetector

if __name__ == '__main__':
    image = cv2.imread("./data/base.jpg", cv2.IMREAD_COLOR)
    # image = cv2.resize(image, (image.shape[0] // 2, image.shape[1] // 2), fx=0, fy=0, interpolation=cv2.INTER_AREA)
    detector = QRDetector()
    points = detector.localization(image)
    result = detector.compute_transformation_points(image).astype(np.int)
    print(result)
    image = cv2.drawContours(image, [result], -1, (0, 0, 255), 5)
    # image = cv2.resize(image, (512, 512), fx=0, fy=0, interpolation=cv2.INTER_CUBIC)
    # for point in points:
    #     cv2.circle(image, (point[0], point[1]), 1, (0, 0, 255), 100)
    image = cv2.resize(image, (512, 512), fx=0, fy=0, interpolation=cv2.INTER_AREA)
    cv2.imshow("Windows", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

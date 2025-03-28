import cv2
import numpy as np

def region_of_interest(img, vertices):
    mask = np.zeros_like(img)
    cv2.fillPoly(mask, vertices, 255)
    masked_img = cv2.bitwise_and(img, mask)
    return masked_img

def draw_lines(img, lines):
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 5)

def detect_lane_lines(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)

    height, width = image.shape[:2]
    roi_vertices = np.array([[(0, height), (width//2, height//2), (width, height)]], dtype=np.int32)
    cropped_edges = region_of_interest(edges, roi_vertices)

    lines = cv2.HoughLinesP(cropped_edges, 1, np.pi/180, 50, minLineLength=100, maxLineGap=50)
    lane_image = image.copy()
    draw_lines(lane_image, lines)

    return lane_image

# Load and process an image
image = cv2.imread("road.jpg")
lane_detected_image = detect_lane_lines(image)

cv2.imshow("Lane Detection", lane_detected_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
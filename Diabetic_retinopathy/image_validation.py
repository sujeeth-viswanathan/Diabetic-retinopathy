import cv2
import numpy as np

def has_circular_fundus_pattern(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (256, 256))
    center = gray[78:178, 78:178]
    return np.mean(center) > np.mean(gray) + 10

def has_texture_complexity(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 30, 100)
    edge_density = np.sum(edges > 0) / edges.size
    return edge_density > 0.05

def color_check(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_red = np.array([0, 30, 30])
    upper_red = np.array([25, 255, 255])
    mask_red = cv2.inRange(hsv, lower_red, upper_red)
    red_ratio = np.sum(mask_red) / mask_red.size
    return red_ratio > 0.02
    
def validate_retina_image(image):
    return (has_circular_fundus_pattern(image)
            and has_texture_complexity(image)
            and color_check(image)
)
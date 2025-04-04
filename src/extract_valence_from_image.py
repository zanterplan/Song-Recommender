import cv2
import numpy as np

def extract_valence_arousal_from_graph(img):
    gray = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2GRAY)
    
    dpi = 300
    cm_to_px = dpi / 2.54
    
    # Graph dimensions
    left_margin_cm = 0.65
    right_margin_cm = 0.12
    top_margin_cm = 0.22
    bottom_margin_cm = 0.42
    
    # Convert the margins to pixels
    left_margin_px = int(left_margin_cm * cm_to_px)
    right_margin_px = int(right_margin_cm * cm_to_px)
    top_margin_px = int(top_margin_cm * cm_to_px)
    bottom_margin_px = int(bottom_margin_cm * cm_to_px)
    
    # Crop image to graph area
    width, height = img.size
    cropped_img = img.crop((
        left_margin_px, 
        top_margin_px, 
        width - right_margin_px, 
        height - bottom_margin_px
    ))
    
    # Perform thresholding and contouring
    gray = cv2.cvtColor(np.array(cropped_img), cv2.COLOR_RGB2GRAY)
    _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Get the lower cross (1 - value, 2 - (0, 0))
    cross_contour = min(contours, key=cv2.contourArea)
    
    # Get moments
    moments = cv2.moments(cross_contour)
    cX = int(moments["m10"] / moments["m00"])
    cY = int(moments["m01"] / moments["m00"])
    
    # print(f"Cross center (x, y): ({cX}, {cY})")
    
    # Map to scale [0, 1]
    valence = map_to_valence(cX, cropped_img.width)
    arousal = map_to_arousal(cY, cropped_img.height)
    
    return valence, arousal

def map_to_valence(x, img_width):
    valence = (x / img_width)
    return valence


def map_to_arousal(y, img_height):
    arousal = 1 - (y / img_height)
    return arousal

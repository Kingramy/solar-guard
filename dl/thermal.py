# dl/thermal.py
import cv2
import numpy as np
from typing import List, Tuple, Optional

def segment_modules(temp_map: np.ndarray, threshold_c: float = 32.0) -> List[List[int]]:
    """
    Finds the 24 modules by thresholding the background gap (25.0C) from the modules (>=36.0C).
    Returns bounding boxes in [x, y, w, h] format, sorted row-major.
    """
    mask = (temp_map > threshold_c).astype(np.uint8) * 255
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    
    boxes = []
    # Skip label 0 (the background)
    for i in range(1, num_labels):
        x, y, w, h, area = stats[i]
        if area > 1000: # Filter out tiny noise artifacts
            boxes.append([int(x), int(y), int(w), int(h)])
            
    # Sort top-to-bottom (row), then left-to-right (col) using grid approximations
    boxes.sort(key=lambda b: (b[1] // 100, b[0] // 100))
    
    if len(boxes) != 24:
        raise ValueError(f"Geometry error: Expected 24 modules, found {len(boxes)}")
        
    return boxes

def compute_scene_baseline(temp_map: np.ndarray, boxes: List[List[int]]) -> float:
    """
    Calculates the median of the 24 individual module medians.
    This guarantees robustness against a few extremely hot faulty modules.
    """
    mod_medians = []
    for (x, y, w, h) in boxes:
        module_pixels = temp_map[y:y+h, x:x+w]
        mod_medians.append(float(np.median(module_pixels)))
        
    return round(float(np.median(mod_medians)), 2)

def find_hot_cell_bbox(temp_map: np.ndarray, bbox: List[int], margin_c: float = 12.0) -> Optional[List[int]]:
    """
    Heuristic to locate a localized hotspot inside a single module.
    """
    x, y, w, h = bbox
    module_temp = temp_map[y:y+h, x:x+w]
    
    local_median = np.median(module_temp)
    threshold = local_median + margin_c
    
    mask = (module_temp > threshold).astype(np.uint8)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=4)
    
    best_box = None
    max_area = 0
    
    for i in range(1, num_labels):
        hx, hy, hw, hh, area = stats[i]
        # Must be large enough to be a cell, but not the entire module
        if 16 <= area < (w * h * 0.5): 
            region_max = np.max(module_temp[hy:hy+hh, hx:hx+hw])
            if area > max_area and region_max >= 70.0:
                max_area = area
                # Convert back to full-scene coordinates
                best_box = [int(x + hx), int(y + hy), int(hw), int(hh)]
                
    return best_box
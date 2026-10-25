# dl/radiometric.py
import cv2
import numpy as np
from typing import Tuple

def raw16_to_celsius(raw_16bit: np.ndarray, scale: float = 0.01, offset_k: float = 273.15) -> np.ndarray:
    """
    Converts a 16-bit radiometric thermal array (standard FLIR/DJI centi-Kelvin format)
    into a float32 temperature matrix in degrees Celsius.
    Formula: T_C = (R * 0.01) - 273.15
    """
    if raw_16bit.dtype != np.uint16 and raw_16bit.dtype != np.int32:
        raise ValueError(f"Expected 16-bit radiometric array, got {raw_16bit.dtype}")
    temp_c = (raw_16bit.astype(np.float32) * scale) - offset_k
    return temp_c

def visual_to_proxy_celsius(
    image: np.ndarray, 
    ambient_c: float = 35.0, 
    max_expected_c: float = 95.0
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Fallback when a user uploads an 8-bit grayscale or RGB false-color thermal image
    without raw 16-bit telemetry.
    """
    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    if gray.dtype != np.uint8:
        g_min, g_max = float(np.min(gray)), float(np.max(gray))
        gray = ((gray - g_min) / (g_max - g_min + 1e-6) * 255.0).astype(np.uint8)

    norm = gray.astype(np.float32) / 255.0
    proxy_temp_c = ambient_c + norm * (max_expected_c - ambient_c)
    return gray, proxy_temp_c

def order_quad_corners(pts: np.ndarray) -> np.ndarray:
    """
    Orders 4 polygon vertices into strict order:
    [Top-Left, Top-Right, Bottom-Right, Bottom-Left]
    """
    rect = np.zeros((4, 2), dtype=np.float32)
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect

def rectify_panel_quad(
    img: np.ndarray,
    temp_map: np.ndarray,
    quad_pts: np.ndarray,
    out_w: int = 24,
    out_h: int = 40,
    inset_ratio: float = 0.96
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Applies a 3x3 Projective Homography Matrix (H) to warp a tilted/angled
    4-point solar panel in a drone frame into a flat orthogonal (out_h, out_w) crop.
    Uses inset_ratio (0.96) to sample strictly inside the panel interior, avoiding
    mounting frame edges and desert background bleed.
    """
    src_rect = order_quad_corners(np.asarray(quad_pts, dtype=np.float32))
    
    # Inset corners slightly toward the panel centroid to avoid frame/sand edge bleed
    if inset_ratio < 1.0:
        centroid = np.mean(src_rect, axis=0, keepdims=True)
        src_rect = centroid + inset_ratio * (src_rect - centroid)
    
    dst_rect = np.array([
        [0, 0],
        [out_w - 1, 0],
        [out_w - 1, out_h - 1],
        [0, out_h - 1]
    ], dtype=np.float32)
    
    H = cv2.getPerspectiveTransform(src_rect, dst_rect)
    warped_img = cv2.warpPerspective(img, H, (out_w, out_h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    warped_temp = cv2.warpPerspective(temp_map, H, (out_w, out_h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    
    return warped_img, warped_temp

def normalize_crop_for_cnn(crop: np.ndarray, temp_crop: np.ndarray) -> np.ndarray:
    """
    Variance-preserving domain calibration:
    Avoids blowing up low-variance healthy panels into fake anomalies.
    If the image crop is already in valid uint8 grayscale space, preserves its
    natural contrast while suppressing single-pixel interpolation spikes.
    """
    raw = crop.astype(np.float32)
    p2, p98 = np.percentile(raw, [2.0, 98.0])
    spread = p98 - p2

    # If the crop already sits in standard 8-bit dynamic range (like rectified uint8 frames),
    # return it directly without artificial min-max amplification.
    if 0.0 <= np.min(raw) and np.max(raw) <= 255.0 and spread < 180.0:
        return np.clip(raw, 0, 255).astype(np.uint8)

    # Otherwise, scale softly according to physical thermal spread
    temp_spread = float(np.percentile(temp_crop, 98.0) - np.percentile(temp_crop, 2.0))
    target_span = np.clip(temp_spread * 12.0, 20.0, 220.0)
    mid = np.median(raw)
    scaled = 128.0 + (raw - mid) * (target_span / (spread + 1e-6))
    return np.clip(scaled, 0, 255).astype(np.uint8)
"""
Single-Image Region Detection for SatQuery AI.

Detects and draws bounding rectangles around salient/distinct regions in a single
satellite image using classical computer vision:
  - Multi-scale edge detection (Canny)
  - Contour finding and hierarchical filtering
  - Saliency scoring based on area, compactness, edge density, and spectral contrast
  - Query-aware filtering to highlight the most relevant regions for the user's query
  - Annotated overlay with bright colour-coded bounding boxes

No trained neural model is required — fully deterministic and deployable offline.
"""

from typing import Any, Dict, List, Tuple
import cv2
import numpy as np
from PIL import Image


# ---------------------------------------------------------------------------
# Query-Keyword → Target-Colour mapping for context-aware box colours
# ---------------------------------------------------------------------------
QUERY_COLOUR_MAP = {
    "vegetation": (34, 197, 94),       # green
    "forest": (34, 197, 94),
    "crop": (34, 197, 94),
    "water": (56, 189, 248),           # sky blue
    "river": (56, 189, 248),
    "reservoir": (56, 189, 248),
    "road": (251, 191, 36),            # amber
    "highway": (251, 191, 36),
    "building": (251, 113, 133),       # rose
    "structure": (251, 113, 133),
    "urban": (251, 113, 133),
    "industrial": (249, 115, 22),      # orange
    "tank": (249, 115, 22),
    "storage": (249, 115, 22),
    "runway": (167, 139, 250),         # violet
    "airport": (167, 139, 250),
    "default": (0, 229, 255),          # cyan-teal fallback
}

def _pick_box_colour(query: str) -> Tuple[int, int, int]:
    """Returns the most contextually appropriate bounding-box RGB colour for the query."""
    q = query.lower()
    for kw, colour in QUERY_COLOUR_MAP.items():
        if kw in q:
            return colour
    return QUERY_COLOUR_MAP["default"]


def _to_rgb_numpy(image_input: Any) -> np.ndarray:
    """Normalise any image input to a uint8 RGB numpy array."""
    if isinstance(image_input, np.ndarray):
        if image_input.ndim == 2:
            return cv2.cvtColor(image_input, cv2.COLOR_GRAY2RGB)
        if image_input.shape[2] == 4:
            return cv2.cvtColor(image_input, cv2.COLOR_RGBA2RGB)
        return image_input.copy()
    if isinstance(image_input, Image.Image):
        pil_img = image_input
    elif hasattr(image_input, "read"):
        pil_img = Image.open(image_input)
    else:
        raise ValueError(f"Unsupported image type: {type(image_input)}")
    if pil_img.mode != "RGB":
        pil_img = pil_img.convert("RGB")
    return np.array(pil_img, dtype=np.uint8)


def _compute_region_saliency(
    contour: np.ndarray,
    gray: np.ndarray,
    edge_map: np.ndarray,
    h: int,
    w: int,
    min_area: int,
    max_area_fraction: float = 0.80,
) -> float:
    """
    Scores a contour on four independent criteria:
    1. Area normalised to [0, 1] vs. image size — rewards moderately large regions.
    2. Compactness (circularity) — rewards coherent, non-fragmented shapes.
    3. Edge density inside the bounding rect — rewards structurally rich regions.
    4. Mean spectral contrast inside the bounding rect vs. global image mean.

    Returns a composite saliency float in [0, 1].
    """
    area = cv2.contourArea(contour)
    if area < min_area:
        return 0.0

    total_pixels = h * w
    if area > max_area_fraction * total_pixels:
        return 0.0

    # 1. Area score — peaks at ~5% of image, falls off at extremes
    area_frac = area / total_pixels
    area_score = float(np.exp(-((area_frac - 0.05) ** 2) / (2 * 0.04 ** 2)))

    # 2. Compactness / shape regularity
    perimeter = cv2.arcLength(contour, True)
    compactness = (4 * np.pi * area / (perimeter ** 2 + 1e-6)) if perimeter > 0 else 0.0
    compactness = float(np.clip(compactness, 0.0, 1.0))

    # 3. Edge density inside bounding box
    x, y, bw, bh = cv2.boundingRect(contour)
    x, y = max(x, 0), max(y, 0)
    x2, y2 = min(x + bw, w), min(y + bh, h)
    roi_edges = edge_map[y:y2, x:x2]
    edge_density = float(np.mean(roi_edges) / 255.0) if roi_edges.size > 0 else 0.0

    # 4. Spectral contrast vs. global image mean
    roi_gray = gray[y:y2, x:x2]
    global_mean = float(np.mean(gray))
    local_mean = float(np.mean(roi_gray)) if roi_gray.size > 0 else global_mean
    contrast_score = float(np.clip(abs(local_mean - global_mean) / 128.0, 0.0, 1.0))

    composite = (
        0.25 * area_score
        + 0.20 * compactness
        + 0.30 * edge_density
        + 0.25 * contrast_score
    )
    return float(np.clip(composite, 0.0, 1.0))


def detect_regions(
    image_input: Any,
    query: str = "",
    min_area: int = 400,
    max_regions: int = 8,
    canny_low: int = 30,
    canny_high: int = 120,
) -> Dict[str, Any]:
    """
    Detects salient/distinct regions in a single satellite image and returns an
    annotated overlay with bounding rectangles drawn on each detected area.

    Parameters
    ----------
    image_input : PIL Image, numpy array, or file-like buffer
    query : str – user's natural-language query (used for contextual colour coding)
    min_area : int – minimum contour pixel area to consider (filters noise)
    max_regions : int – maximum number of boxes to draw (top-scored regions only)
    canny_low / canny_high : int – Canny edge-detector thresholds

    Returns
    -------
    dict with keys:
      - annotated_image  : np.ndarray (RGB), the original image with coloured
                           bounding boxes drawn on every detected region
      - bounding_boxes   : list of dicts {id, x, y, w, h, area_px, saliency_score}
      - num_regions      : int, number of boxes drawn
      - mean_saliency    : float, average saliency of retained regions
      - trace_details    : dict, parameters and pipeline metadata for trace panel
    """
    rgb = _to_rgb_numpy(image_input)
    h, w = rgb.shape[:2]

    # ── 1. Pre-process ──────────────────────────────────────────────────────
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    # Bilateral filter preserves edges while smoothing homogeneous areas
    blurred = cv2.bilateralFilter(gray, d=9, sigmaColor=75, sigmaSpace=75)

    # ── 2. Multi-scale Canny edge detection (merges two scale responses) ────
    edges_fine = cv2.Canny(blurred, canny_low, canny_high)
    edges_coarse = cv2.Canny(blurred, max(5, canny_low // 2), canny_high * 2)
    edges = cv2.bitwise_or(edges_fine, edges_coarse)

    # Dilate slightly to close small gaps between edge fragments
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    edges = cv2.dilate(edges, kernel, iterations=1)

    # ── 3. Find external contours ────────────────────────────────────────────
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # ── 4. Score and filter contours ────────────────────────────────────────
    scored: List[Tuple[float, np.ndarray]] = []
    for cnt in contours:
        score = _compute_region_saliency(cnt, gray, edges, h, w, min_area)
        if score > 0.0:
            scored.append((score, cnt))

    # Sort descending by saliency and keep top-N
    scored.sort(key=lambda x: x[0], reverse=True)
    selected = scored[:max_regions]

    # ── 5. Non-maximum suppression: remove heavily overlapping boxes ─────────
    def _iou(a: Tuple, b: Tuple) -> float:
        ax, ay, aw, ah = a
        bx, by, bw2, bh = b
        ix = max(0, min(ax + aw, bx + bw2) - max(ax, bx))
        iy = max(0, min(ay + ah, by + bh) - max(ay, by))
        inter = ix * iy
        union = aw * ah + bw2 * bh - inter
        return inter / (union + 1e-6)

    kept: List[Tuple[float, np.ndarray]] = []
    for s, cnt in selected:
        x, y, bw2, bh = cv2.boundingRect(cnt)
        overlaps = False
        for _, kcnt in kept:
            kx, ky, kw2, kbh = cv2.boundingRect(kcnt)
            if _iou((x, y, bw2, bh), (kx, ky, kw2, kbh)) > 0.40:
                overlaps = True
                break
        if not overlaps:
            kept.append((s, cnt))

    # ── 6. Draw bounding rectangles ──────────────────────────────────────────
    annotated = rgb.copy()
    box_colour = _pick_box_colour(query)
    bboxes: List[Dict] = []

    for idx, (sal, cnt) in enumerate(kept, start=1):
        x, y, bw2, bh = cv2.boundingRect(cnt)
        area_px = int(cv2.contourArea(cnt))

        # Thick outer rectangle
        cv2.rectangle(annotated, (x, y), (x + bw2, y + bh), box_colour, 2)

        # Corner accent marks for a clean "detection frame" style
        corner_len = max(8, min(bw2, bh) // 5)
        thickness = 3
        tl = (x, y)
        cv2.line(annotated, tl, (tl[0] + corner_len, tl[1]), box_colour, thickness)
        cv2.line(annotated, tl, (tl[0], tl[1] + corner_len), box_colour, thickness)
        tr = (x + bw2, y)
        cv2.line(annotated, tr, (tr[0] - corner_len, tr[1]), box_colour, thickness)
        cv2.line(annotated, tr, (tr[0], tr[1] + corner_len), box_colour, thickness)
        bl = (x, y + bh)
        cv2.line(annotated, bl, (bl[0] + corner_len, bl[1]), box_colour, thickness)
        cv2.line(annotated, bl, (bl[0], bl[1] - corner_len), box_colour, thickness)
        br = (x + bw2, y + bh)
        cv2.line(annotated, br, (br[0] - corner_len, br[1]), box_colour, thickness)
        cv2.line(annotated, br, (br[0], br[1] - corner_len), box_colour, thickness)

        # Label tag
        label = f"R{idx}  {sal:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        tag_y = max(y - 4, th + 4)
        cv2.rectangle(annotated, (x, tag_y - th - 3), (x + tw + 6, tag_y + 2), box_colour, -1)
        text_col = (10, 10, 10)   # dark text on coloured tag background
        cv2.putText(annotated, label, (x + 3, tag_y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, text_col, 1, cv2.LINE_AA)

        bboxes.append({
            "id": idx,
            "x": int(x), "y": int(y),
            "w": int(bw2), "h": int(bh),
            "area_px": area_px,
            "saliency_score": round(sal, 4),
        })

    mean_sal = round(float(np.mean([b["saliency_score"] for b in bboxes])), 4) if bboxes else 0.0

    return {
        "annotated_image": annotated,
        "bounding_boxes": bboxes,
        "num_regions": len(bboxes),
        "mean_saliency": mean_sal,
        "trace_details": {
            "algorithm": "Multi-scale Canny + Contour Saliency + NMS",
            "canny_thresholds": f"{canny_low} / {canny_high}",
            "min_contour_area_px": min_area,
            "max_regions_requested": max_regions,
            "total_contours_found": len(contours),
            "contours_above_min_area": len(scored),
            "regions_after_nms": len(bboxes),
            "mean_saliency_score": mean_sal,
            "box_colour_rgb": list(box_colour),
            "query_keyword_colour_routing": query.lower()[:60],
        },
    }

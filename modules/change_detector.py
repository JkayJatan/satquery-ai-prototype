"""
Bi-Temporal Change Detection Engine for SatQuery AI.
Uses classical computer vision (OpenCV & scikit-image) to compute:
- Grayscale alignment
- SSIM (Structural Similarity Index) difference map
- Otsu automatic thresholding
- Morphological filtering and Connected Components analysis
- Bounding boxes and translucent heat overlay
- Real change-percentage and anomaly statistics
"""

from typing import Dict, Any, Tuple, List
import numpy as np
from PIL import Image
import cv2
from skimage.metrics import structural_similarity as ssim


class BiTemporalChangeDetector:
    """
    Classical Computer Vision Change Detection pipeline.
    Deterministic, explainable, and fully self-contained without neural model dependencies.
    """

    def __init__(self, min_area: int = 50, ssim_win_size: int = 7):
        """
        :param min_area: Minimum connected component area in pixels to be classified as a valid change region.
        :param ssim_win_size: Window size for structural similarity calculation (must be odd).
        """
        self.min_area = min_area
        # win_size must be odd
        self.ssim_win_size = ssim_win_size if ssim_win_size % 2 == 1 else ssim_win_size + 1

    @staticmethod
    def _to_rgb_numpy(image_input: Any) -> np.ndarray:
        """Converts PIL Image or file-like buffer into standard RGB uint8 numpy array."""
        if isinstance(image_input, np.ndarray):
            if image_input.ndim == 2:
                return cv2.cvtColor(image_input, cv2.COLOR_GRAY2RGB)
            elif image_input.shape[2] == 4:
                return cv2.cvtColor(image_input, cv2.COLOR_RGBA2RGB)
            return image_input.copy()
        
        if hasattr(image_input, "read"):
            pil_img = Image.open(image_input)
        elif isinstance(image_input, Image.Image):
            pil_img = image_input
        else:
            raise ValueError(f"Unsupported image type: {type(image_input)}")
            
        if pil_img.mode != "RGB":
            pil_img = pil_img.convert("RGB")
        return np.array(pil_img, dtype=np.uint8)

    def detect_changes(
        self,
        img_before_input: Any,
        img_after_input: Any,
        min_area: int = None
    ) -> Dict[str, Any]:
        """
        Executes end-to-end bi-temporal change detection pipeline.

        Returns a dictionary containing:
        - overlay_image: RGB image with bounding boxes & translucent change mask overlaid on 'after' image
        - change_mask: Binary uint8 mask (255 = change, 0 = background)
        - diff_map_vis: Normalized SSIM difference heatmap (for visualization)
        - change_percentage: Float percentage of total image area changed
        - num_change_regions: Integer count of distinct change clusters >= min_area
        - bounding_boxes: List of dicts with [x, y, w, h, area]
        - ssim_score: Mean structural similarity index between images
        - otsu_threshold: Real threshold computed by Otsu's algorithm
        - trace_details: Detailed execution dictionary for the Orchestration Trace panel
        """
        if min_area is not None:
            effective_min_area = min_area
        else:
            effective_min_area = self.min_area

        # 1. Ingest and standardize to numpy RGB
        t0_rgb = self._to_rgb_numpy(img_before_input)
        t1_rgb = self._to_rgb_numpy(img_after_input)

        h0, w0 = t0_rgb.shape[:2]
        h1, w1 = t1_rgb.shape[:2]
        resized = False

        # Align dimensions if different
        if (h0, w0) != (h1, w1):
            t1_rgb = cv2.resize(t1_rgb, (w0, h0), interpolation=cv2.INTER_LINEAR)
            resized = True
            h, w = h0, w0
        else:
            h, w = h0, w0

        # 2. Convert to Grayscale
        gray_t0 = cv2.cvtColor(t0_rgb, cv2.COLOR_RGB2GRAY)
        gray_t1 = cv2.cvtColor(t1_rgb, cv2.COLOR_RGB2GRAY)

        # 3. Compute SSIM and full difference map
        # Win size must not exceed image dimensions
        win_size = min(self.ssim_win_size, min(h, w))
        if win_size % 2 == 0:
            win_size -= 1
        if win_size < 3:
            win_size = 3

        score, diff_map = ssim(
            gray_t0,
            gray_t1,
            full=True,
            win_size=win_size,
            data_range=255
        )

        # diff_map has values in [-1, 1], where 1 means identical and lower means changed
        # Scale to [0, 255] uint8: identical regions ~ 255, changed regions ~ 0
        diff_u8 = (diff_map * 255).clip(0, 255).astype(np.uint8)

        # 4. Apply Otsu's thresholding (Inverted so changed pixels = 255)
        otsu_thresh_val, raw_thresh = cv2.threshold(
            diff_u8,
            0,
            255,
            cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU
        )

        # 5. Morphological Cleanup (remove sensor noise, coalesce coherent regions)
        kernel_open = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        morph_opened = cv2.morphologyEx(raw_thresh, cv2.MORPH_OPEN, kernel_open, iterations=1)
        morph_cleaned = cv2.morphologyEx(morph_opened, cv2.MORPH_CLOSE, kernel_close, iterations=1)

        # 6. Connected Components Analysis
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            morph_cleaned,
            connectivity=8
        )

        filtered_mask = np.zeros_like(morph_cleaned, dtype=np.uint8)
        detected_boxes: List[Dict[str, int]] = []

        # Label 0 is background; evaluate labels 1 to num_labels - 1
        region_idx = 1
        for i in range(1, num_labels):
            area = int(stats[i, cv2.CC_STAT_AREA])
            if area >= effective_min_area:
                x = int(stats[i, cv2.CC_STAT_LEFT])
                y = int(stats[i, cv2.CC_STAT_TOP])
                bw = int(stats[i, cv2.CC_STAT_WIDTH])
                bh = int(stats[i, cv2.CC_STAT_HEIGHT])

                filtered_mask[labels == i] = 255
                detected_boxes.append({
                    "id": region_idx,
                    "x": x,
                    "y": y,
                    "w": bw,
                    "h": bh,
                    "area_px": area
                })
                region_idx += 1

        # 7. Render Overlays on 'After' Image
        overlay_img = t1_rgb.copy()
        
        # Color highlight: Crimson red overlay for detected change pixels
        color_mask = np.zeros_like(t1_rgb, dtype=np.uint8)
        color_mask[filtered_mask > 0] = [255, 45, 45]  # High-visibility red
        
        # Alpha blend overlay
        alpha = 0.45
        mask_indices = filtered_mask > 0
        overlay_img[mask_indices] = cv2.addWeighted(
            t1_rgb, 1.0 - alpha,
            color_mask, alpha,
            0
        )[mask_indices]

        # Draw crisp bounding boxes & region labels
        for box in detected_boxes:
            bx, by, bw, bh = box["x"], box["y"], box["w"], box["h"]
            # Bright cyan-teal bounding box (hex #00E5FF)
            cv2.rectangle(overlay_img, (bx, by), (bx + bw, by + bh), (0, 229, 255), 2)
            
            # Small label tag
            label_text = f"Δ{box['id']} ({box['area_px']}px)"
            (tw, th), baseline = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
            tag_y = max(by - 4, th + 2)
            cv2.rectangle(overlay_img, (bx, tag_y - th - 2), (bx + tw + 4, tag_y + 2), (0, 229, 255), -1)
            cv2.putText(overlay_img, label_text, (bx + 2, tag_y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (10, 15, 20), 1, cv2.LINE_AA)

        # 8. Compute Real Statistics
        total_scene_pixels = int(h * w)
        total_changed_pixels = int(np.count_nonzero(filtered_mask))
        change_pct = (total_changed_pixels / total_scene_pixels) * 100.0
        change_pct_rounded = round(change_pct, 3)

        # Difference map normalized visualization (inverted so changes are bright)
        diff_vis = cv2.applyColorMap(255 - diff_u8, cv2.COLORMAP_INFERNO)
        diff_vis = cv2.cvtColor(diff_vis, cv2.COLOR_BGR2RGB)

        return {
            "overlay_image": overlay_img,
            "change_mask": filtered_mask,
            "diff_map_vis": diff_vis,
            "change_percentage": change_pct_rounded,
            "num_change_regions": len(detected_boxes),
            "bounding_boxes": detected_boxes,
            "ssim_score": round(float(score), 4),
            "otsu_threshold": float(otsu_thresh_val),
            "total_pixels": total_scene_pixels,
            "changed_pixels": total_changed_pixels,
            "image_dimensions": f"{w}x{h}",
            "trace_details": {
                "algorithm": "SSIM + Otsu Threshold + Connected Components",
                "ssim_mean": round(float(score), 4),
                "ssim_window_size": win_size,
                "otsu_threshold_calculated": float(otsu_thresh_val),
                "min_area_filter_px": effective_min_area,
                "total_candidate_components": num_labels - 1,
                "filtered_valid_regions": len(detected_boxes),
                "total_pixels": total_scene_pixels,
                "changed_pixels": total_changed_pixels,
                "change_fraction": round(total_changed_pixels / total_scene_pixels, 6),
                "confidence_metric": f"{change_pct_rounded}% (real computed fraction of surface changed)",
                "dimension_alignment": "Resized T1 to match T0" if resized else "Exact match"
            }
        }

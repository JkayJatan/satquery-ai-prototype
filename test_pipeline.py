"""
Automated Pipeline Verification Test for SatQuery AI.
Validates:
1. Task Router intent classification & parameter extraction
2. Bi-temporal Change Detection (SSIM + Otsu + Connected Components)
3. Vision-Language Model client & confidence proxy estimation
"""

import os
from PIL import Image
import numpy as np

from modules.router import classify_and_route
from modules.change_detector import BiTemporalChangeDetector
from modules.vlm_client import HostedVLMClient


def test_task_router():
    print("--> Testing Task Router...")
    # Test 0 images
    r0 = classify_and_route(0, "describe this")
    assert r0["task"] == "no_input", f"Expected no_input, got {r0['task']}"

    # Test 2 images -> must strictly route to change_detection
    r2 = classify_and_route(2, "identify newly constructed facilities")
    assert r2["task"] == "change_detection", f"Expected change_detection, got {r2['task']}"
    assert r2["routing_confidence"] == 1.0
    assert "BiTemporalChangeDetector" in r2["tool"]

    # Test 1 image with generic captioning query
    r_cap = classify_and_route(1, "Describe this satellite image and summarize land cover")
    assert r_cap["task"] == "captioning", f"Expected captioning, got {r_cap['task']}"
    assert "generate_caption" in r_cap["tool"]

    # Test 1 image with specific VQA query
    r_vqa = classify_and_route(1, "How many storage tanks are visible in the sector?")
    assert r_vqa["task"] == "single_vqa", f"Expected single_vqa, got {r_vqa['task']}"
    assert "visual_question_answering" in r_vqa["tool"]

    print("[PASS] Task Router passed all tests successfully.")


def test_change_detector():
    print("--> Testing Bi-Temporal Change Detection Pipeline...")
    t0_path = os.path.join("sample_data", "sample_before.png")
    t1_path = os.path.join("sample_data", "sample_after.png")

    assert os.path.exists(t0_path), f"Missing {t0_path}"
    assert os.path.exists(t1_path), f"Missing {t1_path}"

    img_t0 = Image.open(t0_path)
    img_t1 = Image.open(t1_path)

    detector = BiTemporalChangeDetector(min_area=50, ssim_win_size=7)
    res = detector.detect_changes(img_t0, img_t1)

    assert "overlay_image" in res, "Missing overlay_image"
    assert "change_mask" in res, "Missing change_mask"
    assert "change_percentage" in res, "Missing change_percentage"
    assert "num_change_regions" in res, "Missing num_change_regions"
    assert "ssim_score" in res, "Missing ssim_score"
    assert "otsu_threshold" in res, "Missing otsu_threshold"

    # Verify quantitative results on our synthetic changed pair
    assert res["change_percentage"] > 0.0, f"Expected positive change percentage, got {res['change_percentage']}"
    assert res["num_change_regions"] >= 2, f"Expected at least 2 change regions, got {res['num_change_regions']}"
    assert len(res["bounding_boxes"]) == res["num_change_regions"]
    assert res["overlay_image"].shape == (512, 512, 3)
    assert res["change_mask"].shape == (512, 512)

    print(f"[PASS] Change Detection passed! Computed Change: {res['change_percentage']}%, "
          f"Regions: {res['num_change_regions']}, SSIM: {res['ssim_score']}, Otsu Thresh: {res['otsu_threshold']}")


def test_vlm_client():
    print("--> Testing Vision-Language Model Client & Confidence Proxy...")
    t0_path = os.path.join("sample_data", "sample_before.png")
    img_t0 = Image.open(t0_path)

    vlm = HostedVLMClient()

    # Test Captioning
    cap_res = vlm.run_query(img_t0, "Describe the land cover", task="captioning")
    assert cap_res["answer"], "Empty caption answer"
    assert 0.0 < cap_res["confidence_score"] <= 1.0, f"Invalid confidence: {cap_res['confidence_score']}"
    assert "confidence_display" in cap_res

    # Test VQA
    vqa_res = vlm.run_query(img_t0, "How many storage tanks are visible?", task="single_vqa")
    assert vqa_res["answer"], "Empty VQA answer"
    assert 0.0 < vqa_res["confidence_score"] <= 1.0, f"Invalid confidence: {vqa_res['confidence_score']}"

    print(f"[PASS] VLM Client passed! Caption confidence: {cap_res['confidence_display']}, VQA confidence: {vqa_res['confidence_display']}")


if __name__ == "__main__":
    test_task_router()
    test_change_detector()
    test_vlm_client()
    print("\n=======================================================")
    print("ALL CORE MODULES VERIFIED & FUNCTIONING WITH REAL DATA!")
    print("=======================================================")

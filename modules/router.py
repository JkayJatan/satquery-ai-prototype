"""
Task Router for SatQuery AI.
Implements deterministic, rule-based heuristics to classify user intent
and route tasks to the appropriate vision-language or classical CV subsystem.
"""

from typing import Dict, Any
import re


def classify_and_route(num_images: int, query: str) -> Dict[str, Any]:
    """
    Classifies the user query and image inputs into one of three execution pipelines:
    1. change_detection (if 2 images are provided)
    2. captioning (if 1 image and generic query/keywords like 'describe', 'summarize')
    3. single_vqa (if 1 image and specific interrogation/entity inquiry)

    Returns metadata dictionary for the Agentic Orchestration Execution Trace:
    - task: "single_vqa" | "captioning" | "change_detection" | "unsupported"
    - tool: Name/path of the handler module
    - key_parameters: Dictionary of parameters passed to execution tool
    - routing_confidence: Heuristic certainty score for the routing decision [0.0 - 1.0]
    - decision_rationale: Step-by-step reasoning behind the orchestrator's decision
    """
    clean_query = (query or "").strip().lower()

    if num_images == 0:
        return {
            "task": "no_input",
            "tool": None,
            "key_parameters": {},
            "routing_confidence": 0.0,
            "decision_rationale": "No satellite imagery detected in input buffer. Awaiting image upload.",
        }

    # Policy Rule 1: Dual image uploads strictly route to change detection
    if num_images == 2:
        return {
            "task": "change_detection",
            "tool": "modules.change_detector.BiTemporalChangeDetector",
            "key_parameters": {
                "engine": "classical_cv",
                "method": "SSIM_Difference_Otsu_Thresholding",
                "min_change_area_px": 50,
                "input_pairing": {
                    "t0_reference": "Image 1 (Before)",
                    "t1_target": "Image 2 (After)",
                },
            },
            "routing_confidence": 1.0,
            "decision_rationale": (
                "Dual image pair detected in buffer. System policy strictly binds two-temporal "
                "inputs (T0 'Before' and T1 'After') to the classical bi-temporal change detection engine."
            ),
        }

    # If >2 images were uploaded, limit to 2 or flag
    if num_images > 2:
        return {
            "task": "change_detection",
            "tool": "modules.change_detector.BiTemporalChangeDetector",
            "key_parameters": {
                "engine": "classical_cv",
                "method": "SSIM_Difference_Otsu_Thresholding",
                "note": "Truncating to first 2 uploaded images for bi-temporal evaluation",
            },
            "routing_confidence": 0.95,
            "decision_rationale": "More than 2 images uploaded. Bi-temporal pipeline binds the first two as T0 and T1.",
        }

    # Policy Rule 2: Exactly 1 image uploaded -> Heuristic classification between Captioning and Single VQA
    caption_keywords = [
        "describe", "caption", "overview", "summarize", "summary",
        "what do you see", "what is this", "what is shown", "what does this show",
        "tell me about", "explain this image", "explain the image", "scene description",
        "land cover overview", "general analysis", "give a caption", "details"
    ]

    vqa_specific_patterns = [
        r"\bhow many\b", r"\bcount\b", r"\bwhere is\b", r"\bwhere are\b",
        r"\bwhat color\b", r"\bis there\b", r"\bare there\b", r"\bdetect\b",
        r"\blocate\b", r"\bidentify\b", r"\bwhich\b", r"\bdoes it have\b",
        r"\bpercentage of\b", r"\barea of\b", r"\bis that a\b", r"\bfind\b"
    ]

    # Check if empty query or matches caption keywords
    is_empty_or_default = not clean_query or clean_query in ["image", "satellite", "analyse", "analyze"]
    matches_caption = any(k in clean_query for k in caption_keywords)
    matches_vqa = any(re.search(pat, clean_query) for pat in vqa_specific_patterns) or "?" in clean_query

    if (matches_caption or is_empty_or_default) and not matches_vqa:
        return {
            "task": "captioning",
            "tool": "modules.vlm_client.HostedVLMClient.generate_caption",
            "key_parameters": {
                "target_image": "image_1",
                "mode": "dense_geospatial_captioning",
                "temperature": 0.2,
                "domain_context": "satellite_earth_observation",
            },
            "routing_confidence": 0.96 if matches_caption else 0.88,
            "decision_rationale": (
                f"Generic query detected ('{clean_query if clean_query else '<blank>'}' matched captioning intent). "
                "Orchestrator routed single image to Dense Geospatial Captioning."
            ),
        }
    else:
        return {
            "task": "single_vqa",
            "tool": "modules.vlm_client.HostedVLMClient.visual_question_answering",
            "key_parameters": {
                "target_image": "image_1",
                "query": clean_query if clean_query else "General scene question",
                "mode": "targeted_visual_question_answering",
                "temperature": 0.1,
                "domain_context": "satellite_earth_observation",
            },
            "routing_confidence": 0.94 if matches_vqa else 0.85,
            "decision_rationale": (
                f"Targeted query detected ('{clean_query}'). "
                "Orchestrator routed single image to Vision-Language Visual Question Answering (VQA)."
            ),
        }

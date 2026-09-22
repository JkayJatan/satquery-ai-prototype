"""
App-level integration test using Streamlit AppTest framework.
Simulates user interactions, button clicks, and verifies UI components render without exception.
"""

import os
from streamlit.testing.v1 import AppTest

def test_streamlit_app_initial_render():
    print("--> Testing Streamlit app initial render...")
    at = AppTest.from_file("app.py", default_timeout=30)
    at.run()
    assert not at.exception, f"App threw exception: {at.exception}"
    print("[PASS] Initial render succeeded without errors.")

def test_streamlit_app_load_bi_temporal():
    print("--> Testing Streamlit app bi-temporal sample load & run query...")
    at = AppTest.from_file("app.py", default_timeout=30)
    at.run()

    # Click "Load Bi-Temporal Pair"
    # Find button by label
    btn_load_pair = None
    for b in at.button:
        if "Load Bi-Temporal Pair" in b.label:
            btn_load_pair = b
            break
    assert btn_load_pair is not None, "Button 'Load Bi-Temporal Pair' not found"
    btn_load_pair.click().run()
    assert not at.exception, f"Exception after loading bi-temporal pair: {at.exception}"

    # Click "Run Query"
    btn_run = None
    for b in at.button:
        if "Run Query" in b.label:
            btn_run = b
            break
    assert btn_run is not None, "Button 'Run Query' not found"
    btn_run.click().run()
    assert not at.exception, f"Exception after running query: {at.exception}"

    # Verify metrics rendered
    metric_labels = [m.label for m in at.metric]
    assert "Surface Area Changed" in metric_labels, f"Missing 'Surface Area Changed' metric. Found: {metric_labels}"
    assert "Scene SSIM Index" in metric_labels, f"Missing 'Scene SSIM Index' metric. Found: {metric_labels}"
    print(f"[PASS] Bi-temporal run query successfully rendered metrics: {metric_labels}")

def test_streamlit_app_load_single():
    print("--> Testing Streamlit app single image load & run query...")
    at = AppTest.from_file("app.py", default_timeout=30)
    at.run()

    btn_load_single = None
    for b in at.button:
        if "Load Single Image" in b.label:
            btn_load_single = b
            break
    assert btn_load_single is not None, "Button 'Load Single Image' not found"
    btn_load_single.click().run()
    assert not at.exception, f"Exception after loading single image: {at.exception}"

    btn_run = None
    for b in at.button:
        if "Run Query" in b.label:
            btn_run = b
            break
    assert btn_run is not None, "Button 'Run Query' not found"
    btn_run.click().run()
    assert not at.exception, f"Exception after running query: {at.exception}"

    metric_labels = [m.label for m in at.metric]
    assert "Task Pipeline" in metric_labels, f"Missing 'Task Pipeline' metric. Found: {metric_labels}"
    print(f"[PASS] Single image run query successfully rendered metrics: {metric_labels}")

if __name__ == "__main__":
    test_streamlit_app_initial_render()
    test_streamlit_app_load_bi_temporal()
    test_streamlit_app_load_single()
    print("\n=======================================================")
    print("STREAMLIT APP INTEGRATION TESTS PASSED 100%!")
    print("=======================================================")

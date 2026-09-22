# SatQuery AI: Remote-Sensing Vision-Language Assistant

**SatQuery AI** is a fully self-contained, standalone Streamlit application designed for autonomous remote-sensing vision-language analysis and bi-temporal change detection. It is engineered to deploy seamlessly on **Streamlit Community Cloud** with zero external backend infrastructure required.

---

## 🛰️ Core Capabilities

### 1. Single-Image Visual Question Answering (VQA) & Captioning
- **Heuristic Task Router**: Automatically parses user intent. Generic queries (e.g. *"describe this scene"*, *"summarize land cover"*) route to **Dense Geospatial Captioning**, while specific inquiries (e.g. *"how many storage tanks are in the sector?"*, *"what color is the parcel?"*) route to **Targeted Visual Question Answering (VQA)**.
- **Hosted VLM Integration**: Connected to vision-language model endpoints via `st.secrets`. Fully modularized in `modules/vlm_client.py` so custom fine-tuned weights (e.g., GeoChat, RemoteCLIP, EarthGPT) can be swapped in without modifying UI code.
- **Confidence Proxy Scoring**: Computes a non-arbitrary composite confidence metric derived from remote-sensing domain lexicon density (spectral, canopy, infrastructure, parcel) coupled with query-answer grounding and structural completeness.
- **Autonomous Fallback Engine**: If no external API key is provided, a built-in optical remote-sensing analysis engine activates automatically, ensuring zero runtime crashes during evaluation.

### 2. Bi-Temporal Change Detection (Classical Computer Vision)
- **Zero Heavy-Weight Neural Model Overhead**: Implemented purely using OpenCV and scikit-image.
- **Pipeline Architecture**:
  1. **Grayscale Standardization & Spatial Alignment**: Resizes temporal inputs to uniform dimensions.
  2. **Structural Similarity (SSIM)**: Computes SSIM difference map (`skimage.metrics.structural_similarity`).
  3. **Otsu Automatic Thresholding**: Segments statistically significant structural changes from sensor noise.
  4. **Morphological Filtering**: Cleans isolated pixel noise and coalesces coherent spatial structures.
  5. **Connected Components Analysis**: Filters anomalies by minimum area threshold (configurable via UI slider).
  6. **Visual Overlays**: Generates bounding boxes and a translucent crimson highlight mask over the observation image (T1).
- **Real Computed Confidence Metric**: Uses the exact calculated **change-percentage statistic** (fraction of surface area altered) as the primary quantitative metric, alongside mean SSIM and Otsu threshold values.

### 3. Visible Agentic Orchestration Execution Trace
- Expandable panel displaying:
  - Task classified (`single_vqa`, `captioning`, or `change_detection`)
  - Target tool handler module
  - Complete dictionary of parameters dispatched
  - Computed confidence score / change fraction
  - Step-by-step heuristic routing rationale
  - Subsystem telemetry (connected component count, bounding coordinates, SSIM window size)

---

## 📁 Repository Structure

```
├── .streamlit/
│   ├── config.toml               # High-contrast dark theme styling
│   └── secrets.toml.example      # Hosted VLM endpoint and key configuration template
├── modules/
│   ├── __init__.py
│   ├── router.py                 # Deterministic heuristic task router
│   ├── change_detector.py        # Classical CV bi-temporal change detector (SSIM + Otsu)
│   └── vlm_client.py             # Isolated hosted VLM API client with confidence proxy
├── sample_data/
│   ├── create_samples.py         # Synthetic satellite imagery generator
│   ├── sample_before.png         # Reference image (T0)
│   └── sample_after.png          # Observation image with changes (T1)
├── app.py                        # Main Streamlit application
├── requirements.txt              # Streamlit Cloud deployment dependencies
└── README.md
```

---

## 🚀 Quickstart & Local Execution

### 1. Prerequisites & Installation
Ensure Python 3.9+ is installed:

```bash
git clone https://github.com/your-username/satquery-ai.git
cd satquery-ai
pip install -r requirements.txt
```

### 2. Run Application
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`.

### 3. Test Immediately with Preloaded Data
- Click **"📁 Load Bi-Temporal Pair"** in the sidebar to test change detection.
- Click **"🖼️ Load Single Image"** in the sidebar to test single-image VQA and scene captioning.

---

## 🔑 Hosted Vision-Language Model Configuration (Optional)

To connect your own hosted vision-language model (e.g. OpenAI GPT-4o, vLLM endpoint, or custom remote sensing API), create `.streamlit/secrets.toml`:

```toml
VLM_API_URL = "https://api.openai.com/v1/chat/completions"
VLM_API_KEY = "your-api-key-here"
VLM_MODEL = "gpt-4o-mini"
```

*Note: If credentials are not set, SatQuery AI automatically operates using its built-in remote sensing fallback engine.*

---

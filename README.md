# Feature Detector Sandbox

![screenshot](sandbox_screenshot.png?raw=true)

Made with Claude Opus 4.5 to experiment with feature matching for another project I am working on. As a tool its very useful so id like to share it with others who might benefit. 

A comprehensive web-based testbed for experimenting with stereo feature detection and matching algorithms. Compare traditional OpenCV detectors against state-of-the-art deep learning matchers like SuperPoint+LightGlue and EfficientLoFTR.

![Python](https://img.shields.io/badge/Python-3.11+-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.109+-green)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red)

## Features

### Traditional Detectors
- **ORB** - Oriented FAST and Rotated BRIEF (fast, binary descriptors)
- **AKAZE** - Accelerated-KAZE (robust to scale/rotation)
- **BRISK** - Binary Robust Invariant Scalable Keypoints
- **SIFT** - Scale-Invariant Feature Transform (classic, accurate)
- **SURF** - Speeded-Up Robust Features (requires opencv-contrib)

### Deep Learning Matchers
- **SuperPoint + LightGlue** - Fast learned feature detector + matcher
- **DISK + LightGlue** - Discrete keypoint detector + LightGlue matcher
- **EfficientLoFTR** - Semi-dense detector-free matcher (~2.5x faster than LoFTR)

### Match Filtering
- **Lowe's Ratio Test** - Filter ambiguous matches
- **Rank Filtering** - Keep top N matches by confidence
- **Spatial Distance Filter** - Remove matches beyond a pixel threshold

### Robust Estimation (Outlier Rejection)
- **RANSAC** - Random Sample Consensus
- **LMEDS** - Least Median of Squares
- **RHO** - PROSAC-based method
- **MAGSAC++** - Marginalizing Sample Consensus (when available)

### Visualization Modes
- **Side-by-Side** - Traditional match visualization
- **Anaglyph** - Red/Cyan stereo overlay (great for 3D glasses!)

### Performance Monitoring
- Timing breakdown for each stage (detection, matching, refinement, visualization)
- GPU memory usage for deep learning models

## Installation

### Prerequisites
- Python 3.11 or higher
- (Optional) NVIDIA GPU with CUDA for accelerated deep learning

### Step 1: Clone and Create Virtual Environment

```bash
git clone <repo-url>
cd "Test Feature Detectors"
python -m venv venv
```

### Step 2: Activate Virtual Environment

**Windows (PowerShell):**
```powershell
.\venv\Scripts\Activate.ps1
```

**Windows (CMD):**
```cmd
.\venv\Scripts\activate.bat
```

**Linux/macOS:**
```bash
source venv/bin/activate
```

### Step 3: Install Core Dependencies

```bash
pip install -r requirements.txt
```

This installs the core dependencies for traditional detectors (ORB, AKAZE, BRISK, SIFT).

### Step 4: (Optional) Install SURF Support

SURF requires opencv-contrib-python:

```bash
pip install opencv-contrib-python
```

### Step 5: (Optional) Install Deep Learning Matchers

For SuperPoint+LightGlue, DISK+LightGlue, and EfficientLoFTR:

#### 5a. Install PyTorch with CUDA

First, install PyTorch matching your CUDA version. Visit https://pytorch.org/get-started/locally/ for the exact command.

**CUDA 12.4:**
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
```

**CUDA 12.1:**
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

**CUDA 11.8:**
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

**CPU only (no GPU):**
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

#### 5b. Install Transformers and Kornia

```bash
pip install transformers kornia pillow
```

## Usage

### Start the Server

**Windows:**
```powershell
.\venv\Scripts\python.exe run.py
```

**With venv activated:**
```bash
python run.py
```

**Alternative (direct uvicorn):**
```bash
uvicorn backend.main:app --host 127.0.0.1 --port 8000
```

### Access the Web Interface

Open your browser and navigate to: **http://127.0.0.1:8000**

### Using the Interface

1. **Upload Images** - Select left and right stereo images
2. **Choose Mode** - Toggle between Traditional (OpenCV) and Deep Learning matchers
3. **Select Detector/Matcher** - Pick from available algorithms
4. **Adjust Filters** - Configure ratio test, rank limits, and distance thresholds
5. **Compute Matches** - Click the button to run matching
6. **Refine** - Apply RANSAC/MAGSAC++ to filter outliers
7. **Switch Visualization** - Toggle between side-by-side and anaglyph views

## Project Structure

```
Test Feature Detectors/
├── backend/
│   ├── __init__.py
│   ├── main.py                  # FastAPI application & endpoints
│   ├── utils.py                 # Image encoding/decoding, visualization
│   ├── detectors/
│   │   ├── __init__.py          # Detector registry
│   │   ├── base.py              # BaseDetector abstract class
│   │   └── opencv_detectors.py  # ORB, AKAZE, BRISK, SIFT, SURF
│   ├── matchers/
│   │   ├── __init__.py          # Matcher registry
│   │   ├── base.py              # BaseMatcher abstract class
│   │   ├── opencv_matchers.py   # BFMatcher with filtering
│   │   └── robust_estimation.py # RANSAC, LMEDS, MAGSAC++
│   └── dense_matchers/
│       ├── __init__.py          # Dense matcher registry
│       ├── base.py              # BaseDenseMatcher abstract class
│       ├── lightglue_matcher.py # SuperPoint+LightGlue, DISK+LightGlue
│       └── efficientloftr_matcher.py  # EfficientLoFTR
├── frontend/
│   └── index.html               # Web UI (Tailwind CSS)
├── venv/                        # Python virtual environment
├── requirements.txt             # Python dependencies
├── run.py                       # Server startup script
├── CLAUDE.md                    # Development notes
└── README.md                    # This file
```

## API Reference

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Serve frontend HTML |
| `/api/detectors` | GET | List available traditional detectors |
| `/api/matchers` | GET | List available matchers |
| `/api/dense-matchers` | GET | List available deep learning matchers |
| `/api/robust-methods` | GET | List robust estimation methods |
| `/api/match` | POST | Compute matches (traditional pipeline) |
| `/api/dense-match` | POST | Compute matches (deep learning) |
| `/api/refine` | POST | Refine matches with robust estimation |
| `/api/health` | GET | Health check |

## Troubleshooting

### "No dense matchers available"
Install PyTorch and transformers. See Step 5 above.

### SURF not available
Install opencv-contrib-python:
```bash
pip install opencv-contrib-python
```

### CUDA out of memory
- Try smaller images
- Use EfficientLoFTR (more memory efficient)
- Reduce confidence threshold to get fewer matches

### Models downloading slowly
The first time you use a deep learning matcher, it downloads the model from HuggingFace. This is a one-time download.

### Server won't stop with Ctrl+C (Windows)
Use Task Manager or:
```powershell
taskkill /IM python.exe /F
```

## Adding New Detectors

1. Create a new file in `backend/detectors/`
2. Inherit from `BaseDetector`
3. Implement `detect_and_compute()` method
4. Register in `backend/detectors/__init__.py`

```python
from .base import BaseDetector, DetectionResult, KeyPoint

class MyDetector(BaseDetector):
    name = "MyDetector"
    description = "My custom detector"

    def detect_and_compute(self, image, mask=None):
        # Your detection logic here
        keypoints = [...]
        descriptors = np.array(...)
        return DetectionResult(keypoints=keypoints, descriptors=descriptors)

    def get_norm_type(self):
        return "L2"  # or "HAMMING" for binary descriptors
```

## Adding New Dense Matchers

1. Create a new file in `backend/dense_matchers/`
2. Inherit from `BaseDenseMatcher`
3. Implement `match()` method
4. Register in `backend/dense_matchers/__init__.py`

```python
from .base import BaseDenseMatcher, DenseMatchResult

class MyDenseMatcher(BaseDenseMatcher):
    name = "MyMatcher"
    description = "My custom dense matcher"

    def match(self, image1, image2, threshold=0.0):
        # Your matching logic here
        return DenseMatchResult(
            keypoints1=kp1_array,  # Nx2
            keypoints2=kp2_array,  # Nx2
            scores=scores_array    # N
        )
```

## License

MIT

## Acknowledgments

- [OpenCV](https://opencv.org/) for traditional feature detectors
- [HuggingFace Transformers](https://huggingface.co/transformers/) for deep learning models
- [LightGlue](https://github.com/cvg/LightGlue) by CVG, ETH Zurich
- [EfficientLoFTR](https://github.com/zju3dv/efficientloftr) by ZJU 3DV Lab

# YOLOv8 Object Detection
Real-time object detection using YOLOv8 for images and webcam streams. This implementation provides a simple yet powerful interface for detecting objects in various scenarios.

## 🚀 Features

- 🖼️ **Image & Video Processing**: Detect objects in images and video files
- 📹 **Real-time Webcam**: Live object detection using your webcam
- 🛠️ **Multiple Models**: Support for YOLOv8 models (nano to xlarge)
- ⚙️ **Customizable**: Adjust confidence and IOU thresholds
- 🖥️ **Cross-Platform**: Works on Windows, macOS, and Linux
- 🚀 **Optimized Performance**: Supports both CPU and GPU acceleration

## 📋 Prerequisites

- Python 3.7 or higher
- pip (Python package installer)
- Webcam (for real-time detection)
- CUDA-compatible GPU (recommended for better performance)

## 🛠️ Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/Nareshrana1999/object-detection-yolov8n.git
   cd object-detection-yolov8n
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

   > 💡 The YOLOv8n model will be automatically downloaded on first run (~12MB).

## 🚀 Quick Start

1. **Run the detection script**:
   ```bash
   python object.py
   ```

2. **Select an option**:
   - Press `1` to detect objects in an image
   - Press `2` for real-time webcam detection
   - Press `q` to quit

## 🏗️ Model Architecture

The project utilizes YOLOv8, the latest version of the YOLO (You Only Look Once) object detection model. Key features include:

- **Backbone**: CSPDarknet53
- **Neck**: PANet
- **Head**: YOLOv8 Head with anchor-free detection
- **Activation**: SiLU activation function

## ⚙️ Usage Options

```bash
python object.py [--source SOURCE] [--model MODEL] [--conf CONF] [--iou IOU]
```

### Arguments:
- `--source`: Input source (0 for webcam, path to image/video)
- `--model`: YOLOv8 model to use (default: 'yolov8n.pt')
- `--conf`: Confidence threshold (0-1, default: 0.5)
- `--iou`: IOU threshold for NMS (0-1, default: 0.4)

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.


## Usage

1. Run the detection script:
## 🛠️ Usage

### 🖥️ Interactive Mode (Recommended)

Simply run the script without arguments to use the interactive menu:

```bash
python object.py
```

You'll see a menu like this:
```
==================================================
             YOLOv8 Object Detection              
==================================================

Select detection mode:
1. Image Detection (process an image file)
2. Webcam Detection (use your camera)
0. Exit

Enter your choice (0-2):
```

### 🖼️ Command Line Mode

Alternatively, you can still use command line arguments for automation:

#### Image Detection
```bash
# Basic usage
python yolov8.py --image path/to/your/image.jpg

# With custom output
python yolov8.py --image input.jpg --output result.jpg


# Face Detection & Anonymization CLI Tool

A command-line tool that detects human faces in images and videos using OpenCV's Haar Cascade classifier, and anonymizes them by either drawing bounding boxes or applying Gaussian blur. Built for the course - CSE3010 (Computer Vision)

---

## Features

- Detects faces in images (`.jpg`, `.jpeg`, `.png`)
- Detects faces in videos (`.mp4`, `.avi`)
- Two anonymization modes: **box** (draw rectangle) and **blur** (Gaussian blur)
- Fully runnable from the command line — no GUI required
- Lightweight — only depends on `opencv-python`

---

## Project Structure

```
face-anonymizer/
├── main.py             # CLI entry point
├── detector.py         # Face detection logic using Haar Cascade
├── requirements.txt    # Python dependencies
├── README.md           # This file
├── sample_input/       # Place your test images/videos here
└── sample_output/      # Results are saved here
```

---

## Requirements

- Python 3.7 or higher
- opencv-python

---

## Installation & Setup

**1. Clone the repository**

```bash
git clone https://github.com/{your-username}/face-anonymizer.git
cd face-anonymizer
```

**2. (Optional but recommended) Create a virtual environment**

```bash
python -m venv venv

# On Windows:
venv\Scripts\activate

# On Mac/Linux:
source venv/bin/activate
```

**3. Install dependencies**

```bash
pip install -r requirements.txt
```

---

## How to Run

Place your input image or video inside the `sample_input/` folder, then run one of the following commands:

**Blur faces in an image:**
```bash
python main.py --input sample_input/test_image.jpg --mode blur --output sample_output/result.jpg
```

**Draw bounding boxes on faces in an image:**
```bash
python main.py --input sample_input/test_image.jpg --mode box --output sample_output/result.jpg
```

**Blur faces in a video:**
```bash
python main.py --input sample_input/test_video.mp4 --mode blur --output sample_output/result.mp4
```

**Arguments:**

| Argument   | Required | Description |
|------------|----------|-------------|
| `--input`  | Yes      | Path to input image or video file |
| `--mode`   | Yes      | `box` to draw rectangle, `blur` to apply Gaussian blur |
| `--output` | Yes      | Path to save the output file |

---

## How It Works

1. **Input loading** — The script reads the input using `cv2.imread` (image) or `cv2.VideoCapture` (video)
2. **Preprocessing** — Each frame is converted to grayscale and histogram-equalized for better detection
3. **Face detection** — OpenCV's Haar Cascade classifier (`haarcascade_frontalface_default.xml`) detects face regions
4. **Anonymization** — Detected face regions are either boxed with a green rectangle or blurred using a 51×51 Gaussian kernel
5. **Output saving** — The processed image/video is saved to the specified output path

---

## Limitations

- Detection accuracy is lower on angled or partially visible faces
- Performance degrades in very poor lighting or low-resolution images
- Haar Cascade is trained primarily on frontal faces; side profiles may be missed
- Very small faces (under 30×30 pixels) are not detected

---

## Future Work

- Replace Haar Cascade with a DNN-based detector (e.g., MTCNN or RetinaFace) for better accuracy
- Add real-time webcam support using `cv2.VideoCapture(0)`
- Add a `--show` flag to preview results in a window
- Support batch processing of multiple files

---

## References

- Viola, P., & Jones, M. (2001). *Rapid object detection using a boosted cascade of simple features.* CVPR 2001.
- OpenCV Documentation: https://docs.opencv.org

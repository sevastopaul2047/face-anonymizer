"""
main.py
-------
CLI tool for face detection and anonymization.
Supports images (.jpg, .jpeg, .png) and videos (.mp4, .avi).

Usage:
    python main.py --input <path> --mode <box|blur> --output <path>

Examples:
    python main.py --input sample_input/test_image.jpg --mode blur --output sample_output/result.jpg
    python main.py --input sample_input/test_image.jpg --mode box  --output sample_output/result.jpg
    python main.py --input sample_input/test_video.mp4 --mode blur --output sample_output/result.mp4
"""

import argparse
import os
import sys
import cv2
from detector import detect_faces


# ──────────────────────────────────────────────────────────────
# Helper: apply anonymization to a single frame
# ──────────────────────────────────────────────────────────────

def anonymize_frame(frame, mode):
    """
    Detect faces in a frame and apply box or blur anonymization.

    Parameters:
        frame (numpy.ndarray): A single BGR image / video frame
        mode  (str)          : 'box' draws a green rectangle,
                               'blur' applies Gaussian blur

    Returns:
        Processed frame with faces anonymized.
        Count of faces detected.
    """
    faces = detect_faces(frame)

    for (x, y, w, h) in faces:
        if mode == "box":
            # Draw a thick green rectangle around the face
            # Last number is thickness — 3 is much more visible than 2
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)

            # Also fill a semi-visible solid green overlay on top of the face
            overlay = frame.copy()
            cv2.rectangle(overlay, (x, y), (x + w, y + h), (0, 255, 0), -1)  # -1 = filled
            cv2.addWeighted(overlay, 0.4, frame, 0.6, 0, frame)  # 40% green overlay

        elif mode == "blur":
            # Expand the face region slightly so edges are fully covered
            pad = 10
            x1 = max(x - pad, 0)
            y1 = max(y - pad, 0)
            x2 = min(x + w + pad, frame.shape[1])
            y2 = min(y + h + pad, frame.shape[0])

            face_region = frame[y1:y2, x1:x2]

            # Apply pixelation effect — shrink then enlarge = heavy blur
            small = cv2.resize(face_region, (10, 10), interpolation=cv2.INTER_LINEAR)
            pixelated = cv2.resize(small, (x2 - x1, y2 - y1), interpolation=cv2.INTER_NEAREST)

            # Also apply Gaussian blur on top of pixelation for extra strength
            blurred = cv2.GaussianBlur(pixelated, (51, 51), 30)

            frame[y1:y2, x1:x2] = blurred

    return frame, len(faces)


# ──────────────────────────────────────────────────────────────
# Image processing
# ──────────────────────────────────────────────────────────────

def process_image(input_path, mode, output_path):
    """Load an image, anonymize faces, and save the result."""

    print(f"[INFO] Loading image: {input_path}")
    image = cv2.imread(input_path)

    if image is None:
        print(f"[ERROR] Could not load image at '{input_path}'")
        print("        Make sure the file exists and is a valid image.")
        sys.exit(1)

    print("[INFO] Running face detection...")
    result, face_count = anonymize_frame(image, mode)

    # Make sure the output folder exists
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)

    # Save the result
    cv2.imwrite(output_path, result)

    print(f"[INFO] Faces detected : {face_count}")
    print(f"[INFO] Result saved to: {output_path}")


# ──────────────────────────────────────────────────────────────
# Video processing
# ──────────────────────────────────────────────────────────────

def process_video(input_path, mode, output_path):
    """Read a video frame by frame, anonymize faces, and save."""

    print(f"[INFO] Loading video: {input_path}")
    cap = cv2.VideoCapture(input_path)

    if not cap.isOpened():
        print(f"[ERROR] Could not open video at '{input_path}'")
        sys.exit(1)

    # Get video properties for the writer
    fps    = cap.get(cv2.CAP_PROP_FPS) or 25
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"[INFO] Video info — {width}x{height} @ {fps:.1f} fps, {total} frames")

    # Make sure the output folder exists
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)

    # Set up the video writer — uses mp4v codec for .mp4 output
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out    = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    frame_num   = 0
    total_faces = 0

    print("[INFO] Processing frames...")

    while True:
        ret, frame = cap.read()
        if not ret:
            break  # No more frames

        processed, faces_in_frame = anonymize_frame(frame, mode)
        total_faces += faces_in_frame
        out.write(processed)

        frame_num += 1
        # Print progress every 30 frames
        if frame_num % 30 == 0:
            print(f"        Processed {frame_num}/{total} frames...")

    cap.release()
    out.release()

    print(f"[INFO] Total frames processed : {frame_num}")
    print(f"[INFO] Total face detections  : {total_faces}")
    print(f"[INFO] Result saved to        : {output_path}")


# ──────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Face Detection & Anonymization CLI Tool",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python main.py --input sample_input/test_image.jpg --mode blur --output sample_output/result.jpg\n"
            "  python main.py --input sample_input/test_image.jpg --mode box  --output sample_output/result.jpg\n"
            "  python main.py --input sample_input/test_video.mp4 --mode blur --output sample_output/result.mp4\n"
        )
    )
    parser.add_argument(
        "--input",  required=True,
        help="Path to input image (.jpg/.jpeg/.png) or video (.mp4/.avi)"
    )
    parser.add_argument(
        "--mode",   required=True, choices=["box", "blur"],
        help="'box'  → draw green rectangle around each face\n'blur' → apply Gaussian blur to each face"
    )
    parser.add_argument(
        "--output", required=True,
        help="Path to save the output file (e.g. sample_output/result.jpg)"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Validate input file existence
    if not os.path.exists(args.input):
        print(f"[ERROR] Input file not found: '{args.input}'")
        sys.exit(1)

    # Determine if input is image or video based on extension
    image_exts = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}
    video_exts = {".mp4", ".avi", ".mov", ".mkv"}
    ext = os.path.splitext(args.input)[1].lower()

    if ext in image_exts:
        process_image(args.input, args.mode, args.output)
    elif ext in video_exts:
        process_video(args.input, args.mode, args.output)
    else:
        print(f"[ERROR] Unsupported file type: '{ext}'")
        print(f"        Supported image types : {image_exts}")
        print(f"        Supported video types : {video_exts}")
        sys.exit(1)

    print("[DONE] All finished!")


if __name__ == "__main__":
    main()

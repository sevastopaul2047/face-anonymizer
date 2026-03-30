"""
detector.py
-----------
Handles face detection using OpenCV's Haar Cascade classifier.
"""

import cv2
import os


def get_cascade_path():
    """
    Locate the Haar Cascade XML file bundled with OpenCV.
    Works across different OS and OpenCV installation paths.
    """
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    if not os.path.exists(cascade_path):
        raise FileNotFoundError(
            f"Haar Cascade file not found at: {cascade_path}\n"
            "Make sure opencv-python is installed correctly."
        )
    return cascade_path


def detect_faces(image):
    """
    Detect faces in a BGR image using Haar Cascade.

    Parameters:
        image (numpy.ndarray): Input image in BGR format (as loaded by cv2.imread)

    Returns:
        List of tuples [(x, y, w, h), ...] where:
            x, y = top-left corner of the face rectangle
            w, h = width and height of the face rectangle
    """
    # Convert the image to grayscale — Haar Cascade works on grayscale images
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Improve contrast so detection works better in varied lighting
    gray = cv2.equalizeHist(gray)

    # Load the Haar Cascade face detector
    cascade_path = get_cascade_path()
    face_cascade = cv2.CascadeClassifier(cascade_path)

    # Detect faces in the image
    # scaleFactor=1.1  -> how much the image is scaled down each step
    # minNeighbors=5   -> how many neighbors each candidate rectangle should have
    # minSize=(30,30)  -> minimum face size to detect
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE
    )

    # If no faces found, return empty list
    if len(faces) == 0:
        return []

    # Convert to list of tuples for easy use
    return [(x, y, w, h) for (x, y, w, h) in faces]

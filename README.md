Road Lane Lines Detection

Overview

This project implements a computer vision-based approach to detecting road lane lines from video or image input. The system leverages image processing techniques such as edge detection, color thresholding, and Hough Transform to identify lane markings.

Features

Detects lane lines in real-time video streams or static images

Utilizes OpenCV for image processing

Implements Canny Edge Detection and Hough Transform

Supports region of interest selection to focus on relevant lane areas

Works with different lighting and road conditions

Installation

Prerequisites

Ensure you have the following dependencies installed:

Python  3.13.2

OpenCV

NumPy

Matplotlib (for visualization)

Steps to Install

Clone the repository:

git clone https://github.com/sheesubhajit/Road-Lane-Lines-Detection.git

Navigate to the project directory:

cd Road-Lane-Lines-Detection

Install dependencies:

pip install -r requirements.txt

Usage

Run the script on an image:

python lane_detection.py --image path/to/image.jpg

Run the script on a video:

python lane_detection.py --video path/to/video.mp4

Output

The processed images or videos will display lane markings detected by the algorithm, overlaid on the original input.


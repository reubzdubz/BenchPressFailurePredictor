# BenchPressFailurePredictor
A short school project using computer vision and time series classification to predict bench press rep failures in real-time, providing immediate audio feedback to prevent injury.

## Overview

Virtual Spotter combines MediaPipe pose estimation with an LSTM neural network that was trained on extracted pose data to analyze bench press form and predict potential rep failures. The system monitors elbow angle and wrist position throughout each rep, classifying reps as "normal" or "failure" at completion and alerting users with a beep sound when failure is detected.

## Features

- **Real-time pose tracking** using MediaPipe with optimized ROI tracking
- **LSTM-based failure prediction** trained on bench press motion data
- **Rep-based classification** - analyzes entire rep at completion (bottom → top transition)
- **Audio alerts** - beep notification on failure detection
- **Live visualization** - displays stage (top/bottom), rep count, and failure risk on webcam feed
- **Performance metrics** - latency and FPS monitoring

## Demo

"""TO ADD DEMO VIDEO"""

## Architecture

### Signal Extraction
- **Elbow angle**: Calculated from shoulder-elbow-wrist landmarks (3-point angle)
- **Wrist Y-position**: Normalized vertical position as bar height proxy
- Both signals are smoothed using a rolling 5-frame window

### Stage Detection
- **Top stage**: Elbow angle ≥ 165° (arms extended)
- **Bottom stage**: Elbow angle ≤ 70° (bar at chest)
- Rep completion detected on bottom → top transition

### LSTM Classifier
- **Input**: 400-frame sliding window (left-padded)
- **Features**: 2 (elbow angle, wrist y-position)
- **Architecture**: 4-layer LSTM with 128 hidden units
- **Output**: Binary classification (0=normal, 1=failure)

## Installation

### Requirements
- Python 3.8+
- Webcam (or phone camera connected to Wi-Fi
- CUDA-capable GPU (optional, for faster inference)

### Setup

1. Clone the repository:

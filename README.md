# Adaptive Video Streaming

## Project Overview

Adaptive Video Streaming is a computer-vision-based video streaming system that dynamically adjusts video quality according to detected activity and network conditions.

The system captures video from a webcam, uses YOLO-based detection and activity classification, selects an appropriate streaming quality tier, transmits the video to a server, measures network RTT, and records streaming information for evaluation.

## Objectives

- Capture live video using a webcam.
- Detect people and activity using computer vision.
- Classify activity such as IDLE, MOTION, PERSON, and LOITERING.
- Dynamically select an appropriate video quality tier.
- Monitor network performance using RTT.
- Log streaming information for analysis.
- Compare baseline streaming with adaptive streaming.
- Measure bandwidth reduction.
- Generate graphs for project evaluation.

## System Workflow

```text
Webcam
   |
   v
Video Frame Capture
   |
   v
YOLO Detection
   |
   v
Activity Classification
   |
   v
Tier Controller
   |
   v
Adaptive Video Quality
   |
   v
Network Monitoring
   |
   v
Socket Transmission
   |
   v
Server
   |
   v
Video Display
   |
   v
CSV Logging & Evaluation
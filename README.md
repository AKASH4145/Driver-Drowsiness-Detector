## Driver Drowsiness Detector 

 A real-time assistive safety system designed to help drivers stay awake and alert while navigating safely. The system captures live video from a webcam, detects faces using **MediaPipe's FaceLandmarker**, and determines the state of the eyes by tracking facial landmarks and computing the Eye Aspect Ratio (EAR).

 When the driver's eyes remain closed or squinted beyond a safe threshold for a consecutive number of frames, the system triggers a continuous audible alert using a background thread and a visual "DROWSINESS ALERT!" warning on the screen to wake the driver.

 Designed to be lightweight and efficient, the project runs entirely on a standard computer CPU without requiring any heavy GPU or deep learning frameworks, making it accessible and practical for real-world usage.

---

## Motivation

 Drowsy driving is a leading cause of severe car accidents worldwide. Existing driver monitoring systems are often built into expensive modern vehicles or require specialized infrared hardware. This project explores how modern computer vision can be leveraged into a simple, affordable, and highly accessible safety tool. By automatically tracking facial landmarks and the geometry of the eye, the system aims to provide critical real-time alerts. 
 
 The goal is to demonstrate how lightweight algorithms running on standard hardware (like a dashboard-mounted smartphone or a laptop webcam) can meaningfully improve safety and potentially save lives without relying on complex computational infrastructure.

---

## Features

 - Real-time face and eye detection using MediaPipe FaceLandmarker
 - Eye Aspect Ratio (EAR) calculation for reliable blink detection
 - Mouth Aspect Ratio (MAR) calculation for yawn detection
 - Fully automatic pre-trained landmark model downloading
 - Asynchronous audible alerts using `winsound` threading to ensure smooth video playback
 - Configurable sensitivity (threshold mapping and consecutive frames count)
 - Lightweight — runs entirely on the CPU
 - Works instantly with any standard webcam feed

---

## Concepts Used

 - Face Detection & Landmark Tracking (MediaPipe FaceLandmarker)
 - Eye Aspect Ratio (EAR) formula and Euclidean distance math
 - Mouth Aspect Ratio (MAR) formula for yawn detection
 - Multithreading for non-blocking asynchronous audio alerts
 - Real-time video processing and visual annotation

---

## Tech Stack

 - Language — Python
 - Computer Vision — OpenCV (`cv2`)
 - Facial Recognition & Landmark Detection — MediaPipe (`mediapipe`)
 - Numerical Processing — NumPy 
 - Audio Playback — `winsound` (Native Windows Module)
 - Multi-threading — Python `threading`

---

## System Architecture

Workflow:
1. Live video feed (webcam frame capture)
2. Auto-provisioning of `face_landmarker.task` model if missing
3. Frame conversion to MediaPipe Image format
4. Face & Landmark Detection (MediaPipe FaceLandmarker)
5. Extraction of Left and Right Eye coordinates
6. Compute Euclidean logic for Eye Aspect Ratio (EAR)
7. Compute Euclidean logic for Mouth Aspect Ratio (MAR)
8. Drowsiness check (EAR < 0.25)
9. Yawn check (MAR > 0.6)
10. Consecutive frame tracking (EAR: 20 frames, Yawn: 15 frames)
11. Asynchronous thread trigger for `winsound.Beep` if drowsy
12. Visual Overlay (Eye contours, MAR/EAR metrics, and Alert Text displayed)
13. Display live window

---

## Project Structure

```text

Driver Drowsiness Detector/
├── ddd/                       # Virtual Environment
├── detect.py                  # Core Detection logic
├── requirements.txt           # Dependencies
└── README.md                  # Project Documentation

```
---

## Setup & Run

- `git clone https://github.com/AKASH4145/Driver-Drowsiness-Detector`
- `cd Driver-Drowsiness-Detector`
- `pip install -r requirements.txt`  
- `python detect.py`

*(Note: The `face_landmarker.task` model is downloaded automatically!)*

---

## Demo Screenshots and Video

![Driver Awake](Demo%20Screenshots%20%26%20Recording/Awake.png)

![Driver Yawning](Demo%20Screenshots%20%26%20Recording/Yawning.png)

![Driver Drowsy](Demo%20Screenshots%20%26%20Recording/Drowzy%20detected.png)

🎥 **Demo Video:** [Watch on Drive](https://drive.google.com/file/d/1aqU8oFkwPp50V6ke1Jh53XangXQd7H-5/view?usp=sharing)
---

## Observations

- The Eye Aspect Ratio (EAR) provides an incredibly mathematically stable method for determining if an eye is closed compared to pure image pixel classification.
- Dlib's CPU-based HOG face detector is fast enough to run in real-time, although lighting heavily impacts reliability.
- Implementing the alarm on an asynchronous thread prevents the `cv2.imshow` framerate from freezing or dropping while the alarm rings.

---

## Limitations

- Detection accuracy depends heavily on lighting conditions; performs poorly in the dark without an IR camera.
- Glare from glasses or thick frames can sometimes obscure the eye landmarks.
- Works best with a stable front-facing camera.
- Cannot currently detect yawning or head nods.

---

## Future Scope

- Yawn Detection (monitoring the distance between mouth landmarks)
- Head Pose Estimation (detecting if the driver's head is tilted downward)
- Night Vision support (using an infrared webcam)
- Integration with Raspberry Pi for a dedicated in-car dashboard system
- Custom `.wav` alarm tones using PyGame instead of `winsound`

---

## Applications 

- Advanced Driver Assistance Systems (ADAS) — Essential baseline logic for in-car safety monitors.
- Commercial Fleet Management — Tracking and keeping truck/bus drivers awake on long-haul routes.
- Study Aid — Preventing students from falling asleep at their desks while studying late.
- Train Operations — Monitoring conductors or operators of heavy machinery.
- Computer Vision Research — Serves as an excellent foundational project for learning spatial mapping in Python.

--- 

## Author

Akash GS | Mechanical Engineering student exploring AI, computer vision, and applied Python development

---
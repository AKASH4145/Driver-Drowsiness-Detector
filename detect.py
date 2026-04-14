import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import threading

# ─────────────────────────────────────────────
# Cross-platform alarm
# ─────────────────────────────────────────────
try:
    import winsound
    def _beep():
        winsound.Beep(1000, 500)
except ImportError:
    def _beep():
        print("\a", end="", flush=True)

alarm_event = threading.Event()

def alarm_worker():
    while alarm_event.is_set():
        _beep()

def start_alarm():
    if not alarm_event.is_set():
        alarm_event.set()
        threading.Thread(target=alarm_worker, daemon=True).start()

def stop_alarm():
    alarm_event.clear()


# ─────────────────────────────────────────────
# MediaPipe landmark indices for eyes & mouth
# ─────────────────────────────────────────────
LEFT_EYE  = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33,  160, 158, 133, 153, 144]
MOUTH     = [61,  291, 39,  181, 0,   17,  269, 405]


def eye_aspect_ratio(landmarks, eye_indices, w, h):
    pts = []
    for idx in eye_indices:
        lm = landmarks[idx]
        pts.append((lm.x * w, lm.y * h))
    A = np.linalg.norm(np.array(pts[1]) - np.array(pts[5]))
    B = np.linalg.norm(np.array(pts[2]) - np.array(pts[4]))
    C = np.linalg.norm(np.array(pts[0]) - np.array(pts[3]))
    ear = (A + B) / (2.0 * C)
    return ear, pts


def mouth_aspect_ratio(landmarks, mouth_indices, w, h):
    pts = []
    for idx in mouth_indices:
        lm = landmarks[idx]
        pts.append((lm.x * w, lm.y * h))
    A = np.linalg.norm(np.array(pts[2]) - np.array(pts[6]))
    B = np.linalg.norm(np.array(pts[3]) - np.array(pts[7]))
    C = np.linalg.norm(np.array(pts[0]) - np.array(pts[1]))
    mar = (A + B) / (2.0 * C)
    return mar


def draw_eye_contour(frame, pts, color):
    points = np.array(pts, dtype=np.int32)
    hull = cv2.convexHull(points)
    cv2.drawContours(frame, [hull], -1, color, 1)


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────
def main():
    EAR_THRESH         = 0.25
    EAR_CONSEC_FRAMES  = 20
    MAR_THRESH         = 0.6
    YAWN_CONSEC_FRAMES = 15
    EAR_COUNTER        = 0
    YAWN_COUNTER       = 0

    # Use the new mediapipe tasks API
    base_options = python.BaseOptions(model_asset_path='face_landmarker.task')
    options = vision.FaceLandmarkerOptions(base_options=base_options, num_faces=1)
    face_landmarker = vision.FaceLandmarker.create_from_options(options)

    cap = cv2.VideoCapture(1)
    if not cap.isOpened():
        cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("[ERROR] No camera found.")

    print("[INFO] Drowsiness detector running. Press Q or Enter to quit.")

    while True:
        ret, frame = cap.read()
        if not ret or frame is None or frame.size == 0:
            continue

        h, w = frame.shape[:2]

        # Convert frame to MediaPipe Image
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        
        # Detect face landmarks
        detection_result = face_landmarker.detect(mp_image)

        status_text  = "Awake"
        status_color = (0, 255, 0)

        if detection_result.face_landmarks:
            for face_landmarks in detection_result.face_landmarks:
                lms = face_landmarks

                left_ear,  left_pts  = eye_aspect_ratio(lms, LEFT_EYE,  w, h)
                right_ear, right_pts = eye_aspect_ratio(lms, RIGHT_EYE, w, h)
                ear = (left_ear + right_ear) / 2.0
                mar = mouth_aspect_ratio(lms, MOUTH, w, h)

                eye_color = (0, 255, 0) if ear >= EAR_THRESH else (0, 0, 255)
                draw_eye_contour(frame, left_pts,  eye_color)
                draw_eye_contour(frame, right_pts, eye_color)

                # Drowsiness check
                if ear < EAR_THRESH:
                    EAR_COUNTER += 1
                    if EAR_COUNTER >= EAR_CONSEC_FRAMES:
                        start_alarm()
                        status_text  = "DROWSY! WAKE UP!"
                        status_color = (0, 0, 255)
                    else:
                        status_text  = f"Eyes closing... ({EAR_COUNTER})"
                        status_color = (0, 165, 255)
                else:
                    EAR_COUNTER = 0
                    stop_alarm()

                # Yawn check
                if mar > MAR_THRESH:
                    YAWN_COUNTER += 1
                    if YAWN_COUNTER >= YAWN_CONSEC_FRAMES:
                        cv2.putText(frame, "YAWN DETECTED",
                            (10, 90), cv2.FONT_HERSHEY_SIMPLEX,
                            0.7, (0, 165, 255), 2)
                else:
                    YAWN_COUNTER = 0

                # HUD
                cv2.putText(frame, f"EAR: {ear:.2f}",
                    (w - 160, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (0, 255, 0) if ear >= EAR_THRESH else (0, 0, 255), 2)
                cv2.putText(frame, f"MAR: {mar:.2f}",
                    (w - 160, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (0, 255, 0) if mar <= MAR_THRESH else (0, 165, 255), 2)

        else:
            status_text  = "No face detected"
            status_color = (128, 128, 128)

        # Status text
        cv2.putText(frame, status_text,
            (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
            0.8, status_color, 2)

        # Red border when alarm active
        if alarm_event.is_set():
            cv2.rectangle(frame, (0, 0), (w - 1, h - 1), (0, 0, 255), 4)

        cv2.imshow("Driver Drowsiness Detector", frame)
        key = cv2.waitKey(1) & 0xFF
        if key in (13, ord("q"), ord("Q")):
            break

    stop_alarm()
    face_landmarker.close()
    cap.release()
    cv2.destroyAllWindows()
    print("[INFO] Exited cleanly.")


if __name__ == "__main__":
    main()

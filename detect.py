import dlib
import cv2
import numpy as np
import winsound
import os
import urllib.request
import bz2
import threading

# Helper function to compute euclidean distance
def euclidean(pt1, pt2):
    return np.linalg.norm(np.array(pt1) - np.array(pt2))

def eye_aspect_ratio(eye):
    # compute the euclidean distances between the two sets of
    # vertical eye landmarks (x, y)-coordinates
    A = euclidean(eye[1], eye[5])
    B = euclidean(eye[2], eye[4])

    # compute the euclidean distance between the horizontal
    # eye landmark (x, y)-coordinates
    C = euclidean(eye[0], eye[3])

    # compute the eye aspect ratio
    ear = (A + B) / (2.0 * C)

    # return the eye aspect ratio
    return ear

def download_landmark_model():
    model_file = "shape_predictor_68_face_landmarks.dat"
    if not os.path.exists(model_file):
        print("Downloading shape_predictor_68_face_landmarks.dat...")
        url = "http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2"
        zip_path = model_file + ".bz2"
        urllib.request.urlretrieve(url, zip_path)
        print("Extracting...")
        with bz2.BZ2File(zip_path, 'rb') as source, open(model_file, 'wb') as dest:
            dest.write(source.read())
        os.remove(zip_path)
        print("Downloaded and extracted model.")
        
def get_landmarks(shape):
    # Convert dlib's shape object to numpy array
    coords = np.zeros((68, 2), dtype="int")
    for i in range(0, 68):
        coords[i] = (shape.part(i).x, shape.part(i).y)
    return coords

alarm_on = False
def sound_alarm_thread():
    global alarm_on
    while alarm_on:
        winsound.Beep(1000, 500)

if __name__=='__main__':
    download_landmark_model()

    # Load the pre-trained face detection model and the facial landmark predictor
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

    EYE_AR_THRESH = 0.25
    EYE_AR_CONSEC_FRAMES = 20

    COUNTER = 0

    # indexes of the facial landmarks for the left and right eye (0-indexed)
    (lStart, lEnd) = (42, 48)
    (rStart, rEnd) = (36, 42)

    # Note: trying port 1 then fallback to 0
    cap = cv2.VideoCapture(1)
    if not cap.isOpened():
        cap = cv2.VideoCapture(0)

    while True:
        ret, frame =cap.read()
        if not ret:
            break

        # Convert the frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces in the image
        faces = detector(gray, 0)

        for face in faces:
            # Determine the facial landmarks for the face region
            shape = predictor(gray, face)
            shape = get_landmarks(shape)

            leftEye = shape[lStart:lEnd]
            rightEye = shape[rStart:rEnd]
            
            leftEAR = eye_aspect_ratio(leftEye)
            rightEAR = eye_aspect_ratio(rightEye)

            # average the eye aspect ratio together for both eyes
            ear = (leftEAR + rightEAR) / 2.0

            leftEyeHull = cv2.convexHull(leftEye)
            rightEyeHull = cv2.convexHull(rightEye)
            cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
            cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

            if ear < EYE_AR_THRESH:
                COUNTER += 1
                if COUNTER >= EYE_AR_CONSEC_FRAMES:
                    if not alarm_on:
                        alarm_on = True
                        t = threading.Thread(target=sound_alarm_thread, daemon=True)
                        t.start()
                        
                    cv2.putText(frame, "DROWSINESS ALERT!", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            else:
                COUNTER = 0
                alarm_on = False

            cv2.putText(frame, "EAR: {:.2f}".format(ear), (300, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        cv2.imshow('frame', frame)
        if cv2.waitKey(1) == 13: # ASCII corresponding to Enter key
            break

    alarm_on = False
    cap.release()
    cv2.destroyAllWindows()
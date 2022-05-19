import cv2
import mediapipe as mp
import numpy as np
from gaze_tracking import GazeTracking

face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_alt.xml")
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)
gaze = GazeTracking()
webcam = cv2.VideoCapture(0)

while True:
    # We get a new frame from the webcam
    _, frame = webcam.read()
    _, show = webcam.read()

    # Flip the image horizontally for a later selfie-view display
    frame = cv2.flip(frame, 1)
    show = cv2.flip(show, 1)

    # We send this frame to GazeTracking to analyze it
    gaze.refresh(frame)
    gaze.refresh(show)

    text_eye = ""
    if gaze.is_blinking():
        text_eye = "EYE: closed"
    elif gaze.is_right():
        text_eye = "EYE: Looking right"
    elif gaze.is_left():
        text_eye = "EYE: Looking left"
    elif gaze.is_center():
        text_eye = "EYE: Looking center"

    left_pupil = gaze.pupil_left_coords()
    right_pupil = gaze.pupil_right_coords()

    detections = face_cascade.detectMultiScale(frame, scaleFactor=1.1, minNeighbors=6)

    # Get the result
    results = face_mesh.process(frame)

    img_h, img_w, img_c = frame.shape
    face_3d = []
    face_2d = []

    score_eye = 0
    a = 0
    b = 0
    c = 0
    d = 0
    for face in detections:
        x, y, w, h = face
        a, b, c, d = face

        frame = gaze.annotated_frame(frame)  # eye_tracking

        # 보여주기용 윈도우
        show[y:y + h, x:x + w] = cv2.GaussianBlur(show[y:y + h, x:x + w], (0, 0), 10)
        #cv2.rectangle(show, (x, y), (x + w, y + h), (0, 0, 0), 2)
        show = gaze.annotated_frame(show)

        if text_eye == "EYE: closed" or text_eye == "":
            cv2.putText(show, "EYE: closed", (15, 80), cv2.FONT_HERSHEY_DUPLEX, 0.8, (0, 0, 255), 2)
            cv2.putText(show, "0", (550, 80), cv2.FONT_HERSHEY_DUPLEX, 0.8, (0, 0, 255), 2)
            cv2.putText(show, "Left pupil: None", (15, 440), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (153, 0, 76), 2)
            cv2.putText(show, "Right pupil: None", (15, 460), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (153, 0, 76), 2)
            score_eye = 0
        else:
            if text_eye == "EYE: Looking right":
                cv2.putText(show, str(round(gaze.horizontal_ratio(), 2)), (550, 80), cv2.FONT_HERSHEY_DUPLEX, 0.8, (153, 0, 76), 2)
            else:
                cv2.putText(show, str(round(gaze.horizontal_ratio(), 2)), (550, 80), cv2.FONT_HERSHEY_DUPLEX, 0.8, (153, 0, 76), 2)
            cv2.putText(show, text_eye, (15, 80), cv2.FONT_HERSHEY_DUPLEX, 0.8, (153, 0, 76), 2)
            cv2.putText(show, "Left pupil: " + str(left_pupil), (15, 440), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (153, 0, 76), 2)
            cv2.putText(show, "Right pupil: " + str(right_pupil), (15, 460), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (153, 0, 76), 2)
            score_eye = gaze.horizontal_ratio()

    score_face = 0
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            for idx, lm in enumerate(face_landmarks.landmark):
                if idx == 33 or idx == 263 or idx == 1 or idx == 61 or idx == 291 or idx == 199:
                    if idx == 1:
                        nose_2d = (lm.x * img_w, lm.y * img_h)
                        nose_3d = (lm.x * img_w, lm.y * img_h, lm.z * 8000)

                    x, y = int(lm.x * img_w), int(lm.y * img_h)

                    # Get the 2D Coordinates
                    face_2d.append([x, y])

                    # Get the 3D Coordinates
                    face_3d.append([x, y, lm.z])

                    # Convert it to the NumPy array
            face_2d = np.array(face_2d, dtype=np.float64)

            # Convert it to the NumPy array
            face_3d = np.array(face_3d, dtype=np.float64)

            # The camera matrix
            focal_length = 1 * img_w

            cam_matrix = np.array([[focal_length, 0, img_h / 2],
                                   [0, focal_length, img_w / 2],
                                   [0, 0, 1]])

            # The Distance Matrix
            dist_matrix = np.zeros((4, 1), dtype=np.float64)

            # Solve PnP
            success, rot_vec, trans_vec = cv2.solvePnP(face_3d, face_2d, cam_matrix, dist_matrix)

            # Get rotational matrix
            rmat, jac = cv2.Rodrigues(rot_vec)

            # Get angles
            angles, mtxR, mtxQ, Qx, Qy, Qz = cv2.RQDecomp3x3(rmat)

            # Get the y rotation degree
            x = angles[0] * 360
            y = angles[1] * 360

            text_face = ""
            score_face = 0

            # See where the user's head tilting
            if y < -10:
                text_face = "FACE: Looking Right"
                score_face = y
            elif y > 10:
                text_face = "FACE: Looking Left"
                score_face = y
            elif x < -10:
                text_face = "FACE: Looking Down"
                score_face = x
            else:
                text_face = "FACE: Looking Forward"
                score_face = y

            # Add the text on the image
            cv2.putText(show, text_face, (15, 40), cv2.FONT_HERSHEY_DUPLEX, 0.8, (153, 0, 76), 2)
            if score_face >= 0:
                cv2.putText(show, str(round(score_face, 2)), (550, 40), cv2.FONT_HERSHEY_DUPLEX, 0.8, (153, 0, 76), 2)
            else:
                cv2.putText(show, str(round(score_face, 2)), (530, 40), cv2.FONT_HERSHEY_DUPLEX, 0.8, (153, 0, 76), 2)

            # 정면으로 인식했는데 동공 인식이 안 됐을 경우 눈을 감고 있다고 판단.
            if text_face == "FACE: Looking Forward" and text_eye == "":
                cv2.putText(show, "EYE: closed", (15, 80), cv2.FONT_HERSHEY_DUPLEX, 0.8, (0, 0, 255), 2)


        # final score_face
        score_face = abs(score_face)
        if score_face > 25:
            score_face = 25
        score_face = 50 - (score_face * 2)

        # final score_eye
        if score_eye != 0:  # 눈을 떴고 잘 인식되었을 경우
            score_eye = (score_eye - 0.5) * 100
            score_eye = abs(score_eye)
            if score_eye > 30:
                score_eye = 30
            score_eye = 50 - (score_eye * (5/3))

        # total score
        if score_eye == 0:  # 눈을 감고 있거나 인식이 안되었을 경우
            score = 0
        else:
            score = score_face + score_eye

        perc = "%"
        #cv2.putText(show, str(round(score, 2)) + perc, (((a+c)/2), (b-50)), cv2.FONT_HERSHEY_DUPLEX, 1, (10, 10, 230), 2)
        cv2.putText(show, "score_eye: " + str(round(score_eye, 2)), (15, 300), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (153, 0, 76), 2)
        cv2.putText(show, "score_face: " + str(round(score_face, 2)), (15, 320), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (153, 0, 76), 2)

        if score > 60:
            cv2.rectangle(show, (a, b), (a + c, b + d), (255, 0, 0), 2)
            cv2.putText(show, str(round(score, 2)) + perc, (15, 250), cv2.FONT_HERSHEY_DUPLEX, 1, (255, 0, 0), 2)

        elif score > 40:
            cv2.rectangle(show, (a, b), (a + c, b + d), (0, 255, 0), 2)
            cv2.putText(show, str(round(score, 2)) + perc, (15, 250), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 255, 0), 2)

        else:
            cv2.rectangle(show, (a, b), (a + c, b + d), (0, 0, 255), 2)
            cv2.putText(show, str(round(score, 2)) + perc, (15, 250), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 255), 2)

    cv2.imshow("eye_tracking", frame)
    cv2.imshow("result", show)
    cv2.moveWindow("eye_tracking", 520, 100)
    cv2.moveWindow("result", 520, 100)

    if cv2.waitKey(1) == 27:
        break

webcam.release()
cv2.destroyAllWindows()

import cv2
import mediapipe as mp
import numpy as np

from gaze_tracking import GazeTracking

gaze = GazeTracking()
webcam = cv2.VideoCapture(0)

face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_alt.xml")

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)

while True:
    # We get a new frame from the webcam
    _, frame, test = webcam.read()

    # We send this frame to GazeTracking to analyze it
    gaze.refresh(frame)

    text_eye = ""
    if gaze.is_blinking():
        text_eye = "EYE: Blinking"
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
    for face in detections:
        x, y, w, h = face

        #frame[y:y + h, x:x + w] = cv2.GaussianBlur(frame[y:y + h, x:x + w], (15, 15), cv2.BORDER_DEFAULT) # blur
        frame = gaze.annotated_frame() # eye_tracking

        if text_eye == "EYE: Blinking":
            cv2.putText(frame, text_eye, (15, 80), cv2.FONT_HERSHEY_DUPLEX, 0.8, (10, 10, 230), 2)
            cv2.putText(frame, "0", (560, 80), cv2.FONT_HERSHEY_DUPLEX, 0.8, (40, 40, 230), 2)
            cv2.putText(frame, "Left pupil: None", (15, 420), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (10, 200, 20), 2)
            cv2.putText(frame, "Right pupil: None", (15, 455), cv2.FONT_HERSHEY_SIMPLEX, 0.8,(10, 200, 20), 2)
            score_eye = 0
        else:
            if text_eye == "EYE: Looking right":
                cv2.putText(frame, str(round(gaze.horizontal_ratio(),2)), (550, 80), cv2.FONT_HERSHEY_DUPLEX, 0.8, (150, 60, 30), 2)
            else:
                cv2.putText(frame, str(round(gaze.horizontal_ratio(),2)), (550, 80), cv2.FONT_HERSHEY_DUPLEX, 0.8, (150, 60, 30), 2)
            cv2.putText(frame, text_eye, (15, 80), cv2.FONT_HERSHEY_DUPLEX, 0.8, (150, 60, 30), 2)
            cv2.putText(frame, "Left pupil: " + str(left_pupil), (15, 420), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (10, 200, 20), 2)
            cv2.putText(frame, "Right pupil: " + str(right_pupil), (15, 455), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (10, 200, 20), 2)
            score_eye = abs(gaze.horizontal_ratio())

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

            # print(y)

            text_face = ""
            rate_face = 0
            #score_face = 0

            # See where the user's head tilting
            if y < -10:
                text_face = "FACE: Looking Right"
                rate_face = y
                score_face = abs(rate_face)
            elif y > 10:
                text_face = "FACE: Looking Left"
                rate_face = y
                score_face = rate_face
            elif x < -10:
                text_face = "FACE: Looking Down"
                rate_face = x
                score_face = abs(rate_face)
            else:
                text_face = "FACE: Looking Forward"
                rate_face = y
                score_face = rate_face

            # Display the nose direction
            '''
            nose_3d_projection, jacobian = cv2.projectPoints(nose_3d, rot_vec, trans_vec, cam_matrix, dist_matrix)

            p1 = (int(nose_2d[0]), int(nose_2d[1]))
            p2 = (int(nose_3d_projection[0][0][0]), int(nose_3d_projection[0][0][1]))

            cv2.line(image, p1, p2, (255, 0, 0), 2)
            '''

            # Add the text on the image
            cv2.putText(frame, text_face, (15, 40), cv2.FONT_HERSHEY_DUPLEX, 0.8, (150, 60, 30), 2)
            if rate_face >= 0:
                cv2.putText(frame, str(round(rate_face,2)), (550, 40), cv2.FONT_HERSHEY_DUPLEX, 0.8, (150, 60, 30), 2)
            else:
                cv2.putText(frame, str(round(rate_face,2)), (530, 40), cv2.FONT_HERSHEY_DUPLEX, 0.8, (150, 60, 30), 2)

        # total score
        score = score_face + score_eye
        #if score_eye == 0:
        #    cv2.putText(frame, str(round(score, 2)), (15, 200), cv2.FONT_HERSHEY_DUPLEX, 1, (10, 10, 230), 2)
        cv2.putText(frame, str(round(score_eye, 2)), (15, 240), cv2.FONT_HERSHEY_DUPLEX, 1, (10, 10, 230), 2)
        cv2.putText(frame, str(round(score_face, 2)), (15, 280), cv2.FONT_HERSHEY_DUPLEX, 1, (10, 10, 230), 2)

    cv2.imshow("real-time video", frame)

    if cv2.waitKey(1) == 27:
        break


webcam.release()
cv2.destroyAllWindows()
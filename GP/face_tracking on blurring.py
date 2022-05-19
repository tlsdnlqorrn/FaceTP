import cv2
import mediapipe as mp
import numpy as np

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_alt.xml")
cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, image = cap.read()

    # Flip the image horizontally for a later selfie-view display
    image = cv2.flip(image, 1)

    # Get the result
    results = face_mesh.process(image)
    detections = face_cascade.detectMultiScale(image, scaleFactor=1.1, minNeighbors=6)

    # Convert the color space from RGB to BGR
    #image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    img_h, img_w, img_c = image.shape
    face_3d = []
    face_2d = []

    detections = face_cascade.detectMultiScale(image, scaleFactor=1.1, minNeighbors=6)

    for face in detections:
        x, y, w, h = face

        image[y:y + h, x:x + w] = cv2.GaussianBlur(image[y:y + h, x:x + w], (15, 15), cv2.BORDER_DEFAULT)
        # cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)

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

            # See where the user's head tilting
            if y < -10:
                text = "Looking Left"
                cv2.putText(image, str(round(y, 5)), (520, 40), cv2.FONT_HERSHEY_DUPLEX, 0.8, (150, 60, 30), 2)
            elif y > 10:
                text = "Looking Right"
                cv2.putText(image, str(round(y, 5)), (530, 40), cv2.FONT_HERSHEY_DUPLEX, 0.8, (150, 60, 30), 2)
            elif x < -10:
                text = "Looking Down"
                cv2.putText(image, str(round(x, 5)), (530, 40), cv2.FONT_HERSHEY_DUPLEX, 0.8, (150, 60, 30), 2)
            else:
                text = "Forward"
                cv2.putText(image, str(round(y, 5)), (510, 40), cv2.FONT_HERSHEY_DUPLEX, 0.8, (150, 60, 30), 2)

            # Display the nose direction
            nose_3d_projection, jacobian = cv2.projectPoints(nose_3d, rot_vec, trans_vec, cam_matrix, dist_matrix)

            p1 = (int(nose_2d[0]), int(nose_2d[1]))
            p2 = (int(nose_3d_projection[0][0][0]), int(nose_3d_projection[0][0][1]))

           # cv2.line(image, p1, p2, (255, 0, 0), 2)

            # Add the text on the image
            cv2.putText(image, text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow('Head Pose Estimation', image)

    if cv2.waitKey(5) & 0xFF == 27:
        break

cap.release()
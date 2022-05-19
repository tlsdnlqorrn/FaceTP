import cv2
from gaze_tracking import GazeTracking

gaze = GazeTracking()
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_alt.xml")
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

    text = ""
    if gaze.is_blinking():
        text = "EYE: closed"
    elif gaze.is_right():
        text = "EYE: Looking right"
    elif gaze.is_left():
        text = "EYE: Looking left"
    elif gaze.is_center():
        text = "EYE: Looking center"

    left_pupil = gaze.pupil_left_coords()
    right_pupil = gaze.pupil_right_coords()

    detections = face_cascade.detectMultiScale(frame, scaleFactor=1.1, minNeighbors=6)

    score_eye = 0
    a = 0
    b = 0
    c = 0
    d = 0
    for face in detections:
        x, y, w, h = face
        a, b, c, d = face
        
        frame = gaze.annotated_frame(frame)  # eye_tracking

        show[y:y + h, x:x + w] = cv2.GaussianBlur(show[y:y + h, x:x + w], (0, 0), 10)
        show = gaze.annotated_frame(show)

        cv2.putText(show, "FACE: NONE", (15, 40), cv2.FONT_HERSHEY_DUPLEX, 0.8, (153, 0, 76), 2)
        cv2.putText(show, "--", (550, 40), cv2.FONT_HERSHEY_DUPLEX, 0.8, (153, 0, 76), 2)

        if text == "EYE: closed" or text == "":
            cv2.putText(show, "EYE: closed", (15, 80), cv2.FONT_HERSHEY_DUPLEX, 0.8, (0, 0, 255), 2)
            cv2.putText(show, "0", (550, 80), cv2.FONT_HERSHEY_DUPLEX, 0.8, (0, 0, 255), 2)
            cv2.putText(show, "Left pupil: None", (15, 440), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (153, 0, 76), 2)
            cv2.putText(show, "Right pupil: None", (15, 460), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (153, 0, 76), 2)
            cv2.putText(show, "0" + "%", (15, 250), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 255), 2)
            score_eye = 0
        else:
            if text == "EYE: Looking right":
                cv2.putText(show, str(round(gaze.horizontal_ratio(), 2)), (550, 80), cv2.FONT_HERSHEY_DUPLEX, 0.8,
                            (153, 0, 76), 2)
            else:
                cv2.putText(show, str(round(gaze.horizontal_ratio(), 2)), (550, 80), cv2.FONT_HERSHEY_DUPLEX, 0.8,
                            (153, 0, 76), 2)
            cv2.putText(show, text, (15, 80), cv2.FONT_HERSHEY_DUPLEX, 0.8, (153, 0, 76), 2)
            cv2.putText(show, "Left pupil: " + str(left_pupil), (15, 440), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (153, 0, 76),2)
            cv2.putText(show, "Right pupil: " + str(right_pupil), (15, 460), cv2.FONT_HERSHEY_SIMPLEX, 0.5,(153, 0, 76), 2)
            score_eye = gaze.horizontal_ratio()


        # final score_eye
        if score_eye != 0:  # 눈을 떴고 잘 인식되었을 경우
            score_eye = (score_eye - 0.5) * 100
            score_eye = abs(score_eye)
            if score_eye > 30:
                score_eye = 30
            score_eye = 100 - (score_eye * (10/3))

        # total score
        score = score_eye

        perc = "%"
        # cv2.putText(frame, str(round(score, 2)) + perc, (((a+c)/2), (b-50)), cv2.FONT_HERSHEY_DUPLEX, 1, (10, 10, 230), 2)
        cv2.putText(show, "score_face: " + "--", (15, 300), cv2.FONT_HERSHEY_SIMPLEX, 0.5,(153, 0, 76), 2)
        cv2.putText(show, "score_eye: " + str(round(score_eye, 2)), (15, 320), cv2.FONT_HERSHEY_SIMPLEX, 0.5,(153, 0, 76), 2)

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
import cv2

from gaze_tracking import GazeTracking

webcam = cv2.VideoCapture(0)
gaze = GazeTracking()

def resize(img, new_width=500):
    height, width, _ = img.shape
    ratio = height / width
    new_height = int(ratio * new_width)
    return cv2.resize(img, (new_width, new_height))

face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_alt.xml")

while True:
    # We get a new frame from the webcam
    _, frame = webcam.read()
    frame = cv2.flip(frame, 1)
    frame = resize(frame)

    # We send this frame to GazeTracking to analyze it
    gaze.refresh(frame)

    text = ""

    if gaze.is_blinking():
        text = "Blinking"
    elif gaze.is_right():
        text = "Looking right"
    elif gaze.is_left():
        text = "Looking left"
    elif gaze.is_center():
        text = "Looking center"

    left_pupil = gaze.pupil_left_coords()
    right_pupil = gaze.pupil_right_coords()

    detections = face_cascade.detectMultiScale(frame, scaleFactor=1.1, minNeighbors=6)

    for face in detections:
        x, y, w, h = face

        frame[y:y + h, x:x + w] = cv2.GaussianBlur(frame[y:y + h, x:x + w], (15, 15), cv2.BORDER_DEFAULT) # blur
        frame = gaze.annotated_frame() # eye_tracking

        if text == "Blinking":
            cv2.putText(frame, text, (15, 50), cv2.FONT_HERSHEY_DUPLEX, 1.6, (10, 10, 230), 2)
        else:
            cv2.putText(frame, text, (15, 50), cv2.FONT_HERSHEY_DUPLEX, 1.6, (150, 60, 30), 2)

        cv2.putText(frame, "Left pupil: " + str(left_pupil), (15, 320), cv2.FONT_HERSHEY_DUPLEX, 0.9, (150, 60, 30), 1)
        cv2.putText(frame, "Right pupil: " + str(right_pupil), (15, 355), cv2.FONT_HERSHEY_DUPLEX, 0.9, (150, 60, 30), 1)

        cv2.imshow("real-time video", frame)

        if cv2.waitKey(1) == 27:
            break

    if cv2.waitKey(1) == 27:
        break


webcam.release()
cv2.destroyAllWindows()
from imutils import face_utils
import numpy as np
import imutils
import dlib
import cv2
#얼굴 통채로 진행
def show_raw_detection(image, detector, predictor):
    # 이미지를 grayscale로 바꾸기
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    #사진 중 얼굴 부분만 그레이로 추출
    rects = detector(gray, 1)
    #추출된 얼굴을 하나씩 확인해본다.
    for (i, rect) in enumerate(rects):
    #얼굴 영역에 랜드마크를 결정한 다음 얼굴 랜드마크된 좌표를 numpy 배열로 저장한다.
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)
        landmark.append(shape)
    #dlib의 사각형의 좌표를 opencv 형태의 사각형으로 바꾼다.
    (x, y, w, h) = face_utils.rect_to_bb(rect)
    # 사각형을 그린다.
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
    # 얼굴의 개수를 표시해준다.
    cv2.putText(image, "Face #{}".format(i + 1), (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    # 이미지에 landmark를 표시한다.
    for (x, y) in shape:
        cv2.circle(image, (x, y), 1, (0, 0, 255), -1)
    #얼굴 검출과 랜드마킹된 결과를 보여준다.
    cv2.imshow("Output", image)
    cv2.waitKey(0)
#얼굴 부분별 랜드마크 추출 후 합치기
def draw_individual_detections(image, detector, predictor):
    #이미지를 흑백화한다.
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # 얼굴 부분만 그레이로 추출
    rects = detector(gray, 1)
    # 추출된 얼굴을 하나씩 확인해본다.
    for (i, rect) in enumerate(rects):
        # 얼굴 영역에 랜드마크를 결정한 다음 얼굴 랜드마크된 좌표를 numpy 배열로 저장한다.
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)
        # 얼굴 부분을 개별적으로 반복해준다.
        for (name, (i, j)) in face_utils.FACIAL_LANDMARKS_IDXS.items():
            # 원본 이미지를 복제하여 그릴 수 있도록 한 다음 이미지에 얼굴 부분의 이름을 표시합니다.
            clone = image.copy()
            cv2.putText(clone, name, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            # 얼굴 랜드마크의 각 부분을을 반복하여 특정 얼굴 부분 그리기
            for (x, y) in shape[i:j]:
                cv2.circle(clone, (x, y), 1, (0, 0, 255), -1)
                # 얼굴 영역의 ROI를 별도의 이미지로 추출
                (x, y, w, h) = cv2.boundingRect(np.array([shape[i:j]]))
                roi = image[y:y + h, x:x + w]
                roi = imutils.resize(roi, width=250, inter=cv2.INTER_CUBIC)
            # 특정 얼굴 부분을 보여준다.
            #cv2.imshow("ROI", roi)
            #cv2.imshow("Image", clone)
            cv2.waitKey(0)
        # 모든 얼굴 랜드마크를 시각화
        output = face_utils.visualize_facial_landmarks(image, shape)
        #cv2.imshow("Image", output)
        cv2.waitKey(0)
# dblib의 얼굴 감지기(HOG 기반)를 초기화 한 다음 얼굴 랜드마크 예측기를 생성
detector = dlib.get_frontal_face_detector()
# 다운 링크 http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
# 이미지를 불러오고 resize하고 흑백화시킨다.
landmark=[]
image = cv2.imread('image/face.jpg')
image = imutils.resize(image, width=500)
show_raw_detection(image, detector, predictor)
#draw_individual_detections(image, detector, predictor)
print(landmark)

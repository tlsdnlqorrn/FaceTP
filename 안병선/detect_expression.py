from imutils import face_utils
import numpy as np
import imutils
import dlib
import sys
import glob
import cv2
class detect_expression:
    def detect(count, img_files):
        # dblib의 얼굴 감지기(HOG 기반)를 초기화 한 다음 얼굴 랜드마크 예측기를 생성
        detector = dlib.get_frontal_face_detector()
        # 다운 링크 http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
        predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
        # 이미지를 불러오고 resize하고 흑백화시킨다.
        landmark=[]

        for i in range(count):
            image = cv2.imread(img_files[i])
            image=cv2.resize(image,dsize=(640,640))
            #image = imutils.resize(image, width=500)
            # 이미지를 grayscale로 바꾸기
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            # 사진 중 얼굴 부분만 그레이로 추출
            rects = detector(gray, 1)
            # 추출된 얼굴을 하나씩 확인해본다.
            for (i, rect) in enumerate(rects):
                # 얼굴 영역에 랜드마크를 결정한 다음 얼굴 랜드마크된 좌표를 numpy 배열로 저장한다.
                shape = predictor(gray, rect)
                shape = face_utils.shape_to_np(shape)
                landmark.append(shape)
            # dlib의 사각형의 좌표를 opencv 형태의 사각형으로 바꾼다.
            (x, y, w, h) = face_utils.rect_to_bb(rect)
            # 사각형을 그린다.
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            # 얼굴의 개수를 표시해준다.
            cv2.putText(image, "Face #{}".format(i + 1), (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            # 이미지에 landmark를 표시한다.
            for (x, y) in shape:
                cv2.circle(image, (x, y), 1, (0, 0, 255), -1)
            # 얼굴 검출과 랜드마킹된 결과를 보여준다.
            #print(landmark[i][17:])
            #cv2.imshow("Output", image)
            #cv2.waitKey(0)
        return landmark
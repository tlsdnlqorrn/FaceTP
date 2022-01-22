import numpy as np
import sys
import glob
import cv2
from detect_expression import detect_expression

# images에 있는 모든 jpg 파일을 img_files 리스트에 추가
img_files = glob.glob('.\\image\\*.jpg')
# 이미지 없을때 예외처리
if not img_files:
    print("jpg 이미지가 없어요..")
    sys.exit()

face_files = glob.glob('.\\face\\*.jpg')

if not face_files:
    print("jpg 이미지가 없어요..")
    sys.exit()


standard_landmark=detect_expression.detect(24,img_files)
analyze_face=detect_expression.detect(len(face_files),face_files)

#정규화
row_sums = sum(standard_landmark)
n_standard_landmark = standard_landmark / row_sums
row_sums = sum(analyze_face)
n_analyze_face = analyze_face / row_sums

for j in range(len(analyze_face)):
    distance = []
    for i in range(23):
        distance.append(np.linalg.norm(n_standard_landmark[i][17:]-n_analyze_face[j][17:]))
    les=min(distance)
    index=distance.index(les)
    print(index)
    image = cv2.imread(img_files[index])
    cv2.imshow("output",image)
    cv2.waitKey(0)
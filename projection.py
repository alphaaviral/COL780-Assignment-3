import numpy as np
import cv2
from pupil_apriltags import Detector

at_detector = Detector(
   families="tag36h11",
)

img = cv2.imread('3d.jpg', cv2.IMREAD_COLOR)
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

res = at_detector.detect(img_gray)
for tag in res:
    tag.corners = np.rint(tag.corners).astype('int32')
    tag.center = np.rint(tag.center).astype('int32')
    # img = cv2.putText(img, str(tag.tag_id), int_center, cv2.FONT_HERSHEY_SIMPLEX,10, (0,0,255), 10 )
    # for corner in int_corners:
    #     img = cv2.circle(img, corner, 15, (0,0,255), -1)
# cv2.imwrite('image.jpg', img)
A = np.zeros((12, 12))
# print(res)
calibration_points = np.zeros((6,2), dtype = 'int32')
for i in range(4):
  calibration_points[i][0] = res[0].corners[i][0]
  calibration_points[i][1] = res[0].corners[i][1]
for i in range(2):
  calibration_points[i+4][0] = res[1].corners[i][0]
  calibration_points[i+4][1] = res[1].corners[i][1]

# print(calibration_points)
world = [[0,10,0], [10,10,0], [10,10,10], [0, 10, 10], [0,0,0], [10, 0, 0]]
for i in range(6):
  j=i*2
  A[j][0] = world[i][0]
  A[j][1] = world[i][1]
  A[j][2] = world[i][2]
  A[j][3] = 1
  A[j][4] = 0
  A[j][5] = 0
  A[j][6] = 0
  A[j][7] = 0
  A[j][8] = -1*world[i][0]*calibration_points[i][0]
  A[j][9] = -1*world[i][1]*calibration_points[i][0]
  A[j][10] = -1*world[i][2]*calibration_points[i][0]
  A[j][11] = -1*calibration_points[i][0]
  A[j+1][0] = 0
  A[j+1][1] = 0
  A[j+1][2] = 0
  A[j+1][3] = 0
  A[j+1][4] = world[i][0]
  A[j+1][5] = world[i][1]
  A[j+1][6] = world[i][2]
  A[j+1][7] = 1
  A[j+1][8] = -1*world[i][0]*calibration_points[i][1]
  A[j+1][9] = -1*world[i][1]*calibration_points[i][1]
  A[j+1][10] = -1*world[i][2]*calibration_points[i][1]
  A[j+1][11] = -1*calibration_points[i][1]

Atrans = np.transpose(A)
product = np.matmul(Atrans, A)
evals, evecs = np.linalg.eig(product)
minVal = np.argmin(evals)
projection = evecs[minVal]
projection = projection.reshape(3, 4)
# print(projection)

# test1 = np.array([[1,2,3,4],[5,6,7,8],[9,10,11,12]])
test = np.array([[0],[10],[0],[1]])
pixels = np.matmul(projection, test)
print(pixels[0]/pixels[2], pixels[1]/pixels[2])
# print(evals, minVal, evecs[minVal])
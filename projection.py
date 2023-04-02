import numpy as np
import cv2
from pupil_apriltags import Detector
import math
at_detector = Detector(
   families="tag36h11",
)
def getMod(point1, point2):
    return math.sqrt((point1[0]-point2[0])**2 + (point1[1]-point2[1])**2 + (point1[2]-point2[2])**2)

# img = cv2.imread('3d.jpg', cv2.IMREAD_COLOR)
# img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
frame_id=0
video = cv2.VideoCapture('vid.mov')
# out = cv2.VideoWriter('outputVid.mp4',cv2.VideoWriter_fourcc('m','p','4','v'),30.0, (1080,1920))
cubePoints = np.array([[0,0,0], [6.5,0,0], [6.5, 0, 6.5], [0, 0, 6.5],[0,6.5,0], [6.5,6.5,0], [6.5, 6.5, 6.5], [0, 6.5, 6.5]])
world = [[0,13.2,0], [13,13.2,0], [13,13.2,13.2], [0, 13.2, 13.2], [0,0,0], [13, 0, 0]]
while(frame_id<424):
    
    video.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
    ret, img = video.read()
    frame_id += 1
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    res = at_detector.detect(img_gray)
    if res[0].tag_id ==3:
        temp = res[0]
        res[0] = res[1]
        res[1] = temp
    for tag in res:
        tag.corners = np.rint(tag.corners).astype('int32')
        tag.center = np.rint(tag.center).astype('int32')
        # img = cv2.putText(img, str(tag.tag_id), int_center, cv2.FONT_HERSHEY_SIMPLEX,10, (0,0,255), 10 )
        # for corner in int_corners:
        #     img = cv2.circle(img, corner, 15, (0,0,255), -1)
    # cv2.imwrite('image.jpg', img)
    A = np.zeros((12, 11))
    Y = np.zeros((12, 1))

    # print(res)
    calibration_points = np.zeros((6,2), dtype = 'int32')
    for i in range(4):
        calibration_points[i][0] = res[0].corners[i][0]
        calibration_points[i][1] = res[0].corners[i][1]
    for i in range(2):
        calibration_points[i+4][0] = res[1].corners[i][0]
        calibration_points[i+4][1] = res[1].corners[i][1]

    # print(calibration_points)
    
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
        #   A[j][11] = -1*calibration_points[i][0]
        Y[j][0] = calibration_points[i][0]
        Y[j+1][0] = calibration_points[i][1]
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
        #   A[j+1][11] = -1*calibration_points[i][1]


    Atrans = np.transpose(A)
    pre = np.linalg.inv(np.matmul(Atrans, A))
    post = np.matmul(Atrans,Y)
    projection = np.matmul(pre,post)
    projection = np.append(projection, [[1]], 0)

    # print(res)
    # product = np.matmul(Atrans, A)
    # evals, evecs = np.linalg.eig(product)
    # minVal = np.argmin(evals)
    # projection = evecs[minVal]
    projection = projection.reshape(3, 4)
    # print(projection)

    # # test1 = np.array([[1,2,3,4],[5,6,7,8],[9,10,11,12]])
    # test = np.array([[0],[0],[0],[1]])
    # pixels = np.matmul(projection, test)
    # print(pixels[0]/pixels[2], pixels[1]/pixels[2])

    # for point in world:
    #     pixels = np.matmul(projection, np.array([[point[0]], [point[1]], [point[2]], [1]]))
    #     imageCoords = []
    #     imageCoords.append(pixels[0]/pixels[2])
    #     imageCoords.append(pixels[1]/pixels[2])
        
    #     print(imageCoords)
    # print(evals, minVal, evecs[minVal])

    imageCoordinates = []
    for point in cubePoints:
        pixels = np.matmul(projection, np.array([[point[0]], [point[1]], [point[2]], [1]]))
        
        imageCoords = []
        imageCoords.append(pixels[0][0]/pixels[2][0])
        imageCoords.append(pixels[1][0]/pixels[2][0])
        imageCoords = np.rint(imageCoords).astype('int32')
        imageCoordinates.append(imageCoords)
        # img = cv2.circle(img, imageCoords, 15, (0,0,255), -1)

    for i in range(8):
        for j in range(8):
            if getMod(cubePoints[i], cubePoints[j])==6.5:
                    img = cv2.line(img, imageCoordinates[i], imageCoordinates[j], (0,0,255), 5)

    # out.write(img)
    print(frame_id)
# out.release()
    cv2.imwrite('out/image'+str(frame_id)+'.jpg', img)
    # cv2.waitKey(0)
    # print("ok")
import numpy as np
import cv2
from pupil_apriltags import Detector
import math
detector = Detector(families="tag36h11",)
def getMod(point1, point2):
    return math.sqrt((point1[0]-point2[0])**2 + (point1[1]-point2[1])**2 + (point1[2]-point2[2])**2)

def getArucoCorners(frame_gray):
    
    res = detector.detect(frame_gray)
    if res[0].tag_id > res[1].tag_id:
        temp = res[0]
        res[0] = res[1]
        res[1] = temp
    for tag in res:
        tag.corners = np.rint(tag.corners).astype('int32')
        tag.center = np.rint(tag.center).astype('int32')

    points = np.zeros((6,2), dtype = 'int32')
    for i in range(4):
        points[i][0] = res[0].corners[i][0]
        points[i][1] = res[0].corners[i][1]
    for i in range(2):
        points[i+4][0] = res[1].corners[i][0]
        points[i+4][1] = res[1].corners[i][1]
    return points

def getMatrices(imageCoords, worldCoords):
    A = np.zeros((12, 11))
    Y = np.zeros((12, 1))
    for i in range(6):
        j=i*2
        A[j][0] = worldCoords[i][0]
        A[j][1] = worldCoords[i][1]
        A[j][2] = worldCoords[i][2]
        A[j][3] = 1
        A[j][4] = 0
        A[j][5] = 0
        A[j][6] = 0
        A[j][7] = 0
        A[j][8] = -1*worldCoords[i][0]*imageCoords[i][0]
        A[j][9] = -1*worldCoords[i][1]*imageCoords[i][0]
        A[j][10] = -1*worldCoords[i][2]*imageCoords[i][0]
        Y[j][0] = imageCoords[i][0]
        
        Y[j+1][0] = imageCoords[i][1]
        A[j+1][0] = 0
        A[j+1][1] = 0
        A[j+1][2] = 0
        A[j+1][3] = 0
        A[j+1][4] = worldCoords[i][0]
        A[j+1][5] = worldCoords[i][1]
        A[j+1][6] = worldCoords[i][2]
        A[j+1][7] = 1
        A[j+1][8] = -1*worldCoords[i][0]*imageCoords[i][1]
        A[j+1][9] = -1*worldCoords[i][1]*imageCoords[i][1]
        A[j+1][10] = -1*worldCoords[i][2]*imageCoords[i][1]
    return A, Y

def getProjectionMatrix(A, Y):
    Atrans = np.transpose(A)
    pre = np.linalg.inv(np.matmul(Atrans, A))
    post = np.matmul(Atrans,Y)
    projection = np.matmul(pre,post)
    projection = np.append(projection, [[1]], 0)
    projection = projection.reshape(3, 4)
    return projection

def makeCube(img, cubeWorldCoords, projection):
    imageCoordinates = []
    for point in cubeWorldCoords:
        pixels = np.matmul(projection, np.array([[point[0]], [point[1]], [point[2]], [1]]))
        
        imageCoords = []
        imageCoords.append(pixels[0][0]/pixels[2][0])
        imageCoords.append(pixels[1][0]/pixels[2][0])
        imageCoords = np.rint(imageCoords).astype('int32')
        imageCoordinates.append(imageCoords)

    for i in range(8):
        for j in range(8):
            if getMod(cubeWorldCoords[i], cubeWorldCoords[j])==6.5:
                img = cv2.line(img, imageCoordinates[i], imageCoordinates[j], (0,0,255), 5)
    return img

frame_id=0
video = cv2.VideoCapture('vid.mov')
frameCount = video.get(cv2.CAP_PROP_FRAME_COUNT)
cubePoints = np.array([[0,0,0], [6.5,0,0], [6.5, 0, 6.5], [0, 0, 6.5],[0,6.5,0], [6.5,6.5,0], [6.5, 6.5, 6.5], [0, 6.5, 6.5]])
world = np.array([[0,13.2,0], [13,13.2,0], [13,13.2,13.2], [0, 13.2, 13.2], [0,0,0], [13, 0, 0]])

while(frame_id<frameCount):

    video.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
    
    ret, img = video.read()
    frame_id += 1
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
   
    calibration_points = getArucoCorners(img_gray)
    A,Y = getMatrices(calibration_points, world)
    projection = getProjectionMatrix(A, Y)
    img = makeCube(img, cubePoints, projection)
    print(frame_id)
    cv2.imwrite('out/image'+str(frame_id)+'.jpg', img)
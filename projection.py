import numpy as np
import cv2
from pupil_apriltags import Detector
import math
import scipy
from scipy.ndimage import maximum_filter as maxf2D
from scipy.ndimage import convolve as cnv

detector = Detector(families="tag36h11",)

def calculateDerivatives(image, xaxis, yaxis): # calculate derivatives of image along given axes
    xDer = np.array([[-1, 0, 1]], np.float64)
    yDer = np.array([[1], [0], [-1]], np.float64)
    if xaxis:
        Ix = cnv(image, xDer, mode='constant')
    if yaxis:
        Iy = cnv(image, yDer, mode='constant')
    if xaxis and yaxis:
        return Ix,Iy
    elif xaxis:
        return Ix
    elif yaxis:
        return Iy
    
def calculateHessian(image, patch):  #Determine whether each pixel is a hessian point or not. Return a matrix containing 1s and 0s
    image = image.astype('int64')
    Ix, Iy = calculateDerivatives(image, True, True)
    Ixx, Ixy = calculateDerivatives(Ix, True, True)
    Iyy = calculateDerivatives(Iy, False, True)
    
    Ixx = cnv(Ixx, np.ones((patch,patch)), mode='constant')
    Iyy = cnv(Iyy, np.ones((patch,patch)), mode='constant')
    Ixy = cnv(Ixy, np.ones((patch,patch)), mode='constant')

    determinant = Ixx*Iyy - Ixy**2
    maximas = maxf2D(determinant, size=(27,27),mode='constant', cval=0.0)

    difference = determinant-maximas
    isMaxima = np.zeros_like(image)
    isMaxima[difference==0] = 1

    res = np.zeros_like(image)
    threshold = 0.25*np.max(maximas)
    res[maximas>threshold] = 1

    res = res*isMaxima
    res[0:10,:] = 0
    res[:,0:10] = 0
    res[image.shape[0]-10:image.shape[0],:] = 0
    res[:,image.shape[1]-10:image.shape[1]] = 0
    cnt = np.bincount(res.flatten())
    ret = np.zeros((cnt[1],2), dtype=np.int64)
    count = 0
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            if res[i][j] == 1:
                ret[count] = [i,j]
                count += 1
    return ret

def getSSD(image1, image2, pos1, pos2, patch):  #Calculate sum of squared differences for two points in consecutive frames
    x1min = pos1[0]-int(patch/2)
    y1min = pos1[1]-int(patch/2)
    x2min = pos2[0]-int(patch/2)
    y2min = pos2[1]-int(patch/2)
    
    ssd=0
    for i in range(patch):
        for j in range(patch):
            ssd += (image1[x1min+i][y1min+j]-image2[x2min+i][y2min+j])**2
    return ssd

def mapHessianPoints(image1, image2, hess1, hess2, patch): #Map hessian point in a frame to those in the next frame by minimising SSD.
    if image1 is None or image2 is None:
        return None
    image1 = image1.astype('int64')
    image2 = image2.astype('int64')
    padded_image1 = np.pad(image1, int(patch/2), 'constant', constant_values=0)
    padded_image2 = np.pad(image2, int(patch/2), 'constant', constant_values=0)
    mapping = np.zeros((len(hess1),5), dtype=np.int64)
    
    pointCount = 0

    for firstPoint in hess1:
        i = firstPoint[0]
        j = firstPoint[1]
                
        minSSD = 9223372036854775806
        minPos = (0,0)
        for secondPoint in hess2:
            m = secondPoint[0]
            n = secondPoint[1]
            ssd = getSSD(padded_image1,padded_image2,(i+int(patch/2),j+int(patch/2)),(m+int(patch/2),n+int(patch/2)),patch)
            if ssd<minSSD:
                minSSD = ssd
                minPos = (m,n)

        mapping[pointCount] = [minSSD, i, j, minPos[0], minPos[1]]
        pointCount += 1

    return mapping
    # for i in range(image1.shape[0]):
    #     for j in range(image1.shape[1]):
    #         if hess1[i][j] == 1:
    #             xmin = i-int(image1.shape[0]/8)
    #             xmax = i+int(image1.shape[0]/8)+1
    #             ymin = j-int(image1.shape[1]/8)
    #             ymax = j+int(image1.shape[1]/8)+1
    #             if xmin<0:
    #                 xmin = 0
    #             if ymin<0:
    #                 ymin = 0
    #             if xmax>=image2.shape[0]:
    #                 xmax = image2.shape[0]
    #             if ymax>=image2.shape[1]:  
    #                 ymax = image2.shape[1]
                
    #             minSSD = 9223372036854775806
    #             minPos = (0,0)
    #             for m in range(xmin,xmax):
    #                 for n in range(ymin,ymax):
    #                     if hess2[m][n] == 1:
    #                         ssd = getSSD(padded_image1,padded_image2,(i,j),(m,n),patch)
    #                         if ssd<minSSD:
    #                             minSSD = ssd
    #                             minPos = (m,n)
                
    #             mapping[pointCount] = [minSSD, i, j, minPos[0], minPos[1]]
    #             pointCount += 1

    # sortedMapping = mapping[mapping[:, 0].argsort()]
    # return sortedMapping

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

def makeCube(img, cubeWorldCoords, projectionMatrix):
    imageCoordinates = []
    for point in cubeWorldCoords:
        pixels = np.matmul(projectionMatrix, np.array([[point[0]], [point[1]], [point[2]], [1]]))
        
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

def decomposeProjection(projectionMatrix):
    squarePart = projectionMatrix[0:3, 0:3]
    columnPart = projectionMatrix[0:3, 3:4]
    intrinsic, rotation = scipy.linalg.rq(squarePart)
    translation = np.matmul(np.linalg.inv(intrinsic), columnPart)
    print("Ok")

def getLine(point1, point2):
    A = point1[1] - point2[1]
    B = point2[0] - point1[0]
    C = (point2[0]*point1[1]) - (point2[1]*point1[0])
    return np.array([A, B, C])

def getVanishingPoint(l1, l2):
    D = l1[0]*l2[1] - l1[1]*l2[0]
    Dx = l1[2]*l2[1] - l1[1]*l2[2]
    Dy = l1[0]*l2[2] - l1[2]*l2[0]
    if D==0:
        return None
    x = 1.0*Dx/D
    y = 1.0*Dy/D
    return np.array([[x], [y], [1]])

def getHomography(mapping):
    # homography = np.zeros((3,3))
    src_points = mapping[:, 1:3]
    dst_points = mapping[:, 3:5]
    A = np.zeros((8,8))
    B = np.zeros((8,1))
    for i in range(4):
        j=i*2
        A[j][0] = src_points[i][0]
        A[j][1] = src_points[i][1]
        A[j][2] = 1
        A[j][6] = -1*src_points[i][0]*dst_points[i][0]
        A[j][7] = -1*src_points[i][1]*dst_points[i][0]
        B[j][0] = dst_points[i][0]
        
        A[j+1][3] = src_points[i][0]
        A[j+1][4] = src_points[i][1]
        A[j+1][5] = 1
        A[j+1][6] = -1*src_points[i][0]*dst_points[i][1]
        A[j+1][7] = -1*src_points[i][1]*dst_points[i][1]
        B[j+1][0] = dst_points[i][1]
    homography = np.matmul(np.linalg.inv(A), B)
    homography = np.append(homography, [[1]], 0)
    homography = homography.reshape(3, 3)
    return homography

frame_id=0
video = cv2.VideoCapture('squareVideo.mp4')
frameCount = video.get(cv2.CAP_PROP_FRAME_COUNT)
cubePoints = np.array([[0,0,0], [6.5,0,0], [6.5, 0, 6.5], [0, 0, 6.5],[0,6.5,0], [6.5,6.5,0], [6.5, 6.5, 6.5], [0, 6.5, 6.5]])
world = np.array([[0,13,13.2], [13,13,13.2], [13,0,13.2], [0, 0, 13.2], [0,13.2,0], [13, 13.2, 0]])
prev_img_gray = None
prev_hessian = None

while(frame_id<frameCount):

    video.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
    frame_id += 1
    if frame_id!=1:
        prev_img_gray = img_gray
        prev_hessian = hessian

    ret, img = video.read()
    img = cv2.imread('image3.jpg', cv2.IMREAD_COLOR)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hessian = calculateHessian(img_gray, 7)

    # for point in hessian:
    #     img = cv2.circle(img, (point[1],point[0]), 25, (0,0,255), -1)

    if len(hessian)!=4:
        hessian = prev_hessian
        img_gray = prev_img_gray
        print("not 4")
        continue

    elif frame_id==1:
        continue
    
    else:
        mapping = mapHessianPoints(prev_img_gray, img_gray, prev_hessian, hessian, 27)
        
        for mapIndex in range(len(mapping)):
            map = mapping[mapIndex]
            if mapIndex==0:
                img = cv2.circle(img, (map[4],map[3]), 25, (0,0,255), -1)
            elif mapIndex==1:
                img = cv2.circle(img, (map[4],map[3]), 25, (0,255,0), -1)
            elif mapIndex==2:
                img = cv2.circle(img, (map[4],map[3]), 25, (255,0,0), -1)
            elif mapIndex==3:
                img = cv2.circle(img, (map[4],map[3]), 25, (0,255,255), -1)
        hessian = mapping[:, 3:5]
        homography = getHomography(mapping)
        
        

        # line1 = getLine(hessian[0,:], hessian[1,:])
        # line2 = getLine(hessian[2,:], hessian[3,:])
        # line3 = getLine(hessian[0,:], hessian[2,:])
        # line4 = getLine(hessian[1,:], hessian[3,:])
        # vx = getVanishingPoint(line1, line2)
        # vz = getVanishingPoint(line3, line4)
        # K = np.array([[1,2,3],[0,2,3], [0,0,1]])
        # r3 = np.matmul(np.linalg.inv(K), vz)
        # r3 = r3/np.linalg.norm(r3)
        # r1 = np.matmul(np.linalg.inv(K), vx)
        # r1 = r1/np.linalg.norm(r1)
        # r2 = np.cross(r3.flatten(), r1.flatten())
        # r2 = r2/np.linalg.norm(r2)
        # r2 = np.reshape(r2, (3,1))
        # R = np.concatenate((r1, r2, r3), axis=1)

    # calibration_points = getArucoCorners(img_gray)
    # A,Y = getMatrices(calibration_points, world)
    # projection = getProjectionMatrix(A, Y)
    # decomposeProjection(projection)
    # img = makeCube(img, cubePoints, projection)
    print(frame_id)
    cv2.imwrite('out/image'+str(frame_id)+'.jpg', img)
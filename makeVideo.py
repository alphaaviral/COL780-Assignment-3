import cv2
import os
import numpy as np
def fileNames(loc):
    return os.listdir(loc)
def getFrame(dirLoc,index):
    # files=fileNames(dirLoc)
    frame=cv2.imread(dirLoc+str(index)+'.jpg', cv2.IMREAD_COLOR)
    return frame
dirLoc="./out/image"
resLoc = "./outputnew"
frame=getFrame(dirLoc,1)
result = cv2.VideoWriter(resLoc+'.mp4',cv2.VideoWriter_fourcc('m','p','4','v'),30.0, (frame.shape[1],frame.shape[0]))
counter=1
finalFrame=len(fileNames("./out"))
while(counter<finalFrame):
    frame = getFrame(dirLoc,counter)
    result.write(frame)
    counter+=1
result.release()
import numpy as np
import cv2
from application import get_results

print("IN CV_TEST")

vidcap = cv2.VideoCapture('test_videos/video2.mp4')

print("CAP OPEN")

sec = 0
frameRate = 1 #it will capture image in each 1 second
results = []

def getFrame(sec):
	print("AT FRAME " + str(sec))	
	vidcap.set(cv2.CAP_PROP_POS_MSEC,sec*1000)
	hasFrames,image = vidcap.read()
	if hasFrames:
		width, height = image.shape[:2]
		info = get_results(image, width, height)
		print(info)
		results.append(info)
	return hasFrames


success = getFrame(sec)
while success:
	sec = sec + frameRate
	success = getFrame(sec)
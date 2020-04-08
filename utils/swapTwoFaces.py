import sys
# sys.path.remove(sys.path[1])
import matplotlib.pyplot as plt
import cv2
import time
import numpy as np
import dlib
from imutils import face_utils
from scipy import interpolate
import math
from scipy import interpolate
from utils.tps import *

def swapVideo(videoPath, saveFlag = True, savePath = "../Data/Test2_out.avi"):
	cap = cv2.VideoCapture(videoPath)
	firstTime = True
	while (cap.isOpened()):
		ret, image = cap.read()
		if ret:
			if saveFlag and firstTime:
				fourcc = cv2.VideoWriter_fourcc(*'MJPG')
				vw = cv2.VideoWriter(savePath, fourcc, 30, (image.shape[1], image.shape[0]))
				firstTime = False
			gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
			target_points, source_points, cent1, cent2, detectedFaces = giveTwoface(gray)
			if detectedFaces:
				newImage = swap(image, image, gray, gray, 
							target_points, cent1, source_points, cent2,
							debug = False, showW = 240, showH = 220)
				newGray = cv2.cvtColor(newImage, cv2.COLOR_BGR2GRAY)
				newImage1 = swap(image, newImage, gray, newGray, 
							source_points, cent2, target_points, cent1,
							debug = False, showW = 240, showH = 220)
				# cv2.imshow("Swapped", newImage)
				cv2.imshow("Oppo", newImage1)
				if saveFlag:
					vw.write(newImage1)
			else:
				cv2.imshow("Oppo", image)
				if saveFlag:
					vw.write(image)
			if cv2.waitKey(1) & 0xFF == ord('q'):
				break
		else:
			break
	print("Done")
	cap.release()
	cv2.destroyAllWindows()
	if saveFlag:
		vw.release()
	pass

def main():
	videoPath = "../Data/Test2.mp4"
	swapVideo(videoPath, saveFlag = True)

if __name__ == '__main__':
	main()
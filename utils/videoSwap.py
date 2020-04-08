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

def swapVideo(videoPath, sourceImagePath, saveFlag = True, savePath = "../Data/outputVideo.avi"):
	cap = cv2.VideoCapture(videoPath)
	fourcc = cv2.VideoWriter_fourcc(*'MJPG')
	# ret, trg = cap.read()
	
	targetImage = cv2.imread(sourceImagePath)
	targetImage_gray = cv2.imread(sourceImagePath, 0)
	source_points, source_cent, sourceDetected = giveFeaturePoints(targetImage_gray)
	frameCount = 1
	firstTime = True
	if sourceDetected:
		while (cap.isOpened()):
			ret, sourceImage = cap.read()
			print(frameCount)
			frameCount +=1
			if ret:
				if saveFlag and firstTime:
					vw = cv2.VideoWriter(savePath, fourcc, 30, (sourceImage.shape[1], sourceImage.shape[0]))
					firstTime = False
				sourceImage_gray = cv2.cvtColor(sourceImage, cv2.COLOR_BGR2GRAY)
				target_points, target_cent, targetDetected = giveFeaturePoints(sourceImage_gray)
				if targetDetected:
					newImage = swap(targetImage, sourceImage, targetImage_gray, sourceImage_gray, 
								source_points, source_cent, target_points, target_cent, 
								debug = False, showW = 240, showH = 220)
					cv2.imshow("Swapped", newImage)
					if saveFlag:
						vw.write(newImage)
				else:
					cv2.imshow("Swapped", sourceImage)
					if saveFlag:
						vw.write(sourceImage)
					print("Frame dropped")
				
				if cv2.waitKey(1) & 0xFF == ord('q'):
					break
			else:
				break
	else:
		print(" No faces detected")
	if saveFlag:
		vw.release()
	cap.release()
	cv2.destroyAllWindows()
	print("Done")
	pass


def main():
	videoPath = "../Data/Test3.mp4"
	sourceImagePath = "../Data/Scarlett.jpg"
	swapVideo(videoPath, sourceImagePath, saveFlag = True)

if __name__ == '__main__':
	main()
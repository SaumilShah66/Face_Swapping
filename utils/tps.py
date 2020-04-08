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

p = "Data/shape_predictor_68_face_landmarks.dat"
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(p)

# names = ["Data/harsh.jpeg", "Data/photo.jpg", "../Data/H.jpeg", "../Data/U.jpeg"]

# sourceImageName = names[1]
# targetImageName = names[0]

# sourceImage = cv2.imread(targetImageName)
# sourceImage_gray = cv2.imread(targetImageName,0)

# targetImage = cv2.imread(sourceImageName)
# targetImage_gray = cv2.imread(sourceImageName,0)

def giveFeaturePoints(gray):
	rects = detector(gray, 0)
	detectedFaces = False
	shape = None
	cent = None
	if rects:
		sx, sy = rects[0].left(), rects[0].top()
		ex,ey = sx+rects[0].width(), sy+rects[0].height()
		patch = gray[sy:ey, sx:ex]
		cent = [sx + rects[0].width()//2, sy + rects[0].height()//2]
		rect = rects[0]
		shape = predictor(gray, rect)
		shape = face_utils.shape_to_np(shape)
		detectedFaces = True
	return shape, cent, detectedFaces

def giveTwoface(gray):
	rects = detector(gray, 1)
	print("Number of faces - " + str(len(rects)))
	shape1, shape2 = None, None
	cent1, cent2 = None, None
	detectedFaces = False
	if len(rects)==2:
		rect1 = rects[0]
		sx, sy = rects[0].left(), rects[0].top()
		cent1 = [sx + rects[0].width()//2, sy + rects[0].height()//2]
		shape1 = predictor(gray, rect1)
		shape1 = face_utils.shape_to_np(shape1)
		rect2 = rects[1]
		sx, sy = rects[1].left(), rects[1].top()
		cent2 = [sx + rects[1].width()//2, sy + rects[1].height()//2]
		shape2 = predictor(gray, rect2)
		shape2 = face_utils.shape_to_np(shape2)
		detectedFaces = True
	return shape1, shape2, cent1, cent2, detectedFaces 

def computeK(x, y, source):    
	sx, sy = source[:,0], source[:,1] 
	kx_,ky_ = np.tile(x, (source.shape[0],1)).T, np.tile(y,(source.shape[0],1)).T
	tmp = (kx_ - sx)**2 + (ky_ - sy)**2 + sys.float_info.epsilon**2
	K = tmp*np.log(tmp)    
	return K

def NewTps(source, target, lam):
	P = np.append(source,np.ones([source.shape[0],1]),axis=1)
	P_Trans = P.T
	Z = np.zeros([3,3])
	K = computeK(source[:,0], source[:,1], source)
	M = np.vstack([np.hstack([K,P]),np.hstack([P_Trans,Z])])
	I = np.identity(M.shape[0])
	L = M+lam*I
	L_inv = np.linalg.inv(L)
	V = np.concatenate([target,np.zeros([3,2])])
	weights = np.matmul(L_inv,V)
	return weights

def mask_from_points(size, points,erode_flag=0):
	radius = 10  # kernel size
	kernel = np.ones((radius, radius), np.uint8)
	mask = np.zeros(size, np.uint8)
	print(points.dtype)
	cv2.fillConvexPoly(mask, cv2.convexHull(points), 255)
	# if erode_flag:
	# 	mask = cv2.erode(mask, kernel,iterations=1)
	return mask

def NewWarp(img_target, img_src, points1, points2, weights, mask2):
	xy1_min = np.float32([min(points1[:,0]),min(points1[:,1])])
	xy1_max = np.float32([max(points1[:,0]),max(points1[:,1])])
	xy2_min = np.float32([min(points2[:,0]),min(points2[:,1])])
	xy2_max = np.float32([max(points2[:,0]),max(points2[:,1])])
	x = np.arange(xy1_min[0],xy1_max[0]).astype(int)
	y = np.arange(xy1_min[1],xy1_max[1]).astype(int)
	# print(x[0],x[-1]+1)
	# print(y[0],y[-1]+1)
	X,Y = np.mgrid[x[0]:x[-1]+1,y[0]:y[-1]+1]
	w,h = X.shape
	# print(X.shape)
	X,Y = X.ravel(), Y.ravel()

	pts_src = np.hstack((X.reshape([-1,1]),Y.reshape([-1,1]))) 
	P = np.append(pts_src,np.ones([pts_src.shape[0],1]),axis=1)
	Z = np.zeros([3,3])
	K = computeK(X,Y, points1)
	M = np.hstack([K,P])
	vv = M.dot(weights).astype(np.int64)
	
	# outImg = np.zeros([img_target.shape[1], img_target.shape[0], 3])
	outImg = img_target.copy()*0
	# outImg1 = img_target.copy()*0
	map_x = (vv[:,0]).reshape([w,h]).astype(np.float32)
	map_y = (vv[:,1]).reshape([w,h]).astype(np.float32)
	dst = cv2.remap(img_src, map_x, map_y, cv2.INTER_LINEAR)
	dst = cv2.flip(np.rot90(dst, k=3), 1)
	h,w,_ = dst.shape
	# cv2.ROTATE_90_COUNTERCLOCKWISE(dst)
	outImg[y[0]:y[0]+h, x[0]:x[0]+w , :] = dst
	# plt.imshow(dst)
	# plt.show()
	# cv2.imwrite("remap.png",outImg1)
	# print(len(X))
	# print(len(vv))
	# for i in range(len(X)):
	# 	outImg[Y[i],X[i],:] = img_src[int(round(vv[i,1])), int(round(vv[i,0])),:]
	# plt.imshow(outImg)
	# plt.show()
	# cv2.imwrite("manual.png", outImg)
	return outImg

def swap(targetImage, sourceImage, targetImage_gray, sourceImage_gray, 
	source_points, source_cent, target_points, target_cent, 
	debug = False, showW = 160, showH = 120):
	w, h = targetImage_gray.shape
	mask_target_ = mask_from_points((w, h), source_points)
	w, h = sourceImage_gray.shape
	mask_source_ = mask_from_points((w, h), target_points)
	weights = NewTps(target_points, source_points, 1e-8)
	
	tt = NewWarp(sourceImage, targetImage, target_points, source_points,weights, mask_source_)
	newWarped = cv2.seamlessClone(tt, sourceImage, mask_source_, tuple(target_cent) ,cv2.NORMAL_CLONE)
	if debug:
		mask_target = np.where(mask_target_!=0, targetImage_gray, 0)
		mask_source = np.where(mask_source_!=0, sourceImage_gray, 0)
		cv2.namedWindow("Warped", cv2.WINDOW_NORMAL)
		cv2.namedWindow("Source", cv2.WINDOW_NORMAL)
		cv2.namedWindow("Mask", cv2.WINDOW_NORMAL)		
		cv2.namedWindow("Final", cv2.WINDOW_NORMAL)
		cv2.namedWindow("Target mask", cv2.WINDOW_NORMAL)
		cv2.namedWindow("Source mask", cv2.WINDOW_NORMAL)

		cv2.resizeWindow("Warped", showW, showH)
		cv2.resizeWindow("Source", showW, showH)
		cv2.resizeWindow("Mask", showW, showH)
		cv2.resizeWindow("Final", showW, showH)
		cv2.resizeWindow("Target mask", showW, showH)
		cv2.resizeWindow("Source mask", showW, showH)
		
		cv2.imwrite("Warped.jpg", tt)
		cv2.imwrite("Source.jpg", sourceImage)
		cv2.imwrite("Mask.jpg", mask_source_)
		cv2.imwrite("Final.jpg", newWarped)
		cv2.imwrite("Target_mask.jpg",mask_target)
		cv2.imwrite("Source_mask.jpg",mask_source)

		cv2.imshow("Warped", tt)
		cv2.imshow("Source", sourceImage)
		cv2.imshow("Mask", mask_source_)
		cv2.imshow("Final", newWarped)
		cv2.imshow("Target mask",mask_target)
		cv2.imshow("Source mask",mask_source)
		cv2.waitKey(0)
		cv2.destroyAllWindows()
	return newWarped

if __name__ == '__main__':
	start = time.time()
	source_points, source_cent, _ = giveFeaturePoints(targetImage_gray)
	target_points, target_cent, _ = giveFeaturePoints(sourceImage_gray)
	newImage = swap(targetImage, sourceImage, targetImage_gray, sourceImage_gray, 
		source_points, source_cent, target_points, target_cent, 
		debug = True, showW = 240, showH = 220)
	print(time.time()-start)
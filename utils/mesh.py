import sys
# sys.path.remove(sys.path[1])
import matplotlib.pyplot as plt
import cv2
import numpy as np
import dlib
from imutils import face_utils
from scipy import interpolate

def detectedFaces(img):
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	imgCopy = img.copy()
	p = "Data/shape_predictor_68_face_landmarks.dat"
	detector = dlib.get_frontal_face_detector()
	predictor = dlib.shape_predictor(p)
	faces = detector(gray, 0)
	if(len(faces) == 0):
		faces = detector(gray, 1)
	allFaceMarkers = []
	allFaceImgs = []
	count = 0
	for face in faces:
		count += 1
#         (x, y, w, h) = face_utils.rect_to_bb(face)
		markers = predictor(gray, face)
		markers = face_utils.shape_to_np(markers)
		convexHull= cv2.convexHull(markers)
		mask = np.zeros_like(gray)
		cv2.fillConvexPoly(mask, convexHull, 255)
		faceImg = cv2.bitwise_and(img, img, mask=mask)
		allFaceMarkers.append(markers)
		allFaceImgs.append(faceImg)
		
#         newPts= np.array([[x,y],[x+w,y],[x,y+h],[x+w,y+h]])
#     allFaceMarkers = np.concatenate((markers,newPts), axis=0)
#     print (markers.shape)
#     (x, y, w, h) = face_utils.rect_to_bb(rect)
	return count, np.array(allFaceMarkers), np.array(allFaceImgs)

def delaunay(src, src_pts, trg, trg_pts, name):
#     srcCpy = src.copy()
#     trgCpy = trg.copy()
	trg_swap = trg.copy()
	srcGray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
	trgGray = cv2.cvtColor(trg, cv2.COLOR_BGR2GRAY)
	th, tw = srcGray.shape[0], srcGray.shape[1]
	srcHull = cv2.convexHull(src_pts)
	srcBox = cv2.boundingRect(srcHull)
	(x,y,w,h) = srcBox
#     cv2.rectangle(srcCpy, (x, y), (x + w, y + h), (0, 255, 0), 2)  
	subdiv = cv2.Subdiv2D(srcBox)
	for p in src_pts:
		subdiv.insert(tuple(p))
	triangles = subdiv.getTriangleList()
	triangles = np.array(triangles, dtype=np.int32)
	delaunaySrc = []
	delaunayTrg = []
	c = 0
	x = x - 10
	y = y - 10
	w = w + 20
	h = h + 20
	for t in triangles:
		if(t[0]>=x and t[0]<=x+w and t[1]>=y and t[1]<=y+h and t[2]>=x and t[2]<=x+w and t[3]>=y and t[3]<=y+h and t[4]>=x and t[4]<=x+w and t[5]>=y and t[5]<=y+h):
#         if(t[0]>=0 and t[0]<=tw and t[1]>=0 and t[1]<=th and t[2]>=0 and t[2]<=tw and t[3]>=0 and t[3]<=th and t[4]>=0 and t[4]<=tw and t[5]>=0 and t[5]<=th):
			c+=1
			pt1 = (t[0], t[1])
			pt2 = (t[2], t[3])
			pt3 = (t[4], t[5])
			srcTri = np.array([pt1, pt2, pt3])
			rect1 = cv2.boundingRect(srcTri)
			(x1,y1,w1,h1) = rect1
			#cv2.rectangle(srcCpy, (x1, y1), (x1 + w1, y1 + h1), (0, 255, 0), 2)                    
			src_arr = []
			
			for yy in range(y1,y1+h1+1):
				temp = np.linspace((x1,yy,1),(x1+w1,yy,1), w1+1)
				src_arr.append(temp)
			
			src_arr = np.array(src_arr, np.int32)
			src_arr = (src_arr.flatten()).reshape(-1,3)
			src_arr = np.transpose(src_arr)
#             delaunaySrc.append(srcTri)
			
#             index.append([idx1,idx2,idx3])
#             cv2.line(srcCpy, pt1, pt2, (0, 255, 0), 1)
#             cv2.line(srcCpy, pt2, pt3, (0, 255, 0), 1)
#             cv2.line(srcCpy, pt1, pt3, (0, 255, 0), 1)
			idx1 = np.argwhere((src_pts == pt1).all(axis=1))[0][0]
			idx2 = np.argwhere((src_pts == pt2).all(axis=1))[0][0]
			idx3 = np.argwhere((src_pts == pt3).all(axis=1))[0][0]
			
			trg_pt1 = tuple(trg_pts[idx1])
			trg_pt2 = tuple(trg_pts[idx2])
			trg_pt3 = tuple(trg_pts[idx3])
			trgTri = np.array([trg_pt1, trg_pt2, trg_pt3], dtype= np.int32)
			rect2 = cv2.boundingRect(trgTri)
			(x2,y2,w2,h2) = rect2
#             cv2.rectangle(trgCpy, (x2, y2), (x2 + w2, y2 + h2), (0, 255, 0), 2)
			trg_arr = []
			for yy in range(y2,y2+h2+1):
				temp = np.linspace((x2,yy,1),(x2+w2,yy,1), w2+1)
				trg_arr.append(temp)
			trg_arr = np.array(trg_arr, np.int32)
			trg_arr = (trg_arr.flatten()).reshape(-1,3)
			trg_arr = np.transpose(trg_arr)
			
#             delaunayTrg.append(trgTri)
#             cv2.line(trgCpy, trg_pt1, trg_pt2, (0, 255, 0), 1)
#             cv2.line(trgCpy, trg_pt2, trg_pt3, (0, 255, 0), 1)
#             cv2.line(trgCpy, trg_pt1, trg_pt3, (0, 255, 0), 1)
			
			src_match, trg_cord, status = bary(src_arr, trg_arr, srcTri, trgTri, trg, src)
			
			if(status):
				#image interpolation
				X = np.arange(0, src.shape[1])
				Y = np.arange(0, src.shape[0])
	#             ZG = np.transpose(srcGray)
				ZB = (src[:,:,0])
				ZG = (src[:,:,1])
				ZR = (src[:,:,2])
	#             ZB = np.transpose(src[:,:,0])
	#             ZG = np.transpose(src[:,:,1])
	#             ZR = np.transpose(src[:,:,2])
				fb = interpolate.interp2d(X, Y, ZB, kind='cubic', fill_value=0)
				fg = interpolate.interp2d(X, Y, ZG, kind='cubic', fill_value=0)
				fr = interpolate.interp2d(X, Y, ZR, kind='cubic', fill_value=0)
	#             fg = interpolate.interp2d(X, Y, ZG, kind='cubic')

				for i in range(len(trg_cord)):
	#                 g = fg(src_match[0,i], src_match[1,i])
					blue = fb(src_match[0,i], src_match[1,i])
					red = fr(src_match[0,i], src_match[1,i])
					green = fg(src_match[0,i], src_match[1,i])
					w_, h_ = trg_arr[0,trg_cord[i]], trg_arr[1,trg_cord[i]]
#                     print (src_match[0,i], src_match[1,i],blue)
	#                 trgGray[w_,h_] = g
					trg_swap[h_,w_,0] = blue
					trg_swap[h_,w_,1] = green
					trg_swap[h_,w_,2] = red
#             input()
#     plt.imshow(srcCpy)
#     print c
#     cv2.imwrite(name+"_srcl.jpg",srcCpy)
#     cv2.imwrite(name+"_trg.jpg",trgCpy)
#     cv2.imwrite('swap.jpeg', trg_swap)
	
	trgHull = cv2.convexHull(trg_pts)
	trgBox = cv2.boundingRect(trgHull)
	
	return trg_swap, trgBox

def bary(src, trg, srcTri, trgTri, trgImg, srcImg):
	mask = np.zeros_like(trgImg)
	mask2 = np.zeros_like(srcImg)
#     cv2.fillConvexPoly(mask, convexHull, 255)
#     faceImg = cv2.bitwise_and(img, img, mask=mask)
#     print (trg)
#     print (trgTri)
	bary_trg = np.array([[trgTri[0][0], trgTri[1][0], trgTri[2][0]],
						 [trgTri[0][1], trgTri[1][1], trgTri[2][1]],
						 [1, 1, 1]])
	bary_src = np.array([[srcTri[0][0], srcTri[1][0], srcTri[2][0]],
						 [srcTri[0][1], srcTri[1][1], srcTri[2][1]],
						 [1, 1, 1]])
#     binv_src = np.linalg.inv(bary_src)
	try:
		binv_trg = np.linalg.inv(bary_trg)
		trg_bcord = np.matmul(binv_trg, trg)
	#     trg_T = np.transpose(trg_bcord)
		inliers = []
		coord = []
		for i in range (trg_bcord.shape[1]):
			a, b, y = trg_bcord[0,i], trg_bcord[1,i], trg_bcord[2,i]
			if(a>=0 and a<=1 and b>=0 and b<=1 and y>=0 and y<=1 and (a+b+y)>0):
				inliers.append([a,b,y])
				coord.append(int(i))
				mask[trg[1,i], trg[0,i]]= trgImg[trg[1,i], trg[0,i]]
	#     cv2.imwrite('mask_trg.jpg', mask)
		inliers = np.array(inliers)
		inliers = (inliers.flatten()).reshape(-1,3)
		inliers = np.transpose(inliers)
		src_match = np.matmul(bary_src, inliers, dtype= np.float64)
#         if(src_match[2] ==0):
#             src_match[2] = 0.0000001
		src_match = src_match/src_match[2]
		return src_match, coord, True
	
	except:
		return [],[], False


def triangles(src, src_pts, name):
	srcCpy = src.copy()
	srcGray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
	th, tw = srcGray.shape[0], srcGray.shape[1]
	srcHull = cv2.convexHull(src_pts)
	srcBox = cv2.boundingRect(srcHull)
	(x,y,w,h) = srcBox
	cv2.rectangle(srcCpy, (x, y), (x + w, y + h), (0, 255, 0), 2)  
	subdiv = cv2.Subdiv2D(srcBox)
	for p in src_pts:
		subdiv.insert(tuple(p))
	triangles = subdiv.getTriangleList()
	triangles = np.array(triangles, dtype = np.int32)
	delaunaySrc = []
	c = 0
	x = x - 10
	y = y - 10
	w = w + 20
	h = h + 20
	for t in triangles:
		if(t[0]>=x and t[0]<=x+w and t[1]>=y and t[1]<=y+h and t[2]>=x and t[2]<=x+w and t[3]>=y and t[3]<=y+h and t[4]>=x and t[4]<=x+w and t[5]>=y and t[5]<=y+h):
#         if(t[0]>=0 and t[0]<=tw and t[1]>=0 and t[1]<=th and t[2]>=0 and t[2]<=tw and t[3]>=0 and t[3]<=th and t[4]>=0 and t[4]<=tw and t[5]>=0 and t[5]<=th):
			c += 1
			pt1 = (t[0], t[1])
			pt2 = (t[2], t[3])
			pt3 = (t[4], t[5])
			srcTri = np.array([pt1, pt2, pt3])
			#cv2.rectangle(srcCpy, (x1, y1), (x1 + w1, y1 + h1), (0, 255, 0), 2)                    
			delaunaySrc.append(srcTri)
#             print ('src ',pt1, pt2, pt3)
			cv2.line(srcCpy, pt1, pt2, (0, 255, 0), 1)
			cv2.line(srcCpy, pt2, pt3, (0, 255, 0), 1)
			cv2.line(srcCpy, pt1, pt3, (0, 255, 0), 1)
	cv2.imwrite(name+"_Tri.jpg", srcCpy)
	return

def ColorBlend(dst, src, srcPts):
	srcHull = cv2.convexHull(srcPts)
	srcBox = cv2.boundingRect(srcHull)
	(x,y,w,h) = srcBox
#     src_mask = np.full(src.shape, 255, dtype = np.uint8)
	src_mask = np.zeros_like(src)
#     poly = np.array([ [x,y], [x+w,y], [x,y+h], [x+w,y+h] ], np.int32)
	
#     print src_mask.shape
	cv2.fillConvexPoly(src_mask, srcHull, (255, 255, 255))
	
#     center = ((src.shape[1])/2,(src.shape[0])/2)
	center = (int((x+x+w)/2),int((y+y+h)/2))
#     src_mask[int((y+h)/2), int((x+w)/2)] = [255,0,0]
#     kernel = np.ones((5, 5), np.uint8) 
  
	# Using cv2.erode() method  
#     src_mask = cv2.erode(src_mask, kernel)
	
	# plt.imshow(src_mask)

#     print (x,y,w,h,center)
	# Clone seamlessly.
	imgNew = cv2.seamlessClone(src, dst, src_mask, center, cv2.NORMAL_CLONE)
	return imgNew

def meshSwap(videoPath, sourceImagePath, saveFlag = False):
	print("Came for swap")
	# plt.ion()
	src = cv2.imread(sourceImagePath)
	srcOrg = src.copy()
	cap = cv2.VideoCapture(videoPath)
	ret, trg= cap.read()
	if saveFlag:
		fourcc = cv2.VideoWriter_fourcc(*'MJPG')
		vw = cv2.VideoWriter("Rambo_video.avi", fourcc, 30, (trg.shape[1], trg.shape[0]))
	src_face_count, srcAllMarkers, srcFaceImgs = detectedFaces(src)
	frameCount = 1
	while (cap.isOpened()):
		ret, trg = cap.read()
		trgOrg = trg.copy()
		frameCount += 1
		print(frameCount)
		#     src_face_count, srcAllMarkers, srcFaceImgs = detectedFaces(src)
		trg_face_count, trgAllMarkers, trgFaceImgs = detectedFaces(trg)
		if(len(trgAllMarkers)):
			swap, box = delaunay(src, srcAllMarkers[0], trg, trgAllMarkers[0], '1')
			swapClr = ColorBlend(trg, swap, trgAllMarkers)
			cv2.imshow("Img", swapClr)
			if cv2.waitKey(1) & 0xFF == ord('q'):
				break
			if saveFlag:
				vw.write(swapClr)
		if saveFlag:
			vw.release()
	cap.release()

def swapTwoFaces(videoPath, saveFlag = False, savePath="outputVid.avi"):
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
			trg_face_count, trgAllMarkers, trgFaceImgs = detectedFaces(image)
			if trg_face_count>=2:
				print("Got two faces")
				swap1, box1 = delaunay(image.copy(), trgAllMarkers[0], image.copy(), trgAllMarkers[1], '1')
				swapClr1 = ColorBlend(image, swap1, trgAllMarkers[1])
				swap2, box2 = delaunay(image.copy(), trgAllMarkers[1], image.copy(), trgAllMarkers[0], '1')
				swapClr2 = ColorBlend(swapClr1, swap2, trgAllMarkers[0])
				cv2.imwrite("Swap1.jpg",swapClr1)
				cv2.imwrite("Swap2.jpg",swapClr2)
				# cv2.imshow("1", swapClr2)
				# if cv2.waitKey(1) & 0xFF == ord('q'):
				# 	break
				if saveFlag:
					vw.write(swapClr2)
			else:
				print("Frame drop")
				if saveFlag:
					vw.write(image)
	if saveFlag:
		vw.release()
	cap.release()
	pass

def main():
	videoPath = "../Data/Test1.mp4"
	sourceImagePath = "../Data/Rambo.jpg"
	meshSwap(videoPath, sourceImagePath)
	
if __name__ == '__main__':
	main()

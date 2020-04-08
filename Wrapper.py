from utils import videoSwap, mesh, swapTwoFaces
import argparse


Parser = argparse.ArgumentParser()
Parser.add_argument('--swapTwoFace', default=0, help="When you want to swap two faces in one video")
Parser.add_argument('--swapOneFace', default=1, help="When you want to swap one face in video with an image")
Parser.add_argument('--Method', default="TPS", help="Which method for warping")
Parser.add_argument('--imagePath', default="Data/Rambo.jpg", help="Path to the image whose face you want to use as source")
Parser.add_argument('--videoPath', default="Data/Test1.mp4", help="Path to the target video file")
Parser.add_argument('--saveVideo', default=1, help="True if you want to save video")
Args = Parser.parse_args()


if Args.Method=="TPS":
	if Args.swapTwoFace=='1':
		swapTwoFaces.swapVideo(Args.videoPath, saveFlag = int(Args.saveVideo),
		 savePath = "Data/outputVideo.avi")
	else:
		pass
	if Args.swapOneFace=='1':
		videoSwap.swapVideo(Args.videoPath, Args.imagePath,
			saveFlag=False, savePath="Data/outputVideo.avi")
	else:
		pass
else:
	if Args.swapTwoFace=='1':
		mesh.swapTwoFaces(Args.videoPath, saveFlag = int(Args.saveVideo),
		 savePath = "Data/outputVideo.avi")
	else:
		pass
	if Args.swapOneFace=='1':
		mesh.meshSwap(Args.videoPath, Args.imagePath)	
	else:
		pass

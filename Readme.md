
Please make sure that shape_predictor_68_face_landmarks.dat file is available in the Data folder

To run test 1 

with thin plate spline
```
python3 --swapOneFace=1 --Method="TPS" --imagePath="Data/Rambo.jpg" --videoPath="Data/Test1.mp4" 
```

with thin Delaunay Triangulation
```
python3 --swapOneFace=1 --Method="Delau" --imagePath="Data/Rambo.jpg" --videoPath="Data/Test1.mp4" 
```

To run test 2

with thin plate spline
```
python3 --swapTwoFace=1 --swapOneFace=0 --Method="TPS" --videoPath="Data/Test2.mp4" 
```

with thin Delaunay Triangulation
```
python3 --swapTwoFace=1 --swapOneFace=0 --Method="Delau" --videoPath="Data/Test2.mp4" 
```

To run test 3

with thin plate spline
```
python3 --swapOneFace=1 --Method="TPS" --imagePath="Data/Scarlett.jpg" --videoPath="Data/Test3.mp4" 
```

with thin Delaunay Triangulation
```
python3 --swapOneFace=1 --Method="Delau" --imagePath="Data/Scarlett.jpg" --videoPath="Data/Test3.mp4" 
```


To run the PRNet deep learning model, download the repository provided in the link on the instructions page of the project along with the data from google drive.

Copy the files provided in this submission in the directory where PRNet (after copying the repository)
1) demo_copy.py
2) tpsDeepLearning.py
3) api.py

To run the code use the below command

Edit the below parameter at line 346 and 349
src = cv2.imread("TestImages/Scarlett.jpg")
path = "a.mp4" for video file path
To swap 2 faces in a video set --swap_video= True for the video at the path provided in above parameter.

```
python2 demo_copy.py --isKpt=True --swap_video=True
```

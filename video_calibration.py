import cv2
import os
import numpy as np

rtsp_link =''
output_dir =''
os.makedirs(output_dir, exist_ok=True)

video_capture = cv2.VideoCapture(rtsp_link) # capture the video from rtsp_link

if not video_capture.isOpened():         #check if video is captured
    print("Error could not open video stream")
    exit()
    
frame_count = 0                          #count no. of frames 
while True:
    ret, frame = video_capture.read()
    if not ret:
        break
    
    
    frame_path = os.path.join(output_dir,f'frame_{frame_count}.jpg')  #provide path where frame is to stored
    cv2.imwrite(frame_path,frame)                                     #store images
    
    frame_count+=1                                 
     
    if frame_count >=100:                                             #stop after reading 100 frames
        break
    
video_capture.release()

CHECKERBOARD = (5,7)
subpix_criteria = (cv2.TermCriteria_MAX_ITER + cv2.TERM_CRITERIA_EPS,30,0.01)
calibration_flags = cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC+cv2.fisheye.CALIB_CHECK_COND+cv2.fisheye.CALIB_FIX_SKEW

objp = np.zeros((1,CHECKERBOARD[0]*CHECKERBOARD[1],3), np.float32)
objp[0,:,:2] = np.mgrid[0:CHECKERBOARD[0],0:CHECKERBOARD[1]].T.reshape(-1,2)

img_shape = None
objpoints = []
imgpoints = []

for frame in frame_path:
    img = cv2.imread(frame)
    
    if _img_shape == None:
        _img_shape = img.shape[:2]
    else:
        assert _img_shape == img.shape[:2]
        
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret,corners = cv2.findChessboardCorners(gray,CHECKERBOARD,cv2.CALIB_CB_ADAPTIVE_THRESH+cv2.CALIB_CB_FAST_CHECK+cv2.CALIB_CB_NORMALIZE_IMAGE)
    
    if ret == True:
        objpoints.append(objp)
        corner2 = cv2.cornerSubPix(gray, corners,(3,3),(-1,-1),subpix_criteria)
        imgpoints.append(corner2)
        
N_OK = len(objpoints)
camMatrix = np.zeros((3,3))
coeffs = np.zeros((4,1))
rvecs = [np.zeros((1,1,3), dtype=np.float64) for i in range(N_OK)]
tvecs = [np.zeros((1,1,3), dtype=np.float64) for i in range(N_OK)]

rms, camMatrix, coeffs, rvecs, tvecs = cv2.fisheye.calibrate(objpoints,imgpoints,gray.shape[::-1],camMatrix,coeffs,rvecs,tvecs,calibration_flags,(cv2.TermCriteria_EPS+cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-6))
DIM = _img_shape[::-1]

np.savez("calibration_data.npz", camMatrix=camMatrix, coeffs=coeffs, DIM= DIM)

import numpy as np
import cv2
import os
import datetime

calibration_data = np.load("calibration_data_1.npz")#load calibration data saved after calibration
camera_matrix = calibration_data['camMatrix'] #read camera matrix
dist_coeffs = calibration_data['coeffs'] #read dist_coeffs
DIM = calibration_data["DIM"] #read dimensiions of image on which calbration was done

rtsp_link ='rtsp://10.8.0.12:8554/h264'
os.environ['OPENCV_FFMPEG_CAPTURE_OPTIONS'] = "video_codec"

cap = cv2.VideoCapture(rtsp_link) # capture the video from rtsp_link

if not cap.isOpened():         #check if video is captured
    print("Error could not open video stream")
    exit()

def undistort(camera_matrix , dist_coeffs, DIM,balance):
    dim2 = None
    dim3 = None
    
    while True:
        ret,frame = cap.read()
        if not ret:
            print("Failed to capture frame from camera:")
            break
        
        try:
        
            font = cv2.FONT_HERSHEY_PLAIN
            map1, map2 = cv2.fisheye.initUndistortRectifyMap(camera_matrix, dist_coeffs, np.eye(3),camera_matrix,DIM, cv2.CV_16SC2)
            undistorted_frame = cv2.remap(frame, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode= cv2.BORDER_CONSTANT)
            date_time = str(datetime.datetime.now()) + str('   ')+str('Streaming camera id 001')
            undistorted_frame = cv2.putText(undistorted_frame,date_time,(100,700),font,2,(255,255,0),2,cv2.LINE_4)
            #Display undistorted image
            cv2.imshow('Original',frame)
            cv2.imshow('Undistorted Video', undistorted_frame)
            key = cv2.waitKey(1)
            # Exit key for program
            if key == ord("q"):
                break
        except Exception as e:
            print("Error in camera",":",e)
    
    cap.release()

undistort(camera_matrix,dist_coeffs,DIM,0.8)

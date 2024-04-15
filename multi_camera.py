import cv2
import numpy as np
import threading

#Undistort function 
def undistort(camera_matrix , dist_coeffs, DIM, vid, camera_index):
    dim2 = None
    dim3 = None
    
    while True:
        ret,frame = vid.read()
        if not ret:
            print("Failed to capture frame from camera:",camera_index)
            break
        
        try:
            dim1 = frame.shape[:2][::-1]
            assert dim1[0]/dim1[1] == DIM[0]/DIM[1]
            if not dim2:
                dim2 = dim1
            if not dim2:
                dim3 = dim1
            #Scale calibration matrix as per aspect ratio
            scaled_K = camera_matrix * dim1[0]/DIM[1]
            scaled_K[2][2] = 1.0
            # Undstort the image 
            new_K = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(scaled_K,dist_coeffs,dim2,np.eye(3),balance=0.0)
            map1, map2 = cv2.fisheye.initUndistortRectifyMap(scaled_K, dist_coeffs, np.eye(3), camera_matrix, DIM, cv2.CV_16SC2)
            undistorted_frame = cv2.remap(frame, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode= cv2.BORDER_CONSTANT)
            #Display undistorted image
            cv2.imshow('Undistorted Video' + str(camera_index), undistorted_frame)
            key = cv2.waitKey(1)
            # Exit key for program
            if key == ord("q"):
                break
        except Exception as e:
            print("Error in camera", camera_index,":",e)
        
    vid.release()

#Read calibration data, saved by calibration program    
calibration_data = np.load("calibration_data.npz")#load calibration data saved after calibration
camera_matrix = calibration_data['camera_matrix'] #read camera matrix
dist_coeffs = calibration_data['dist_coefss'] #read dist_coeffs
DIM = calibration_data["DIM"] #read dimensiions of image on which calbration was done

#Insert rtsp_url of different cameras
rtsp_ulr = [ "",""]
#Read video frame 
vids = [cv2.VideoCapture(url) for url in rtsp_ulr]

threads =[]
# Run all camera frame using multithreading
for i, vid in enumerate(vids):
    t = threading.Thread(target=undistort,args=(camera_matrix,dist_coeffs, DIM, vid, i))
    t.start()
    threads.append(t)
    
for t in threads:
    t.join()

#Exit the code
cv2.destroyAllWindows

            
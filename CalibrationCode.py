import cv2
import glob
import numpy as np 

CHECKERBOARD = (5,7)

subpix_criteria = (cv2.TERM_CRITERIA_MAX_ITER+cv2.TERM_CRITERIA_EPS,30,0.01)
calibration_flags = cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC+cv2.fisheye.CALIB_CHECK_COND+cv2.fisheye.CALIB_FIX_SKEW

objp = np.zeros((1,CHECKERBOARD[0]*CHECKERBOARD[1],3), np.float32)
objp[0,:,:2] = np.mgrid[0:CHECKERBOARD[0],0:CHECKERBOARD[1]].T.reshape(-1,2)

_img_shape = None
objpoints = []
imgpoints = []

images = glob.glob('*.png')

for image in images:
    img = cv2.imread(image)

    if _img_shape == None:
        _img_shape = img.shape[:2]
    else:
        assert _img_shape == img.shape[:2]

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    ret, corners = cv2.findChessboardCorners(gray,CHECKERBOARD,cv2.CALIB_CB_ADAPTIVE_THRESH+cv2.CALIB_CB_FAST_CHECK+cv2.CALIB_CB_NORMALIZE_IMAGE)

    if ret == True:
        objpoints.append(objp)
        corner2 = cv2.cornerSubPix(gray,corners,(3,3),(-1,-1),subpix_criteria)
        imgpoints.append(corner2)
    
N_OK = len(objpoints)
camMatrix = np.zeros((3,3))
coeffs = np.zeros((4,1))
rvecs = [np.zeros((1,1,3), dtype=np.float64) for i in range(N_OK)]
tvecs = [np.zeros((1,1,3), dtype=np.float64) for i in range(N_OK)]

rms, camMatrix, coeffs, rvecs, tvecs = cv2.fisheye.calibrate(objpoints, imgpoints,gray.shape[::-1],camMatrix,coeffs,rvecs,tvecs,calibration_flags,
        (cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-6))
DIM = _img_shape[::-1]
print('CAMERA MATRIX:',camMatrix)
print('Distortion Coefficients:',coeffs)

dim2 = None
dim3 = None

img= cv2.imread('/home/userpa/Desktop/calibration/calib119.png')
dim1 = img.shape[:2][::-1]
assert dim1[0]/dim1[1] == DIM[0]/DIM[1]
if not dim2:
      dim2 = dim1
if not dim3:
      dim3 = dim1

scaled_K = camMatrix* dim1[0]/DIM[0]
scaled_K[2][2] = 1.0
new_K = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(scaled_K,coeffs,dim2,np.eye(3))
map1, map2 = cv2.fisheye.initUndistortRectifyMap(scaled_K,coeffs,np.eye(3),camMatrix,DIM,cv2.CV_16SC2)
undistorted_img = cv2.remap(img, map1,map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
undistorted_img1 = cv2.fisheye.undistortImage(img,camMatrix,coeffs)
cv2.imshow('Undistorted Image',undistorted_img)
cv2.imshow('DistortedImage',img)
cv2.waitKey(0)
cv2.destroyAllWindows()




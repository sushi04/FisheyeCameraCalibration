Camera Calibration using RTSP Video Stream

This Python script captures frames from an RTSP (Real-Time Streaming Protocol) video stream, performs camera calibration using a checkerboard pattern, and saves the calibration data.

Requirements
Python 3.x
OpenCV (cv2)
NumPy
Installation

pip install opencv-python numpy

Usage
1.Open the video_calibration.py script.

2.Set the rtsp_link variable to the URL of your RTSP video stream.

3.Set the output_dir variable to the directory where you want to save the captured frames.

4.Run the script

5. Move the checkerboard to capture it in different orientations. Ensure that it covers various positions and angles relative to the camera.

6. Wait for the script to capture 100 frames from the video stream and perform camera calibration.

7. Once the calibration is complete, the script will save the calibration data to a file named calibration_data.npz.

Calibration Data
The calibration data includes:

1.Camera matrix (camMatrix)
2.Distortion coefficients (coeffs)
3.Image dimensions (DIM)



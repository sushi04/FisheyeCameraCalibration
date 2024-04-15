import cv2

# vid = cv2.VideoCapture(0) # For webcam
vid = cv2.VideoCapture(0) # For streaming links

while True:
    rdy, frame = vid.read()
    print(rdy)
    try:
        cv2.imshow('Video Live IP cam', frame)
        print("READINF")
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
    except:
        pass

vid.release()
cv2.destroyAllWindows()

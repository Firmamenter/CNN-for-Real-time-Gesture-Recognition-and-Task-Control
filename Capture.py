# Script to capture images for training and testing.
import cv2

# Capture video image
cap = cv2.VideoCapture(0)
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

cv2.namedWindow('Bigger', cv2.WINDOW_NORMAL)
# fgbg = cv2.createBackgroundSubtractorMOG2()
# fgbg = cv2.bgsegm.createBackgroundSubtractorMOG(history=1, nmixtures=5, backgroundRatio=0.1, noiseSigma=0.02)
fgbg = cv2.createBackgroundSubtractorMOG2(history=0, varThreshold=100, detectShadows=True)
# set rt size as 640x480
ret = cap.set(3, 400)
ret = cap.set(4, 400)

count = 0
while True:
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)

    cv2.imshow('frame', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        count += 1
        print ('%dth image' % count)

        cv2.imwrite('./newimages/ok.png' + str(count) + '.png', frame)

cap.release()
cv2.destroyAllWindows()
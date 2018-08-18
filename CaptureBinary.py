# Script to capture images for training and testing.
import cv2
from skimage import io, transform

# Create binary image.
def binaryMask(frame):
    roi = frame

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 2)
    th3 = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    ret, res = cv2.threshold(th3, 70, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return res

# Resize the input image.
def reSize(img, output_size):
    if len(img.shape) < 3:
        img = img.reshape(img.shape[0], img.shape[1], 1)
    else:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        img = img.reshape(img.shape[0], img.shape[1], 1)
    img = transform.resize(img, (output_size, output_size), mode='constant')
    return img

# Capture video image
cap = cv2.VideoCapture(0)

cv2.namedWindow('Bigger', cv2.WINDOW_NORMAL)
# fgbg = cv2.createBackgroundSubtractorMOG2()
# fgbg = cv2.bgsegm.createBackgroundSubtractorMOG(history=1, nmixtures=5, backgroundRatio=0.1, noiseSigma=0.02)
# fgbg = cv2.createBackgroundSubtractorMOG2(history=0, varThreshold=100, detectShadows=True)
# set rt size as 640x480
ret = cap.set(3, 400)
ret = cap.set(4, 400)

count = 0
while True:
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)
    roi = binaryMask(frame)

    # cv2.imshow('frame', frame)
    cv2.imshow('roi', roi)

    if cv2.waitKey(1) & 0xFF == ord('g'):
        count += 1
        print ('%dth image' % count)
        resized_img = cv2.resize(roi, (200, 200))
        cv2.imwrite('./new_test_img/ok' + str(count) + '.png', resized_img)

cap.release()
cv2.destroyAllWindows()
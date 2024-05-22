import cv2

cap = cv2.VideoCapture('../video.mp4')
#cap = cv2.VideoCapture('../video1.mp4')


bgsub = cv2.createBackgroundSubtractorMOG2()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    bgmask = bgsub.apply(frame)

    cv2.imshow('Frame', frame)
    cv2.imshow('BG Mask', bgmask)

    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

    elif k == ord('s'):
        cv2.imwrite('frame.png', frame)
        cv2.imwrite('bgmask.png', bgmask)

cap.release()
cv2.destroyAllWindows()

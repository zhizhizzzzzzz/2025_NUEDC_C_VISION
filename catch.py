#打开摄像头，按下k键后拍照保存，
#文件递增命名，按下q键退出程序
import cv2

cv2.namedWindow('frame',0)
cv2.resizeWindow('frame', 1280, 720)

cap = cv2.VideoCapture(2)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
cap.set(cv2.CAP_PROP_FPS, 60)

count = 0

while True:
    ret, frame = cap.read()
    cv2.imshow('frame', frame)
    k = cv2.waitKey(1)
    if k == ord('k'):
        cv2.imwrite('calib{}.jpg'.format(count), frame)
        count += 1
    elif k == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()

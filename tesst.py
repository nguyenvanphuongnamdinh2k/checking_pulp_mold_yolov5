import cv2
import datetime
print(datetime.now())
# cap = cv2.VideoCapture( r"rtsp://admin:admin111@192.168.1.160:554/cam/realmonitor?channel=1&qsubtype=00")
# while True:
#     ok,frame = cap.read()
#     # frame = frame[460:905,620:1130]
#     cv2.imshow('2', frame)
#     if cv2.waitKey(1) == ord('q'):
#         cv2.imwrite("img.jpg",frame)
#         break
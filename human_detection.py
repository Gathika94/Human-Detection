import cv2
import time

hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
cap = cv2.VideoCapture("/media/gathika/MainDisk/entgra_repos/human_detection/video/TownCentreXVID.avi")

while True:
    r, frame = cap.read()
    if r:
        start_time = time.time()
        frame = cv2.resize(frame, (1600, 900))  # Downscale to improve frame rate
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)  # HOG needs a grayscale image

        rects, weights = hog.detectMultiScale(gray_frame)

        # Measure elapsed time for detections
        end_time = time.time()
        count = 0
        print("Elapsed time:", end_time - start_time)

        for i, (x, y, w, h) in enumerate(rects):
            if weights[i] < 0.7:
                continue
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            count=count+1
        print("count :", count)

        # Write text on the frame
        font = cv2.FONT_HERSHEY_PLAIN
        bottomLeftCornerOfText = (10, 500)
        fontScale = 1
        fontColor = (255, 255, 255)
        lineType = 2
        cv2.putText(frame, str(count), (100, 100), font, 4, (0, 0, 255), 2, cv2.LINE_AA)

        cv2.imshow("preview", frame)
    k = cv2.waitKey(1)
    if k & 0xFF == ord("q"):  # Exit condition
        break

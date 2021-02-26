
from PIL import Image
import time
import cv2
import re
import os

def main():

    cap = cv2.VideoCapture(0)

    prev_time = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print('cannot get cam')
            break

        cv2.imshow('frame', frame)
        # print(results)
        if cv2.waitKey(1) == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
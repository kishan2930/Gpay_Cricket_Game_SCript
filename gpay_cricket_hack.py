import numpy as np
import cv2
from mss import mss
from PIL import Image
import os
import time

import pyautogui


mon = {"left": 320, "top": 500, "width": 330, "height": 300}

prev_y = 0
ys = []
params = cv2.SimpleBlobDetector.Params()
params.filterByArea = True
params.minArea = 200
detector = cv2.SimpleBlobDetector.create(params)

frames = []

MAX_FPS = 60

def click(x=360, y=300):
    # os.system('adb shell "cd sdcard && dd if=./record1 of=/dev/input/event4"')
    os.system(f"adb shell input tap {x} {y}")


def save_video(frames, frame_rate=60):
    height, width, layers = frames[0].shape
    size = (width, height)
    out = cv2.VideoWriter("project.avi", cv2.VideoWriter_fourcc(*"DIVX"), frame_rate, size)
    for i in range(len(frames)):
        out.write(frames[i])
    out.release()
    print(f"video saved to project.avi")

with mss() as sct:
    start = time.time()
    fc = 0

    prev_ball_detection_time = time.time()
    # set up video capture at MAX_FPS
    while True:
        # if time.time() - prev_frame_time < ( 1 / MAX_FPS ):
        #     continue
        
        prev_frame_time = time.time()
        fc += 1
        screenShot = sct.grab(mon)
        data = np.array(screenShot)
        data_gray = cv2.cvtColor(data, cv2.COLOR_BGR2GRAY)
        ret, data_gray = cv2.threshold(data_gray, 150, 255, cv2.THRESH_BINARY)
        
# Detect blobs.
        keypoints = detector.detect(data_gray)
        frame = cv2.drawKeypoints(data_gray, keypoints, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

        # append numpy array as frame
        # frames.append(cv2.cvtColor(data, cv2.COLOR_BGR2RGB))
        frames.append(frame)
        
        if len(frames) > 1000:
            # drop first 100 frames
            frames = frames[100:]

        if len(keypoints) > 0 and len(keypoints) <= 3:
            prev_ball_detection_time = time.time()
            x = keypoints[0].pt[0]
            y = keypoints[0].pt[1]

            speed = y - prev_y

            if speed > 0:
                print(f"speed {speed:.2f} (y = {y:.2f})")
                prev_y = y
                
            ys.append(y)
            if y > 135:
                p_x = np.random.randint(300, 400)
                p_y = np.random.randint(300, 450)
                pyautogui.click(x=p_x, y=p_y)
                # click(p_x, p_y)
                # click()
                # add green tint to frame
                frame[:, :, 1] = 255
                pass
        else:
            ys = []
            prev_y = 0

        if time.time() - start >= 1:
            print(f"FPS: {fc / (time.time() - start)}")
            start = time.time()
            fc = 0

        if time.time() - prev_ball_detection_time > 3:
            prev_ball_detection_time = time.time()
            pyautogui.click(x=600, y=1000)

        cv2.imshow("test", frame)

        if cv2.waitKey(5) & 0xFF in (
            ord("q"),
            27,
        ):
            ## save frames as video
            save_video(frames, 5)
            break

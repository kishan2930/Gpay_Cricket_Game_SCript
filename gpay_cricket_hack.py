import numpy as np
import cv2
from mss import mss
import os
import time
import pyautogui

# Define screen capture area
mon = {"left": 320, "top": 500, "width": 330, "height": 300}

# Blob detector setup
params = cv2.SimpleBlobDetector_Params()
params.filterByArea = True
params.minArea = 200
detector = cv2.SimpleBlobDetector.create(params)

# Video Writer setup
video_writer = cv2.VideoWriter("optimized_project.avi", cv2.VideoWriter_fourcc(*"XVID"), 30, (mon["width"], mon["height"]))

prev_y = 0  # Previous y-position of the ball
last_ball_time = time.time()  # Time of last detected ball
click_threshold_y = 135  # Threshold to trigger a click
click_delay = 0.2  # Time delay between clicks
frame_count = 0  # Frame counter

def click(x=360, y=300):
    """ Simulate a tap at given coordinates. """
    os.system(f"adb shell input tap {x} {y}")

with mss() as sct:
    try:
        while True:
            start_time = time.time()
            
            # Capture frame
            screenshot = sct.grab(mon)
            frame = np.array(screenshot)
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            _, binary_frame = cv2.threshold(gray_frame, 150, 255, cv2.THRESH_BINARY)

            # Detect blobs (ball detection)
            keypoints = detector.detect(binary_frame)
            output_frame = cv2.drawKeypoints(binary_frame, keypoints, np.array([]), (0, 0, 255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
            
            # Save frame to video file
            video_writer.write(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

            if keypoints:
                last_ball_time = time.time()  # Reset last ball detection time
                x, y = keypoints[0].pt  # Get ball coordinates

                # Compute speed (current y - previous y)
                speed = y - prev_y
                prev_y = y  # Update previous position

                print(f"Speed: {speed:.2f}, Ball Position: {x:.2f}, {y:.2f}")

                # Click logic (when ball reaches threshold)
                if y > click_threshold_y:
                    pyautogui.click(x=int(x), y=int(y) + np.random.randint(-10, 10))
                    time.sleep(click_delay)  # Prevent spamming clicks

            # Handle case where ball is not detected for 3+ seconds
            if time.time() - last_ball_time > 3:
                last_ball_time = time.time()
                pyautogui.click(x=600, y=1000)  # Possible restart button

            # Display frame
            cv2.imshow("Cricket Bot", output_frame)

            # Break if 'q' or 'ESC' is pressed
            if cv2.waitKey(5) & 0xFF in (ord("q"), 27):
                break

            # FPS control to prevent high CPU usage
            elapsed_time = time.time() - start_time
            sleep_time = max(0, (1 / 60) - elapsed_time)
            time.sleep(sleep_time)

    except Exception as e:
        print(f"Error: {e}")

    finally:
        video_writer.release()
        cv2.destroyAllWindows()
        print("Video saved successfully.")

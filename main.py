import cv2
import mediapipe as mp
import pyautogui
import time
import math

# Initialize modules
cap = cv2.VideoCapture(0)
screen_w, screen_h = pyautogui.size()
mpHands = mp.solutions.hands
hands = mpHands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mpDraw = mp.solutions.drawing_utils

drag_mode = False  # Flag to track drag state

while True:
    success, img = cap.read()
    img = cv2.flip(img, 1)
    h, w, _ = img.shape
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)

    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            lmList = []
            for id, lm in enumerate(handLms.landmark):
                cx, cy = int(lm.x * w), int(lm.y * h)
                lmList.append((cx, cy))

            mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)

            if len(lmList) >= 21:
                # Get finger tip coordinates
                index_x, index_y = lmList[8]     # Index tip
                thumb_x, thumb_y = lmList[4]     # Thumb tip
                middle_x, middle_y = lmList[12]  # Middle tip
                ring_x, ring_y = lmList[16]      # Ring tip

                # Move cursor
                pyautogui.moveTo(screen_w * index_x / w, screen_h * index_y / h)

                # Draw finger dots
                cv2.circle(img, (index_x, index_y), 8, (255, 0, 0), cv2.FILLED)
                cv2.circle(img, (thumb_x, thumb_y), 8, (0, 255, 0), cv2.FILLED)

                #CLICK GESTURE
                click_dist = math.hypot(index_x - thumb_x, index_y - thumb_y)
                if click_dist < 30:
                    pyautogui.click()
                    time.sleep(0.3)

                #SCROLLING GESTURE BASED ON FINGERS UP
                fingers_up = []
                tip_ids = [4, 8, 12, 16, 20]
                for i in range(1, 5):
                    if lmList[tip_ids[i]][1] < lmList[tip_ids[i] - 2][1]:
                        fingers_up.append(1)
                    else:
                        fingers_up.append(0)

                # 2 fingers up = scroll down
                if fingers_up == [1, 1, 0, 0]:
                    pyautogui.scroll(-50)
                    time.sleep(0.3)

                # 3 fingers up = scroll up
                elif fingers_up == [1, 1, 1, 0]:
                    pyautogui.scroll(50)
                    time.sleep(0.3)

                #DRAG & DROP GESTURE
                if click_dist < 30:
                    if not drag_mode:
                        pyautogui.mouseDown()
                        drag_mode = True
                        cv2.putText(img, "Dragging...", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                else:
                    if drag_mode:
                        pyautogui.mouseUp()
                        drag_mode = False

    # Show webcam feed
    cv2.imshow("Virtual Mouse", img)

    # Exit on 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Clean up
cap.release()
cv2.destroyAllWindows()



import cv2
import keyboard
import time
import math
import mediapipe as mediapipe  # Changed from mp to mediapipe
import multiprocessing as mp  # Keep mp for multiprocessing


class HandTrackingDynamic:
    def __init__(self, mode=False, maxHands=2, detectionCon=0.5, trackCon=0.5):
        self.__mode__ = mode
        self.__maxHands__ = maxHands
        self.__detectionCon__ = detectionCon
        self.__trackCon__ = trackCon
        self.handsMp = mediapipe.solutions.hands  # Changed from mp to mediapipe
        self.hands = self.handsMp.Hands()
        self.mpDraw = mediapipe.solutions.drawing_utils  # Changed from mp to mediapipe
        self.tipIds = [4, 8, 12, 16, 20]

    def findFingers(self, frame, draw=True):
        imgRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)
        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(
                        frame, handLms, self.handsMp.HAND_CONNECTIONS
                    )
        else:
            keyboard.send("space")
            print("No hands detected")
        return frame

    def findPosition(self, frame, handNo=0, draw=True):
        xList = []
        yList = []
        bbox = []
        self.lmsList = []
        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNo]
            for id, lm in enumerate(myHand.landmark):
                h, w, c = frame.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                xList.append(cx)
                yList.append(cy)
                self.lmsList.append([id, cx, cy])
                if draw:
                    cv2.circle(frame, (cx, cy), 5, (255, 0, 255), cv2.FILLED)
            xmin, xmax = min(xList), max(xList)
            ymin, ymax = min(yList), max(yList)
            bbox = xmin, ymin, xmax, ymax
            print("Hands Keypoint")
            print(bbox)
            if draw:
                cv2.rectangle(
                    frame,
                    (xmin - 20, ymin - 20),
                    (xmax + 20, ymax + 20),
                    (0, 255, 0),
                    2,
                )
        return self.lmsList, bbox

    def findFingerUp(self):
        fingers = []
        if self.lmsList[self.tipIds[0]][1] > self.lmsList[self.tipIds[0] - 1][1]:
            fingers.append(1)
        else:
            fingers.append(0)
        for id in range(1, 5):
            if self.lmsList[self.tipIds[id]][2] < self.lmsList[self.tipIds[id] - 2][2]:
                fingers.append(1)
            else:
                fingers.append(0)
        return fingers

    def findDistance(self, p1, p2, frame, draw=True, r=15, t=3):
        x1, y1 = self.lmsList[p1][1:]
        x2, y2 = self.lmsList[p2][1:]
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        if draw:
            cv2.line(frame, (x1, y1), (x2, y2), (255, 0, 255), t)
            cv2.circle(frame, (x1, y1), r, (255, 0, 255), cv2.FILLED)
            cv2.circle(frame, (x2, y2), r, (255, 0, 0), cv2.FILLED)
            cv2.circle(frame, (cx, cy), r, (0, 0, 255), cv2.FILLED)
        length = math.hypot(x2 - x1, y2 - y1)
        return length, frame, [x1, y1, x2, y2, cx, cy]


def run_hand_detection(data_queue):
    ctime = 0
    ptime = 0
    cap = cv2.VideoCapture(0)
    detector = HandTrackingDynamic()
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    if not cap.isOpened():
        print("Cannot open camera")
        return

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                continue

            frame = detector.findFingers(frame)
            lmsList, bbox = detector.findPosition(frame)

            # Send data through queue
            try:
                data_queue.put_nowait(
                    {
                        "landmarks": lmsList,
                        "bbox": bbox,
                        "hands_visible": len(lmsList) > 0,
                    }
                )
            except:
                pass  # Queue is full, skip this frame

            if len(lmsList) != 0:
                print(lmsList[0])

            ctime = time.time()
            fps = 1 / (ctime - ptime)
            ptime = ctime

            cv2.putText(
                frame,
                str(int(fps)),
                (10, 70),
                cv2.FONT_HERSHEY_PLAIN,
                3,
                (255, 0, 255),
                3,
            )

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            cv2.imshow("frame", gray)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    except KeyboardInterrupt:
        print("Stopping hand detection...")
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    # Create a queue for sharing data between processes
    data_queue = mp.Queue(maxsize=2)

    # Create and start the hand detection process
    hand_process = mp.Process(target=run_hand_detection, args=(data_queue,))
    hand_process.start()

    # Main program loop
    try:
        while True:
            try:
                hand_data = data_queue.get_nowait()
                if not hand_data["hands_visible"]:
                    # No hands detected
                    pass
                else:
                    # Process hand data
                    landmarks = hand_data["landmarks"]
                    bbox = hand_data["bbox"]
            except:
                pass

            time.sleep(0.01)

    except KeyboardInterrupt:
        print("Shutting down...")
    finally:
        hand_process.terminate()
        hand_process.join()


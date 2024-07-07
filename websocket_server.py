import asyncio
import websockets
import cv2
from cvzone.PoseModule import PoseDetector
import os

# Define the WebSocket handler
async def try_on(websocket, path):
    print("Client connected")
    try:
        while True:
            # Wait for a message from the client
            message = await websocket.recv()
            print(f"Message received: {message}")
            
            # Your try.py code
            cap = cv2.VideoCapture(0)
            detector = PoseDetector()

            shirtFolderPath = r"C:\Users\Asus\Downloads\clothes"
            listShirts = [f for f in os.listdir(shirtFolderPath) if f.endswith(('png', 'jpg', 'jpeg'))]

            fixedRatio = 262 / 190  # widthOfShirt / widthOfPoint11to12
            shirtRatioHeightWidth = 581 / 440

            imageNumber = 0
            imgShirt = cv2.imread(os.path.join(shirtFolderPath, listShirts[imageNumber]), cv2.IMREAD_UNCHANGED)

            while True:
                success, img = cap.read()
                if not success:
                    print("Failed to capture image from webcam")
                    break

                img = detector.findPose(img)
                lmList, bboxInfo = detector.findPosition(img, bboxWithHands=False)

                if bboxInfo:
                    center = bboxInfo["center"]
                    shirtWidth = int(fixedRatio * (bboxInfo["bbox"][2] - bboxInfo["bbox"][0]))
                    shirtHeight = int(shirtWidth * shirtRatioHeightWidth)
                    imgShirtResized = cv2.resize(imgShirt, (shirtWidth, shirtHeight))

                    y_offset = center[1] - shirtHeight // 2
                    x_offset = center[0] - shirtWidth // 2

                    y1, y2 = max(0, y_offset), min(img.shape[0], y_offset + shirtHeight)
                    x1, x2 = max(0, x_offset), min(img.shape[1], x_offset + shirtWidth)

                    y1_shirt = max(0, -y_offset)
                    x1_shirt = max(0, -x_offset)
                    y2_shirt = y1_shirt + (y2 - y1)
                    x2_shirt = x1_shirt + (x2 - x1)

                    if y2 - y1 > 0 and x2 - x1 > 0:
                        for c in range(0, 3):
                            img[y1:y2, x1:x2, c] = (
                                imgShirtResized[y1_shirt:y2_shirt, x1_shirt:x2_shirt, c]
                            )

                cv2.imshow("Image", img)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            cap.release()
            cv2.destroyAllWindows()
    except websockets.exceptions.ConnectionClosed:
        print("Client disconnected")

# Start the WebSocket server
async def main():
    async with websockets.serve(try_on, "localhost", 8765):
        print("WebSocket server started")
        await asyncio.Future()  # Run forever

asyncio.run(main())



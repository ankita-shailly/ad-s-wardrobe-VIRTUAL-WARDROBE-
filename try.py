import os
import cv2
from cvzone.PoseModule import PoseDetector
import numpy as np

# Initialize webcam
cap = cv2.VideoCapture(0)
detector = PoseDetector()

# Path to shirt images directory
shirtFolderPath = r"C:\Users\Asus\Downloads\clothes"

if not os.path.isdir(shirtFolderPath):
    raise NotADirectoryError(f"{shirtFolderPath} is not a valid directory")

# Get list of image files in the directory
listShirts = [f for f in os.listdir(shirtFolderPath) if f.endswith(('png', 'jpg', 'jpeg'))]
if not listShirts:
    raise FileNotFoundError(f"No image files found in the directory {shirtFolderPath}")

# Calculate the fixed aspect ratio
fixedRatio = 262 / 190  # widthOfShirt / widthOfPoint11to12
# fixedRatio = 180 / 190  # widthOfShirt / widthOfPoint11to12
shirtRatioHeightWidth = 581 / 440

# Initialize image number for shirt selection
imageNumber = 1

def load_shirt_image(imageNumber):
    shirtImagePath = os.path.join(shirtFolderPath, listShirts[imageNumber])
    imgShirt = cv2.imread(shirtImagePath, cv2.IMREAD_UNCHANGED)
    if imgShirt is None:
        raise FileNotFoundError(f"Failed to load image at path: {shirtImagePath}")
    return imgShirt

def overlay_shirt(img, imgShirt, x_offset, y_offset):
    shirt_h, shirt_w, shirt_c = imgShirt.shape
    img_h, img_w, img_c = img.shape

    if shirt_c == 4:  # Shirt image has an alpha channel
        alpha_s = imgShirt[:, :, 3] / 255.0
        alpha_l = 1.0 - alpha_s

        for c in range(0, 3):
            img[y_offset:y_offset+shirt_h, x_offset:x_offset+shirt_w, c] = (
                alpha_s * imgShirt[:, :, c] +
                alpha_l * img[y_offset:y_offset+shirt_h, x_offset:x_offset+shirt_w, c]
            )
    else:  # No alpha channel, just overlay the image
        img[y_offset:y_offset+shirt_h, x_offset:x_offset+shirt_w] = imgShirt

# Load the first shirt image
imgShirt = load_shirt_image(imageNumber)

# Main loop to read frames from webcam
while True:
    success, img = cap.read()
    if not success:
        print("Failed to capture image from webcam")
        break

    # Detect pose in the image
    img = detector.findPose(img)
    lmList, bboxInfo = detector.findPosition(img, bboxWithHands=False)

    # Check if pose landmarks were detected
    if bboxInfo:
        center = bboxInfo["center"]

        # Example: Draw a circle at the center of the bounding box
        cv2.circle(img, center, 5, (255, 0, 0), cv2.FILLED)

        # Example of how you might overlay the shirt image (simplified)
        # Assuming you want to overlay the shirt image at the center position
        shirtWidth = int(fixedRatio * (bboxInfo["bbox"][2] - bboxInfo["bbox"][0]))
        shirtHeight = int(shirtWidth * shirtRatioHeightWidth)
        imgShirtResized = cv2.resize(imgShirt, (shirtWidth, shirtHeight))

        y_offset = center[1] - shirtHeight // 2
        x_offset = center[0] - shirtWidth // 2

        # Ensure the offsets are within the bounds of the image dimensions
        y1, y2 = max(0, y_offset), min(img.shape[0], y_offset + shirtHeight)
        x1, x2 = max(0, x_offset), min(img.shape[1], x_offset + shirtWidth)

        # Calculate the dimensions of the resized shirt image to fit within the target area
        y1_shirt = max(0, -y_offset)  # Offset for the shirt image if y_offset is negative
        x1_shirt = max(0, -x_offset)  # Offset for the shirt image if x_offset is negative
        y2_shirt = y1_shirt + (y2 - y1)
        x2_shirt = x1_shirt + (x2 - x1)

        # Only overlay the shirt image if the target area has valid dimensions
        if y2 - y1 > 0 and x2 - x1 > 0:
            try:
                overlay_shirt(img[y1:y2, x1:x2], imgShirtResized[y1_shirt:y2_shirt, x1_shirt:x2_shirt], 0, 0)
            except ValueError as e:
                print(f"Error overlaying image: {e}")

    # Display the resulting frame
    cv2.imshow("Image", img)

    # Handle keypress events to change shirts
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

# Release the webcam and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()

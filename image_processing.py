# Install Tesseract OCR
!sudo apt-get install -y tesseract-ocr

# Install pytesseract
!pip install pytesseract

import cv2
import imutils
import numpy as np
import pytesseract
import re
import csv
import sys
import os


def show_image(title, image):
    """Helper function to display images using OpenCV."""
    cv2.imshow(title, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def manual_transformation(image):
    """Allows the user to manually select four points for perspective transformation."""
    points = []

    def select_point(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            points.append((x, y))
            cv2.circle(image, (x, y), 5, (0, 255, 0), -1)
            cv2.imshow("Select points", image)
        if len(points) == 4:
            cv2.destroyAllWindows()

    cv2.imshow("Select points", image)
    cv2.setMouseCallback("Select points", select_point)
    cv2.waitKey(0)

    if len(points) == 4:
        return four_point_transform(image, np.array(points, dtype="float32"))
    else:
        print("You need to select exactly four points!")
        return None

def sharpen_and_close(image):
    """Apply sharpening and morphological operations to enhance text."""
    kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    sharpened = cv2.filter2D(cv2.GaussianBlur(image, (3, 3), 0), -1, kernel)
    return cv2.dilate(cv2.erode(sharpened, np.ones((3, 3), np.uint8), iterations=1), np.ones((3, 3), np.uint8), iterations=1)

def four_point_transform(image, pts):
    """Perform a perspective transform to obtain a top-down view of the image."""
    rect = order_points(pts)
    (tl, tr, br, bl) = rect
    width = max(int(np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))),
                int(np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))))
    height = max(int(np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))),
                 int(np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))))
    dst = np.array([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]], dtype="float32")
    return cv2.warpPerspective(image, cv2.getPerspectiveTransform(rect, dst), (width, height))

def order_points(pts):
    """Order the points in a consistent way: top-left, top-right, bottom-right, bottom-left."""
    rect = np.zeros((4, 2), dtype="float32")
    s, diff = pts.sum(axis=1), np.diff(pts, axis=1)
    rect[0], rect[2], rect[1], rect[3] = pts[np.argmin(s)], pts[np.argmax(s)], pts[np.argmin(diff)], pts[np.argmax(diff)]
    return rect

def text_extraction(warped):
    """Extract text from the warped image using Tesseract OCR."""
    text = pytesseract.image_to_string(warped, config=r'--oem 3 -l eng --psm 6')
    print("Extracted Text:", text)
    extracted_data = clean_and_extract_data(text)
    if extracted_data:
        write_to_csv(extracted_data, "extracted_data.csv")
        print("Data written to extracted_data.csv")
    else:
        print("No data extracted.")

def clean_and_extract_data(text):
    """Clean and extract relevant data from the OCR text."""
    return [{'Item': match.group(1).strip(), 'Qty': int(match.group(2).strip()), 'Total': float(match.group(3).strip())}
            for line in text.splitlines() if (match := re.match(r'(.+?)\s+(\d+)\s+([\d.]+)', line))]

def write_to_csv(extracted_data, output_csv):
    """Write the extracted data to a CSV file."""
    with open(output_csv, mode='w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=['Item', 'Qty', 'Total'])
        writer.writeheader()
        writer.writerows(extracted_data)

def main(image_path):
    """Main function to perform image processing and text extraction."""
    image = cv2.imread(image_path)
    
    if image is None:
        print(f"Error: Unable to load image at {image_path}. Please check the file path.")
        return
    
    ratio = image.shape[0] / 500.0
    orig = image.copy()
    image = imutils.resize(image, height=500)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(gray, 75, 200)

    print("STEP 1: Edge Detection")
    show_image("Original Image", image)
    show_image("Edged Image", edged)
    cv2.imwrite(os.path.join("outputImages", "gray_image.png"), gray)
    cv2.imwrite(os.path.join("outputImages", "edged_image.png"), edged)

    cnts = sorted(imutils.grab_contours(cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)),
                  key=cv2.contourArea, reverse=True)[:5]

    for c in cnts:
        if len((approx := cv2.approxPolyDP(c, 0.02 * cv2.arcLength(c, True), True))) == 4:
            screenCnt = approx
            break
    else:
        screenCnt = None

    if screenCnt is None:
        print("No contour with four points detected. Please manually select the corners.")
        if (warped := manual_transformation(orig)) is not None:
            processed_image = sharpen_and_close(warped)
            cv2.imwrite(os.path.join("outputImages", "processed_image.png"), processed_image)
            show_image("Processed Image", processed_image)
            text_extraction(processed_image)
    else:
        print("STEP 2: Find contours of paper")
        cv2.drawContours(image, [screenCnt], -1, (0, 255, 0), 2)
        show_image("Outline", image)
        warped = four_point_transform(orig, screenCnt.reshape(4, 2) * ratio)
        text_extraction(sharpen_and_close(warped))

if __name__ == "__main__":
    if len(sys.argv) > 1:
        main(sys.argv[1])
    else:
        print("Please provide the image path as a command-line argument.")
import pytesseract
import imutils
from imutils.perspective import four_point_transform
import cv2
import re
import numpy as np 

class ReceiptParser:
    def __init__(self, image):
        self.original_image = image

    def apply_close_morphology(self, img):
        kernel = np.ones((2, 2), np.uint8)
        image = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel, iterations=10)
        return image

    def auto_canny(self, gray_img):
        high_thresh, thresh_im = cv2.threshold(gray_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        lowThresh = 0.5 * high_thresh
        edged = cv2.Canny(gray_img, lowThresh, high_thresh)
        return edged

    def apply_edge_detection(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (11, 11), 0)
        threshold = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
        threshold = cv2.fastNlMeansDenoising(threshold, 11, 31, 9)
        canny = self.auto_canny(threshold)

        HEIGHT, WIDTH = canny.shape 
        topContour = np.array([[0, HEIGHT], [WIDTH, HEIGHT]])
        bottomContour = np.array([[0, 0], [WIDTH, 0]])

        cv2.drawContours(canny,[bottomContour],0,(255,255,255),2)
        cv2.drawContours(canny,[topContour],0,(255,255,255),2)

        canny = cv2.dilate(canny, np.ones((5, 5)), iterations=8)

        return canny

    def find_bounding_rectangle_area(self, contour):
        x,y,w,h = cv2.boundingRect(contour)
        return w * h 

    def find_largest_contour(self, canny):
        contours, hierarchy = cv2.findContours(canny, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        max_contour = np.array([None])
        max_area = 0
        precision_mult = 0.015
        while not max_contour.any():
            for contour in contours:
                    peri = cv2.arcLength(contour, True)
                    corners = cv2.approxPolyDP(contour, precision_mult * peri, True)

                    if len(corners) == 4:
                        contour_area = self.find_bounding_rectangle_area(contour)
                        if contour_area > max_area:
                            max_area, max_contour = contour_area, corners
            precision_mult *= 1.5
            
        return max_contour[:, 0, :]

    def draw_corners(self, image, corners):
        for corner in corners:

            image = cv2.circle(image, (corner[0], corner[1]), radius=10, color=(0, 0, 255), thickness=-1)
        return image

    def find_receipt(self, img):
        img = self.apply_close_morphology(img)
        img = self.apply_edge_detection(img)
        corners = self.find_largest_contour(img)  

        copy = self.draw_corners(self.original_image.copy(), corners)
        self.show_image(copy)
        return corners 

    def apply_perspective_transform(self, img):


    def show_image(self, img):
        cv2.imshow("Demo", img)
        cv2.waitKey(0)





image = cv2.imread("1016-receipt.jpg")
parser = ReceiptParser(image)
parser.apply_perspective_transform(image)




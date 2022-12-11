import pytesseract
import imutils
from imutils.perspective import four_point_transform
import cv2
import re
import numpy as np 

class ReceiptParser:
    def __init__(self):
        pass

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

        canny = cv2.dilate(canny, np.ones((4, 4)), iterations=8)
        return canny

    def find_bounding_rectangle_area(self, contour):
        x,y,w,h = cv2.boundingRect(contour)
        return w * h 

    def addContoursToImage(self, img, contours):
        for contour in contours:
            cv2.drawContours(img, [contour], 0, (255, 255, 255), 32)
        return img

    def find_largest_robust_contour(self, canny):
        HEIGHT, WIDTH = canny.shape 
        image_area = HEIGHT * WIDTH

        bottomContour = np.array([[0, HEIGHT], [WIDTH, HEIGHT]])
        topContour = np.array([[0, 0], [WIDTH, 0]])
        leftContour = np.array([[0, 0], [0, HEIGHT]])
        rightContour = np.array([[WIDTH, 0], [WIDTH, HEIGHT]])

        topBottomImage = self.addContoursToImage(canny.copy(), [topContour, bottomContour])
        leftRightImage = self.addContoursToImage(canny.copy(), [leftContour, rightContour])
        rightBottomImage = self.addContoursToImage(canny.copy(), [rightContour, bottomContour])
        leftBottomImage = self.addContoursToImage(canny.copy(), [leftContour, bottomContour])
        leftTopImage = self.addContoursToImage(canny.copy(), [leftContour, topContour])
        rightTopImage = self.addContoursToImage(canny.copy(), [rightContour, topContour])

        images = [topBottomImage, leftRightImage, rightBottomImage, leftBottomImage, leftTopImage, rightTopImage]
        for image in images:
            receipt = self.find_largest_contour(image)
            print(receipt)
            if self.find_bounding_rectangle_area(receipt) > 0.3 * image_area:
                return receipt 

        return np.array([[0,0], [WIDTH, 0], [0, HEIGHT], [WIDTH, HEIGHT]])

    def find_largest_contour(self, canny):
        contours, hierarchy = cv2.findContours(canny, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        image_area = canny.shape[0] * canny.shape[1]
        max_contour = np.array([None])
        max_area = 0
        precision_mult = 0.01                

        while not max_contour.any() or max_area < 0.3 * image_area and precision_mult < 0.2:
            for contour in contours:
                    peri = cv2.arcLength(contour, True)
                    corners = cv2.approxPolyDP(contour, precision_mult * peri, True)

                    if len(corners) == 4:
                        contour_area = self.find_bounding_rectangle_area(contour)
                        if contour_area > max_area and contour_area < 0.9 * image_area:
                            max_area, max_contour = contour_area, corners

            precision_mult *= 1.5
            
        return max_contour[:, 0, :]

    def draw_corners(self, image, corners):
        for corner in corners:
            image = cv2.circle(image, (corner[0], corner[1]), radius=10, color=(0, 0, 255), thickness=-1)
        return image

    def resize_image(self, img, desired_width):
        height, width, channels = img.shape 
        scale_factor = desired_width / width 
        desired_height = int(scale_factor * height)

        dim = (desired_width, desired_height)
        resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)

        return resized, scale_factor

    def adjust_corners_for_dilation(self, corners, factor=4, iter=8):
        shift = factor * iter 

        lowest = sorted(corners, key=lambda x: x[0])
        lowest[0][0] -= shift
        lowest[1][0] -= shift


        lowest = sorted(lowest, key=lambda x: x[1], reverse=True)
        lowest[0][1] -= shift
        lowest[1][1] -= shift

        return lowest

    def resize_corners(self, corners, scale_factor):
        for i in range(len(corners)):
            corners[i][0] *= (1 / scale_factor)
            corners[i][1] *= (1 / scale_factor)
        
        return corners 

    def clamp_corners(self, orig_image, corners):
        height, width, channels = orig_image.shape 

        def clamp(val, min, max):
            return sorted((min, val, max))[1]

        for i in range(len(corners)):   
            corners[i][0] = clamp(0, corners[i][0], width)
            corners[i][1] = clamp(0, corners[i][1], height)

        return corners 


    def draw_bounding_box(self, img, corners):
        lowest = sorted(corners, key=lambda x: x[0])
        cv2.line(img, (lowest[0][0], lowest[0][1]), (lowest[1][0], lowest[1][1]), (0, 255, 0), 2)
        lowest = sorted(corners, key=lambda x: x[1])
        cv2.line(img, (lowest[0][0], lowest[0][1]), (lowest[1][0], lowest[1][1]), (0, 255, 0), 2)
        lowest = sorted(corners, key=lambda x: x[0], reverse=True)
        cv2.line(img, (lowest[0][0], lowest[0][1]), (lowest[1][0], lowest[1][1]), (0, 255, 0), 2)
        lowest = sorted(corners, key=lambda x: x[1], reverse=True)
        cv2.line(img, (lowest[0][0], lowest[0][1]), (lowest[1][0], lowest[1][1]), (0, 255, 0), 2)
        return img 


    def find_receipt(self, img):
        copy = img.copy()
        img, scale_factor = self.resize_image(img, 500)

        img = self.apply_close_morphology(img)
        img = self.apply_edge_detection(img)
        corners = self.find_largest_robust_contour(img)

        corners = self.adjust_corners_for_dilation(corners)
        corners = self.resize_corners(corners, scale_factor)
        corners = self.clamp_corners(copy, corners)

        copy = self.draw_corners(copy, corners)
        copy = self.draw_bounding_box(copy, corners)
        self.show_image(copy)
        return corners 

    def apply_perspective_transform(self, img, corners):
        pass

    def show_image(self, img):
        cv2.imshow("Demo", img)
        cv2.waitKey(0)


import glob, os

for filename in glob.iglob('images/**', recursive=True):
    if not filename == "images/":
        image = cv2.imread(filename)
        parser = ReceiptParser()
        parser.find_receipt(image)




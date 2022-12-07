import pytesseract
import imutils
from imutils.perspective import four_point_transform
import cv2
import re

class ReceiptParser:
    def __init__(self, image):
        self.original_image = image 
        self.processed_image = None
        self.__initialize_processed_image()
        self.__parse_receipt()
        
    def __initialize_processed_image(self):
        self.processed_image = self.original_image.copy()
        self.processed_image = imutils.resize(self.processed_image, width=500)
        ratio = self.original_image.shape[1] / float(self.processed_image.shape[1])

        # Grayscale the image, blur it slightly, and detect edges 
        gray = cv2.cvtColor(self.processed_image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edged = cv2.Canny(blurred, 100, 200)

        # Find contours in descending order and choose the one with the largest size
        cnts = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        cnts = sorted(cnts, key=cv2.contourArea, reverse=True)

        # Initialize a contour that corresponds to the receipt outline
        receiptCnt = None
        for c in cnts:
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.05 * peri, True)
        
            if len(approx) == 4:
                receiptCnt = approx
                break

        if receiptCnt is None:
            raise Exception(("Could not find receipt outline. "
                "Try debugging your edge detection and contour steps."))

        receipt = four_point_transform(self.original_image.copy(), receiptCnt.reshape(4, 2) * ratio)
        self.processed_image = receipt
    
    
    def __parse_receipt(self):
        

    def get_line_items(self):
        pass 

    def get_subtotal(self):
        pass

    def get_total(self):
        pass 

image = cv2.imread("receipt.jpg")

parser = ReceiptParser(image)
parser.initialize_processed_image()




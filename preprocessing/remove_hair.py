import cv2
import math
import numpy as np
import inpaint as ip

'''
    remove the hair
'''


# use kernel to detect the hair
def detect_hair(img):
    g_i = g_i_kernel()
    g_j = g_j_kernel()
    g_x = cv2.filter2D(img, -1, g_i)
    g_y = cv2.filter2D(img, -1, g_j)
    g = abs(g_y) + abs(g_x)
    g = np.clip(g, 0, 255)

    return g


# create x-direction kernel
def g_i_kernel():
    kernel = np.zeros([3, 3])
    sigma = 0.55
    for i in range(-1, 2):
        for j in range(-1, 2):
            g_i = -j / (math.sqrt(0.8 * math.pi) * (sigma ** 4)) * (math.e ** ((-(j ** 2)) / (2 * (sigma ** 2))))
            if j == 0 and i == 0:
                g_i = -0.1
            kernel[i + 1, j + 1] = g_i
    return kernel


# create y-direction kernel
def g_j_kernel():
    g_i = g_i_kernel()
    return g_i.T


# use Otsu method to calculate the threshold of gradient image
def Otsu(img):

    # calculate the histogram of image
    his = cv2.calcHist([img], [0], None, [256], [0, 256])
    prob = his / (360 * 390)
    i_list = [x for x in range(256)]
    threshold = 1
    maxT = -1
    his_list = [x[0] for x in his]

    # calculate the maximum variance between background and front ground to identify the threshold
    for i in range(1, 255):
        p1 = sum(prob[0 : i])[0]
        p2 = 1 - p1
        if sum(sum(his[0 : i])) == 0:
            continue
        m1 = sum(np.multiply(np.array(his_list[0:i]), np.array(i_list[0:i]))) / sum(his_list[0:i])
        if sum(sum(his[i + 1 : 256])) == 0:
            continue
        m2 = sum(np.multiply(np.array(his_list[i + 1:256]), np.array(i_list[i + 1:256]))) / sum(his_list[i + 1:256])
        tempT = p1*p2*((m1 - m2) ** 2)
        if tempT > maxT:
            maxT = tempT
            threshold = i
    return threshold


# enhance the area of hair
def hair_enhancement(img):

    # calculate Otsu threshold
    threshold = Otsu(img)

    # OpenCV's Otsu
    # ret, img = cv2.threshold(img, 0, 255, cv2.THRESH_OTSU)
    ret, img = cv2.threshold(img, threshold, 255, cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # remove small areas
    for i in range(len(contours)):
        contour = contours[i]
        if len(contour) <= 10:
            cv2.drawContours(img, [contour], -1, 0, -1)

    # dilate
    kernel = np.ones((3, 3), dtype=np.uint8)
    img = cv2.dilate(img, kernel, iterations=2)
    # kernel = np.ones((3, 3), dtype=np.uint8)
    # img = cv2.erode(img, kernel, iterations=1)
    return img


# improve the contrast of images
def improve_contrast(img):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(3, 3))
    b, g, r = cv2.split(img)
    b = clahe.apply(b)
    g = clahe.apply(g)
    r = clahe.apply(r)
    img = cv2.merge([b, g, r])
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    return img


# the backbone of whole process
def backbone(original):
    img = improve_contrast(original)

    # detect the hair areas
    img = detect_hair(img)

    # img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 23, 1)
    img = hair_enhancement(img)
    ip.inpaint(original, img)
    # original = cv2.inpaint(original, img, 3, cv2.INPAINT_TELEA)
    return original


# test function
if __name__ == '__main__':
    img = cv2.imread("/Users/lichengtai/Desktop/CV/CW/CW_DATASET/original/yes/lctyes1_process229.jpeg")[59:419, 124:514]
    img = backbone(img)
    cv2.imshow("1", img)
    cv2.waitKey(0)

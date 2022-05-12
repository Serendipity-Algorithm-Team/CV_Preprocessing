import numpy as np
import cv2


'''
    transfer the color from img1 to img2
'''


def color_transfer(img1, img2):

    # convert the rgb to lab
    img1 = cv2.cvtColor(img1, cv2.COLOR_RGB2LAB).astype("float32")
    img2 = cv2.cvtColor(img2, cv2.COLOR_RGB2LAB).astype("float32")

    l1, a1, b1 = cv2.split(img1)
    l2, a2, b2 = cv2.split(img2)

    # calculate the mean values and standard deviations of two images
    l_mean1, a_mean1, b_mean1 = l1.mean(), a1.mean(), b1.mean()
    l_mean2, a_mean2, b_mean2 = l2.mean(), a2.mean(), b2.mean()
    l_std1, a_std1, b_std1 = l1.std(), a1.std(), b1.std()
    l_std2, a_std2, b_std2 = l2.std(), a2.std(), b2.std()

    # minus the mean values of img2
    l2 -= l_mean2
    a2 -= a_mean2
    b2 -= b_mean2

    # scale using standard deviations
    l = (l_std2 / l_std1) * l2
    a = (a_std2 / a_std1) * a2
    b = (b_std2 / b_std1) * b2

    # add in the img1 mean values
    l += l_mean1
    a += a_mean1
    b += b_mean1

    l = np.clip(l, 0, 255)
    a = np.clip(a, 0, 255)
    b = np.clip(b, 0, 255)

    output = cv2.merge([l, a, b])
    output = cv2.cvtColor(output.astype("uint8"), cv2.COLOR_LAB2RGB)

    return output


# test function
if __name__ == '__main__':
    img = cv2.imread("/Users/lichengtai/Desktop/CV/CW/CW_DATASET/original/yes/hhyes1_process0.jpeg")[59:419, 124:514]
    img2 = cv2.imread("/Users/lichengtai/Desktop/CV/CW/CW_DATASET/original/yes/lctyes1_process54.jpeg")[59:419, 124:514]
    img3 = color_transfer(img, img2)
    cv2.imshow("1", img3)
    cv2.waitKey(0)
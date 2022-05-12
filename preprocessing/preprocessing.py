import cv2
import os
import windows
import abnormal_detection
import uni_skin_color
import remove_hair

'''
    this is the main function to do the preprocessing
'''


# read the dataset
def read_path(file_pathname, taget_file, make_up):

    if make_up == 0:
        original = cv2.imread("/Users/lichengtai/Desktop/CV/CW/CW_DATASET/original/yes/hhyes1_process0.jpeg")[59:419, 124:514]
    else:
        original = cv2.imread("/Users/lichengtai/Desktop/CV/CW/CW_DATASET/original/no/hhno1_process32.jpeg")[59:419, 124:514]

    for filename in os.listdir(file_pathname):
        if os.path.splitext(filename)[1] == '.jpeg':
            img = cv2.imread(file_pathname + '/' + filename)

            # get the centre of images
            img = img[59:419, 124:514]

            # check the images
            if abnormal_detection.random_select_30(img) != 0:
                img = uni_skin_color.color_transfer(original, img)
                img = pre_processing(img)
                img_small = windows.split_into_windows(img)
                for i in range(len(img_small)):
                    cv2.imwrite(taget_file + "/small" + filename[:len(filename) - 5] + "_" + str(i) + ".jpeg", img_small[i])


# do the preprocessing
def pre_processing(img):
    img = remove_hair.backbone(img)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # img = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
    # img = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    # img = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    # img = cv2.cvtColor(img, cv2.COLOR_RGB2LUV)
    # img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    # img = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)

    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(11, 11))
    img = clahe.apply(img)
    return img


# run the preprocessing
if __name__=='__main__':
    read_path("/Users/lichengtai/Desktop/CV/CW/CW_DATASET/original/yes", "/Users/lichengtai/Desktop/CV/CW/CW_DATASET/small/yes", 0)
    read_path("/Users/lichengtai/Desktop/CV/CW/CW_DATASET/original/no", "/Users/lichengtai/Desktop/CV/CW/CW_DATASET/small/no", 1)

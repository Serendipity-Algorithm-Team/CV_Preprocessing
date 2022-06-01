import random

'''
    This file is used to detect the abnormal images
    具体的逻辑是随机挑选30条宽度为10像素的像素条进行对rgb每一个通道的错位比较来确定是否为损坏的图像。
'''


# calculate the mean value of r, g, b
def cal_rgb_mean(img):
    rgb = []
    for i in range(3):
        rgb.append(sum(list(map(sum, img[:, :, i]))) / (img.shape[0] * img.shape[1]))
    return rgb


# if difference between two channel large than 35, it is the abnormal image
def find_abnormal(rgb, mean_list):
    if len(mean_list) == 0:
        return 1
    for i in range(3):
        for j in range(3):
            if any([abs(rgb[i] - rgb_pre[j]) > 35 for rgb_pre in mean_list]):
                return 0
    return 1


# randomly select 30 rows
def random_select_30(img):
    mean_list = []
    for i in range(30):
        row = random.randint(1, 30)
        row_pixel = img[10 * (row-1):10 * row]
        rgb = cal_rgb_mean(row_pixel)
        if find_abnormal(rgb, mean_list) == 0:
            return 0
        mean_list.append(rgb)
    return 1

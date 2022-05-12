'''
    use windows to select the small pieces of images
'''


# split the images into small images
def split_into_windows(img):

    img_list = []

    # identify the size of window
    height = int(img.shape[0] / 2.5)
    width = int(img.shape[1] / 2.5)
    height_step = int(height / 2)
    width_step = int(width / 2)
    window_position_h_1 = 0
    window_position_h_2 = height
    window_position_w_1 = 0
    window_position_w_2 = width

    # move the window to get small images
    for i in range(int(img.shape[1] / width) * 2 - 1):
        for j in range(int(img.shape[0] / height) * 2 - 1):
            img_list.append(img[window_position_h_1:window_position_h_2, window_position_w_1:window_position_w_2])
            window_position_h_1 += height_step
            window_position_h_2 += height_step
        window_position_h_1 = 0
        window_position_h_2 = height
        window_position_w_1 += width_step
        window_position_w_2 += width_step

    return img_list

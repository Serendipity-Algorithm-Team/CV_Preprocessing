import cv2
import numpy as np
import math

'''
    This file is used to inpaint the image
'''


# calculate the gradients of image
def cal_gradient(img, mask):
    sobel_x = np.array([[1, 0, -1]])
    sobel_y = np.array([[1], [0], [-1]])
    gradient_x = cv2.filter2D(img, -1, sobel_x)
    gradient_y = cv2.filter2D(img, -1, sobel_y)

    # remove the impact of hair area
    for i in range(len(mask)):
        for j in range(len(mask)):
            if mask[i, j] == 255 and gradient_x[i, j] >= 2:
                gradient_x[i, j] = 0
            if mask[i, j] == 255 and gradient_x[i, j] >= 1.5:
                gradient_x[i, j] //= 2

            if mask[i, j] == 255 and gradient_y[i, j] >= 2:
                gradient_y[i, j] = 0
            if mask[i, j] == 255 and gradient_y[i, j] >= 1.5:
                gradient_y[i, j] //= 2

    return gradient_x, gradient_y


# calculate the gradients of t
def cal_gradient_t(T_mat):
    sobel_x = np.array([[-0.5, 0, 0.5]])
    sobel_y = np.array([[-0.5], [0], [0.5]])
    tx = cv2.filter2D(T_mat, -1, sobel_x)
    ty = cv2.filter2D(T_mat, -1, sobel_y)
    return tx, ty


# find neighbourhoods(q) of target pixel p
def find_p_q(mask, T_mat, T_value):
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    p_q_list = []
    for i in range(len(contours)):
        contour = contours[i]
        for j in range(len(contour)):
            x = ((contour[j])[0])[0]
            y = ((contour[j])[0])[1]
            size = len(contour)
            p = [y, x, size]
            T_mat[y, x] = T_value
            p_q_list.append(p)
    return p_q_list, T_mat


# keep the boundaries of mask are 0
def deal_mask_bound(mask):
    mask[0, :] = 0
    mask[:, 0] = 0
    mask[359, :] = 0
    mask[:, 389] = 0
    return mask


# get some values of neighbourhoods
def get_nei(img, mask, T_mat, gx, gy, up_b, down_b, left_b, right_b, tx, ty):
    nei_p = img[up_b:down_b, left_b:right_b + 1, :]
    nei_mask = mask[up_b:down_b + 1, left_b:right_b + 1]
    nei_T = T_mat[up_b:down_b + 1, left_b:right_b + 1]
    nei_gx = gx[up_b:down_b + 1, left_b:right_b + 1]
    nei_gy = gy[up_b:down_b + 1, left_b:right_b + 1]
    nei_tx = tx[up_b:down_b + 1, left_b:right_b + 1]
    nei_ty = ty[up_b:down_b + 1, left_b:right_b + 1]
    return nei_p, nei_mask, nei_T, nei_gx, nei_gy, nei_tx, nei_ty


# inpaint one pixel
def inpaint_one_pixel(T_value, radius, com_up, com_down, com_right, com_left, nei_p, nei_mask, nei_T, nei_gy, nei_gx, p_sum, weight_sum, dx, dy, nei_tx, nei_ty):
    for m in range(nei_p.shape[0]):
        for n in range(nei_p.shape[1]):
            p_y = radius // 2 + com_up + com_down
            p_x = radius // 2 + com_right + com_left
            distance_y = p_y - m
            distance_x = p_x - n

            # do not select the p
            if (distance_x == 0) and (distance_y == 0):
                continue
            dis_len_sq = distance_y ** 2 + distance_x ** 2
            dis_len = math.sqrt(dis_len_sq)

            # select neighbourhoods in a circle
            if dis_len > radius:
                continue
            if nei_mask[m, n] == 255:
                continue

            t_value = nei_T[m, n]

            # calculate dir()
            dir_factor = abs(distance_y * nei_gy[m, n] * nei_ty[m, n] + distance_x * nei_gx[m, n] * nei_tx[m, n])
            if dir_factor == 0:
                dir_factor = 0.0000001

            # calculate lev()
            level_factor = 1.0 / (1.0 + abs(t_value - T_value))

            # calculate dis()
            dist_factor = 1.0 / dis_len_sq
            weight = abs(dist_factor * dir_factor * level_factor)
            dx -= weight * nei_gx[m, n] * distance_x
            if dx == 0:
                dx = 0.0000001
            dy -= weight * nei_gy[m, n] * distance_y
            if dy == 0:
                dy = 0.0000001

            # calculate sum of p
            p_sum[0] += weight * nei_p[m, n, 0]
            p_sum[1] += weight * nei_p[m, n, 1]
            p_sum[2] += weight * nei_p[m, n, 2]

            # calculate sum of weight
            weight_sum += weight

    return p_sum, weight_sum, dx, dy


# the inpaint main function
def inpaint(img, mask):
    mask = deal_mask_bound(mask)

    # calculate the T matrix
    T_mat = np.zeros([img.shape[0], img.shape[1]])
    T_value = 1
    grey = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # calculate the gradients of x-direction and y-direction
    gx, gy = cal_gradient(grey, mask)

    # inpaint all pixels
    while np.sum(np.sum(mask)) != 0:

        p_q_list, T_mat = find_p_q(mask, T_mat, T_value)
        tx, ty = cal_gradient_t(T_mat)

        # inpaint a circle of mask
        for i in range(len(p_q_list)):
            weight_sum = 0
            p_sum = [0, 0, 0]
            p = p_q_list[i]
            if p[2] < 16:
                radius = 3
            elif p[2] < 64:
                radius = 5
            else:
                radius = 7

            dx = 0
            dy = 0
            com_up = 0
            com_down = 0
            com_left = 0
            com_right = 0

            # identify the relative position of p
            up_b = p[0] - radius // 2
            if up_b < 0:
                up_b = 0
                com_up = 0 - (p[0] - radius // 2)

            down_b = p[0] + radius // 2 + 1
            if down_b > 359:
                down_b = 359
                com_down = p[0] + radius // 2 - 359

            left_b = p[1] - radius // 2
            if left_b < 0:
                left_b = 0
                com_left = 0 - (p[1] - radius // 2)

            right_b = p[1] + radius // 2 + 1
            if right_b > 389:
                right_b = 389
                com_right = p[1] + radius // 2 - 389

            nei_p, nei_mask, nei_T, nei_gx, nei_gy, nei_tx, nei_ty = get_nei(img, mask, T_mat, gx, gy, up_b, down_b, left_b, right_b, tx, ty)
            p_sum, weight_sum, dx, dy = inpaint_one_pixel(T_value, radius, com_up, com_down, com_right, com_left, nei_p, nei_mask, nei_T, nei_gy, nei_gx, p_sum, weight_sum, dx, dy, nei_tx, nei_ty)

            # calculate the pixel value of p
            p_im = [x / weight_sum + ((dx + dy) / math.sqrt(dx * dx + dy * dy)) for x in p_sum]
            img[p[0], p[1]] = p_im
            mask[p[0], p[1]] = 0
            T_value += 1

    # keep image between 0 to 255
    img = np.clip(img, 0, 255)

    return img


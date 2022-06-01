## Introduction
此项目包含所有关于洁面仪图像预处理的代码。首先我们要明确此项目中的图像预处理的作用: <br>
1. 确保模型学习到的特征是残妆的相关特征。<br>
2. 去除因为各种原因损坏了的照片的影响。<br>
3. 能够应对更多的使用场景。

## System Design
清楚了图像处理的作用，我们基于需求设计了相关系统。

![img](https://github.com/Serendipity-Algorithm-Team/CV_Preprocessing/blob/main/img/system.png)

1. 首先我们需要input一张图像通道为RGB的图像。<br>
2. 因为原图像周围有相机的边框，我们需要取图像的中心。<br>
3. 筛选错误的图像，把筛选错误的图像放在第二步而不是第一步的原因是经过图像的裁剪后，计算量会降低。<br>
4. 统一所有图像中的颜色，这一步是为了消除皮肤的影响。<br>
5. 消除毛发的影响。<br>
6. 转为灰度图。<br>
7. 提升对比度。<br>
8. 将原先的图像分割为小的图像，用一种move window的思想（类似CNN）。这是考虑到在多数情况下，残妆的出现往往是一小块的，并不会占相机区域面积的百分百。<br>
9. 过一遍efficient-v2。<br>
10. 基于小图像投票决定大图像的label，这一步具体要在js中看可以拿到什么类型的数据再做实现。<br>
11. ouput一个有无妆的label。<br>

## File
preprocessing - main function <br>
remove_hair - 去除图片中的头发 (用正常皮肤代替)，确保模型之学习到残妆的相关特征。逻辑是先用高斯的二阶导找到头发梯度，用otsu生成二值化图片再用inpaint的思路对包含二值化mask的皮肤图片进行填涂。<br>
uni_skin_color - 统一皮肤的颜色，去除皮肤的颜色特征的影响。具体逻辑是拉到Lab色彩空间用颜色的均值与方差消除差异。<br>
ab_normal_detection - 筛选正常的照片。用照片的rows进行随机检测。<br>
windows - 移动一个窗口，将大图像分割为小图像。<br>
inpaint - 填补图像。<br>
具体见代码。<br>

## Numpy Functions
np.clip()<br>
np.sum()<br>
np.zeros()<br>
np.array()<br>
np.multiply()<br>
np.ones()

## OpenCV Functions
cv2.filter2D()<br>
cv2.cvtColor()<br>
cv2.findContours()<br>
cv2.drawContours()<br>
cv2.threshold()<br>
cv2.dilate()<br>
cv2.createCLAHE()<br>
cv2.split()<br>
merge()<br>
cv2.calcHist()

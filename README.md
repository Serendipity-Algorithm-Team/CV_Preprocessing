## Introduction
此项目包含所有关于洁面仪图像预处理的代码。首先我们要明确此项目中的图像预处理的作用: <br>
1. 确保模型学习到的特征是残妆的相关特征。<br>
2. 去除因为各种原因损坏了的照片的影响。<br>
3. 能够应对更多的使用场景。

## System Design
清楚了图像处理的作用，我们基于需求设计了相关系统。<br>
![img](https://github.com/Serendipity-Algorithm-Team/CV_Preprocessing/blob/main/img/system.png)

## File
preprocessing - main function <br>
remove_hair - 去除图片中的头发 (用正常皮肤代替)，确保模型之学习到残妆的相关特征。逻辑是先用高斯的二阶导找到头发梯度，用otsu生成二值化图片再用inpaint的思路对包含二值化mask的皮肤图片进行填涂。<br>
uni_skin_color - 统一皮肤的颜色，去除皮肤的颜色特征的影响。具体逻辑是拉到Lab色彩空间用颜色的均值与方差消除差异。<br>
ab_normal_detection - 筛选正常的照片。用照片的rows进行随机检测。<br>
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

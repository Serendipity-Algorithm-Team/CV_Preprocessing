# CV_Preprocessing

## File
preprocessing - main function\\
remove_hair - 去除图片中的头发 (用正常皮肤代替)，确保模型之学习到残妆的相关特征。逻辑是先用高斯的二阶导找到头发梯度，用otsu生成二值化图片再用inpaint的思路对包含二值化mask的皮肤图片进行填涂。\\
uni_skin_color - 统一皮肤的颜色，去除皮肤的颜色特征的影响。具体逻辑是拉到Lab色彩空间用颜色的均值与方差消除差异。\\
ab_normal_detection - 筛选正常的照片。用照片的rows进行随机检测。\\
具体见代码。\\

## Numpy Functions
np.clip()\\
np.sum()\\
np.zeros()\\
np.array()\\
np.multiply()\\
np.ones()

## OpenCV Functions
cv2.filter2D()\\
cv2.cvtColor()\\
cv2.findContours()\\
cv2.drawContours()\\
cv2.threshold()\\
cv2.dilate()\\
cv2.createCLAHE()\\
cv2.split()\\
merge()\\
cv2.calcHist()

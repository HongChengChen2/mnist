import PIL.Image as Image
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt

#图片处理部分
# 颜色信息会被去除，如果要带颜色，则是三维数组
def imageToMatrix(filename):
    # 读取图片
    im = Image.open(filename)
    # 显示图片
    # im.show()
    width, height = im.size
    # 灰度化
    im = im.convert("L")
    data = im.getdata()
    data = np.matrix(data, dtype="float") / 255.0
    new_data = np.reshape(data, (height, width))
    return new_data

def matrixToImage(data):
    data = data * 255.0
    new_im = Image.fromarray(data)
    return new_im

filename = "m2.png"
data = imageToMatrix(filename)
dataframe = pd.DataFrame(data=data)
dataframe.to_csv('out.csv', sep=' ', header=False, float_format='%.2f', index=False)
#后面是将矩阵转换为图片
#new_im = matrixToImage(data)
#plt.imshow(data, cmap=plt.cm.gray, interpolation="nearest")
#new_im.show()
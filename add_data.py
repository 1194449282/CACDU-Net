# 工具类 添加噪声给数据集
import os
import random
import shutil
from glob import glob
from shutil import copy2






# 读取文件夹 获取所有图片
import cv2
import numpy as np
from PIL import Image, ImageFont, ImageDraw

def Drawing_Random_Lines(image):
    # pts = np.array([[30, 10],  [60, 45], [65, 60], [70, 68]])
    h, w = image.shape[0], image.shape[1]
    [x1, y1] = (random.randint(0, w), random.randint(0, h)) #开始点
    [x2, y2] = (random.randint(0, w), random.randint(0, h)) #结束点
    RGB = (0, 0, 0)
    #再制造一个随机点
    if(x1<x2):
        x3 = random.randint(x1, x2)
    else:
        x3 = random.randint(x2, x1)
    if (y1<y2):
        y3 = random.randint(y1, y2)
    else:
        y3 = random.randint(y2, y1)
    # pts = np.append(pts, [1, 2])
    pts = np.array([[x1, y1], [x3, y3], [x2, y2]])


    pts_fit2 = np.polyfit(pts[:, 0], pts[:, 1], 2)





    plotx = np.linspace(1, 256, 100)  # 按步长为1，设置点的x坐标
    # print(np.int_([plotx]))
    ploty2 = pts_fit2[0]*plotx**2 + pts_fit2[1]*plotx + pts_fit2[2]  #
    pts_fited2 = np.array([np.transpose(np.vstack([plotx, ploty2]))])
    # print(np.int_([pts_fited2]))
    cv2.polylines(image, np.int_([pts_fited2]), False, (0, 0, 0), 1)
    # cv2.line(image, (x1, y1), (x2, y2), RGB, 1, 8)

    # cv2.polylines(image, [pts], False, RGB, 1, 8)

    # 在图片中添加文字看上去很简单，但是如果是利用OpenCV来做却很麻烦。
    # OpenCV中并没有使用自定义字体文件的函数，这不仅意味着我们不能使用自己的字体，而且意味着他无法显示中文字符。
    # font=cv2.FONT_HERSHEY_COMPLEX


    # 将BGR转换为RGB
    image = image[..., ::-1]

    image = Image.fromarray(image)

    # image.show()
    image = np.array(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image
imagelist = glob('inputs/final_skin2018/train/images/*')

current_data_length = len(imagelist)


prelistids = round(current_data_length *0.3)

print(prelistids)
current_data_index_list= list(range(current_data_length))
random.shuffle(current_data_index_list)
current_idx = 0
for i in current_data_index_list:
    print(current_idx)
    if current_idx < 500:
        image = cv2.imread(imagelist[i])

        for k in range(10):
            image = Drawing_Random_Lines(image)
        a=imagelist[i]
        # cv2.imwrite("inputs/predict_skin2018/train/images/"+str(current_idx)+".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
        cv2.imwrite(imagelist[i], image, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
        # pathmask = os.path.basename(imagelist[i]).split('/')[0]
        # pathmask = pathmask.replace("jpg", "png")
        # pathmask = pathmask.split(".")[0]
        # print(pathmask)
        # mask = cv2.imread('inputs/predict_skin2018/train/masks/' + pathmask + '.png')
        # cv2.imwrite("inputs/predict_skin2018/train/masks/"+pathmask+".png", mask, [int(cv2.IMWRITE_JPEG_QUALITY), 100])

    current_idx = current_idx + 1

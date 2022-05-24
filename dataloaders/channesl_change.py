import matplotlib.pyplot as plt
import numpy as np
import torch
# import cv2
from PIL import Image

def get_pascal_labels():
    return np.array([[0, 1],
                     [0, 1],
                     [0, 1],  # 红：建筑
                     [0, 1],  # 黄：马路
                     [0, 1],  # 蓝：水
                     [0, 1],  # 紫色：荒地
                     [0, 1],  # 绿：森林
                     [0, 1]])  # 橙：农田   RGB格式


# img_path = r'G:\\LoveDA\\SegmentationClass\\1366.png'
# n_classes = 8
# label_mask = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)


def Channels(label_mask, batch_size):
    # 获取tensor大小
    w, h = label_mask.shape[1], label_mask.shape[2]
    b = batch_size
    n_classes = 8

    # 扩展通道数
    gray_img = label_mask.unsqueeze(dim=1)
    # print(gray_img.shape)  # [2, 1, 512, 512]

    # 初始化tensor定义全零数组c0,c1...  np.zeros==numpy数组--类型不一致
    c0 = torch.zeros(gray_img.shape)
    c1 = torch.zeros(gray_img.shape)
    c2 = torch.zeros(gray_img.shape)
    c3 = torch.zeros(gray_img.shape)
    c4 = torch.zeros(gray_img.shape)
    c5 = torch.zeros(gray_img.shape)
    c6 = torch.zeros(gray_img.shape)
    c7 = torch.zeros(gray_img.shape)


    # 复制数值
    c0.copy_(gray_img)
    c1.copy_(gray_img)
    c2.copy_(gray_img)
    c3.copy_(gray_img)
    c4.copy_(gray_img)
    c5.copy_(gray_img)
    c6.copy_(gray_img)
    c7.copy_(gray_img)

    # 改变数值  如果0通道下，标签为0，则此数值赋值1，否则为0
    label_colours = get_pascal_labels()

    for ll in range(0, n_classes):
        # 将标签为0的通道中，为0则赋值为1，其他为0（转为两种颜色）  因为输出结果的值为0，1
        if ll == 0:
            c0[gray_img == ll] = label_colours[ll, 1]
            c0[gray_img != ll] = label_colours[ll, 0]
        if ll == 1:
            c1[gray_img == ll] = label_colours[ll, 1]
            c1[gray_img != ll] = label_colours[ll, 0]
        if ll == 2:
            c2[gray_img == ll] = label_colours[ll, 1]
            c2[gray_img != ll] = label_colours[ll, 0]
        if ll == 3:
            c3[gray_img == ll] = label_colours[ll, 1]
            c3[gray_img != ll] = label_colours[ll, 0]
        if ll == 4:
            c4[gray_img == ll] = label_colours[ll, 1]
            c4[gray_img != ll] = label_colours[ll, 0]
        if ll == 5:
            c5[gray_img == ll] = label_colours[ll, 1]
            c5[gray_img != ll] = label_colours[ll, 0]
        if ll == 6:
            c6[gray_img == ll] = label_colours[ll, 0]
            c6[gray_img != ll] = label_colours[ll, 0]
        if ll == 7:
            c7[gray_img == ll] = label_colours[ll, 0]
            c7[gray_img != ll] = label_colours[ll, 0]

    # 初始化定义一个8通道的tensor
    mask = torch.zeros(((b, n_classes, w, h)))

    # 将值concat起来，使之变成8通道
    mask = torch.cat((c0,c1,c2,c3,c4,c5,c6,c7), dim=1)

    return mask



"""第二版：  copy的方法要注意数据格式问题，tensor(torch),image,numpy
def Channels(label_mask):
    # gray_img = img.convert('L')
    w, h = label_mask.shape[1], label_mask.shape[2]
    # 将每个通道的图片都进行处理：为甚batch_size = 5,通道数便是5
    # 原本是灰色图（r=g=b），现在将r,g,b的颜色分别涂上标签的像素值————先copy然后对应位置上色

    # tensor转image
    label_mask = label_mask.numpy()
    # 当 image_numpy.shape[0]=1，即 image_numpy 是灰度图像，只要将 image_numpy = image_numpy[0]，其格式就从 [1, h, w] 改变为 [h, w] 了
    if label_mask.shape[0] == 3:
        label_mask = (np.transpose(label_mask, (1, 2, 0)) + 1) / 2.0 * 255.0
    elif label_mask.shape[0] == 1:
        label_mask = (label_mask[0] + 1) / 2.0 * 255
    gray_img = Image.fromarray(label_mask)

    c0 = gray_img.copy()  # (512, 512)
    c1 = gray_img.copy()
    c2 = gray_img.copy()
    c3 = gray_img.copy()
    c4 = gray_img.copy()
    c5 = gray_img.copy()
    c6 = gray_img.copy()
    c7 = gray_img.copy()

    label_colours = get_pascal_labels()

    for ll in range(0, n_classes):  # RGB对应位置上色
        # 将灰度图中标签为0/1/2/3/4/5/6/7的值换成想对应的标签0/1/2/3/4/5/6/7所对应的像素值
        # 将标签为0的通道中，为0则赋值为1，其他为0（转为两种颜色）  因为输出结果的值为0，1
        # for i in range(w):
        #     for j in range(h):
        #         if c0[i][j] == ll:
        #             c0[i][j] = 1
        #         else:
        #             c0[i][j] = 0
        if ll == 0:
            c0[gray_img == ll] = label_colours[ll, 1]
            c0[gray_img != ll] = label_colours[ll, 0]
        if ll == 1:
            c1[gray_img == ll] = label_colours[ll, 1]
            c1[gray_img != ll] = label_colours[ll, 0]
        if ll == 2:
            c2[gray_img == ll] = label_colours[ll, 1]
            c2[gray_img != ll] = label_colours[ll, 0]
        if ll == 3:
            c3[gray_img == ll] = label_colours[ll, 1]
            c3[gray_img != ll] = label_colours[ll, 0]
        if ll == 4:
            c4[gray_img == ll] = label_colours[ll, 1]
            c4[gray_img != ll] = label_colours[ll, 0]
        if ll == 5:
            c5[gray_img == ll] = label_colours[ll, 1]
            c5[gray_img != ll] = label_colours[ll, 0]
        if ll == 6:
            c6[gray_img == ll] = label_colours[ll, 0]
            c6[gray_img != ll] = label_colours[ll, 0]
        if ll == 7:
            c7[gray_img == ll] = label_colours[ll, 0]
            c7[gray_img != ll] = label_colours[ll, 0]

        # 0号标签：[255, 255, 255]白色--> [0,0]=255  [0,1]=255  [0,2]=255
    # 定义一个为0的三通道数组,组合为BGR
    mask = np.zeros((gray_img.shape[0], gray_img.shape[1], 8))  # （513，513，3）
    mask[:, :,
    0] = c0  # rgb[:, :, 0]代表的第一个通道 could not broadcast input array from shape (513,513,3) into shape (513,513)
    mask[:, :, 1] = c1
    mask[:, :, 2] = c2
    mask[:, :, 3] = c3
    mask[:, :, 4] = c4
    mask[:, :, 5] = c5
    mask[:, :, 6] = c6
    mask[:, :, 7] = c7

    # image转tensor
    transform = torchvision.transforms.Compose([
        transforms.ToTensor()])
    mask_i = transform(mask)

    return mask_i
"""

"""原本：
def get_pascal_labels():
    return np.array([[0, 1],
                     [0, 1],
                     [0, 1],  # 红：建筑
                     [0, 1],  # 黄：马路
                     [0, 1],  # 蓝：水
                     [0, 1],  # 紫色：荒地
                     [0, 1],  # 绿：森林
                     [0, 1]])  # 橙：农田   RGB格式

img_path = r'G:\\LoveDA\\SegmentationClass\\1366.png'
n_classes = 8
label_mask = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

w, h = label_mask.shape[0], label_mask.shape[1]
# 将每个通道的图片都进行处理：为甚batch_size = 5,通道数便是5
# 原本是灰色图（r=g=b），现在将r,g,b的颜色分别涂上标签的像素值————先copy然后对应位置上色
c0 = label_mask.copy()  # (512, 512)
c1 = label_mask.copy()
c2 = label_mask.copy()
c3 = label_mask.copy()
c4 = label_mask.copy()
c5 = label_mask.copy()
c6 = label_mask.copy()
c7 = label_mask.copy()

label_colours = get_pascal_labels()

for ll in range(0, n_classes):  # RGB对应位置上色
    # 将灰度图中标签为0/1/2/3/4/5/6/7的值换成想对应的标签0/1/2/3/4/5/6/7所对应的像素值
    # 将标签为0的通道中，为0则赋值为1，其他为0（转为两种颜色）  因为输出结果的值为0，1
    # for i in range(w):
    #     for j in range(h):
    #         if c0[i][j] == ll:
    #             c0[i][j] = 1
    #         else:
    #             c0[i][j] = 0
    if ll == 0:
        c0[label_mask == ll] = label_colours[ll, 1]
        c0[label_mask != ll] = label_colours[ll, 0]
    if ll == 1:
        c1[label_mask == ll] = label_colours[ll, 1]
        c1[label_mask != ll] = label_colours[ll, 0]
    if ll == 2:
        c2[label_mask == ll] = label_colours[ll, 1]
        c2[label_mask != ll] = label_colours[ll, 0]
    if ll == 3:
        c3[label_mask == ll] = label_colours[ll, 1]
        c3[label_mask != ll] = label_colours[ll, 0]
    if ll == 4:
        c4[label_mask == ll] = label_colours[ll, 1]
        c4[label_mask != ll] = label_colours[ll, 0]
    if ll == 5:
        c5[label_mask == ll] = label_colours[ll, 1]
        c5[label_mask != ll] = label_colours[ll, 0]
    if ll == 6:
        c6[label_mask == ll] = label_colours[ll, 0]
        c6[label_mask != ll] = label_colours[ll, 0]
    if ll == 7:
        c7[label_mask == ll] = label_colours[ll, 0]
        c7[label_mask != ll] = label_colours[ll, 0]

    # 0号标签：[255, 255, 255]白色--> [0,0]=255  [0,1]=255  [0,2]=255
# 定义一个为0的三通道数组,组合为BGR
mask = np.zeros((label_mask.shape[0], label_mask.shape[1], 8))  # （513，513，3）
mask[:, :, 0] = c0   # rgb[:, :, 0]代表的第一个通道 could not broadcast input array from shape (513,513,3) into shape (513,513)
mask[:, :, 1] = c1
mask[:, :, 2] = c2
mask[:, :, 3] = c3
mask[:, :, 4] = c4
mask[:, :, 5] = c5
mask[:, :, 6] = c6
mask[:, :, 7] = c7
"""
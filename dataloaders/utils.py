import matplotlib.pyplot as plt
import numpy as np
import torch

def decode_seg_map_sequence(label_masks, dataset='pascal'):
    rgb_masks = []
    for label_mask in label_masks:
        rgb_mask = decode_segmap(label_mask, dataset)
        rgb_masks.append(rgb_mask)
    rgb_masks = torch.from_numpy(np.array(rgb_masks).transpose([0, 3, 1, 2]))
    return rgb_masks


def decode_segmap(label_mask, dataset, plot=False):
    """Decode segmentation class labels into a color image
    Args:
        label_mask (np.ndarray): an (M,N) array of integer values denoting
          the class label at each spatial location.
        plot (bool, optional): whether to show the resulting color image
          in a figure.
    Returns:
        (np.ndarray, optional): the resulting decoded color image.
    """
    if dataset == 'pascal' or dataset == 'coco':
        # n_classes = 21
        n_classes = 6
        label_colours = get_pascal_labels()  # 得到标签上色的像素值 (8, 3)
        # [255, 255, 255]白色, [255, 0, 0]红色,
        # [255, 255, 0]黄色,  [0, 0, 255]蓝色,
        # [159, 129, 183]紫色, [0, 255, 0]绿色,
        # [255, 195, 128]橙色, [0, 0, 0]黑色
    elif dataset == 'cityscapes':
        n_classes = 19
        label_colours = get_cityscapes_labels()
    else:
        raise NotImplementedError

    # 将每个通道的图片都进行处理：为甚batch_size = 5,通道数便是5
    # 原本是灰色图（r=g=b），现在将r,g,b的颜色分别涂上标签的像素值————先copy然后对应位置上色
    r = label_mask.copy()  # (512, 512)
    g = label_mask.copy()
    b = label_mask.copy()
    for ll in range(0, n_classes):  # RGB对应位置上色
        # 将灰度图中标签为0/1/2/3/4/5/6/7的值换成想对应的标签0/1/2/3/4/5/6/7所对应的像素值
        r[label_mask == ll] = label_colours[ll, 0]
        g[label_mask == ll] = label_colours[ll, 1]
        b[label_mask == ll] = label_colours[ll, 2]
        # 0号标签：[255, 255, 255]白色--> [0,0]=255  [0,1]=255  [0,2]=255
    # 定义一个为0的三通道数组,组合为BGR
    rgb = np.zeros((label_mask.shape[0], label_mask.shape[1], 3))  # （513，513，3）
    rgb[:, :, 0] = r / 255.0   # rgb[:, :, 0]代表的第一个通道 could not broadcast input array from shape (513,513,3) into shape (513,513)
    rgb[:, :, 1] = g / 255.0
    rgb[:, :, 2] = b / 255.0
    # 定义一个为0的三通道数组,组合为RGB
    # rgb = np.zeros((3, label_mask.shape[0], label_mask.shape[1]))  # （3, 513，513）
    # rgb[0, :, :] = r / 255.0  # rgb[:, :, 0]代表的第一个通道 could not broadcast input array from shape (513,513,3) into shape (513,513)
    # rgb[1, :, :] = g / 255.0
    # rgb[2, :, :] = b / 255.0
    # rgb的值是0,1 黑白图片：只有当像素值为255的时候才为1
    if plot:
        plt.imshow(rgb)
        plt.show()
    else:
        return rgb


def encode_segmap(mask):
    """Encode segmentation label images as pascal classes
    Args:
        mask (np.ndarray): raw segmentation label image of dimension
          (M, N, 3), in which the Pascal classes are encoded as colours.
    Returns:
        (np.ndarray): class map with dimensions (M,N), where the value at
        a given location is the integer denoting the class index.
    """
    mask = mask.astype(int)
    label_mask = np.zeros((mask.shape[0], mask.shape[1]), dtype=np.int16)
    for ii, label in enumerate(get_pascal_labels()):
        label_mask[np.where(np.all(mask == label, axis=-1))[:2]] = ii
    label_mask = label_mask.astype(int)
    return label_mask


def get_cityscapes_labels():
    return np.array([
        [128, 64, 128],
        [244, 35, 232],
        [70, 70, 70],
        [102, 102, 156],
        [190, 153, 153],
        [153, 153, 153],
        [250, 170, 30],
        [220, 220, 0],
        [107, 142, 35],
        [152, 251, 152],
        [0, 130, 180],
        [220, 20, 60],
        [255, 0, 0],
        [0, 0, 142],
        [0, 0, 70],
        [0, 60, 100],
        [0, 80, 100],
        [0, 0, 230],
        [119, 11, 32]])


def get_pascal_labels():
    """Load the mapping that associates pascal classes with label colors
    Returns:
        np.ndarray with dimensions (21, 3)
    """
    # return np.asarray([[0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0],
    #                    [0, 0, 128], [128, 0, 128], [0, 128, 128], [128, 128, 128],
    #                    [64, 0, 0], [192, 0, 0], [64, 128, 0], [192, 128, 0],
    #                    [64, 0, 128], [192, 0, 128], [64, 128, 128], [192, 128, 128],
    #                    [0, 64, 0], [128, 64, 0], [0, 192, 0], [128, 192, 0],
    #                    [0, 64, 128]])

    # return np.array([[255,255,255],[255,0,0],[255,255,0],
    #                 [0,0,255],[159,129,183],[0,255,0],
    #                 [255,195,128]], dtype=np.uint8)             #类别标签

    # 由于opencv读入的是BGR所以顺序都要调换  顺序是如下的：1，2，3，4，5，6，7
    # IGNORE = (0, 0, 0),  # 黑色
    # Background = (255, 255, 255),  # 白色
    # Building = (0, 0, 255),  # 红色(255, 0, 0)
    # Road = (0, 255, 255),  # 黄色(255, 255, 0)
    # Forest = (0, 255, 0),  # 绿色
    # Water = (255, 0, 0),  # 蓝色(0, 0, 255)
    # Barren = (183, 129, 159),  # 紫色(159, 129, 183)
    # Agricultural = (128, 195, 255),  # 橙色(255, 195, 128)

    # return np.array([[0, 0, 0],
    #                  [255, 255, 255],
    #                  [255, 0, 0],  # 红：建筑
    #                  [255, 255, 0],  # 黄：马路
    #                  [0, 0, 255],  # 蓝：水
    #                  [159, 129, 183],  # 紫色：荒地
    #                  [0, 255, 0],  # 绿：森林
    #                  [255, 195, 128]])  # 橙：农田   RGB格式

    # return np.array([[0, 0, 0],
    #                  [255, 255, 255],  # 白：背景
    #                  [0, 0, 255],   # 红：建筑
    #                  [0, 255, 255],  # 黄：马路
    #                  [255, 0, 0],  # 蓝：水
    #                  [183, 129, 159],  # 紫：荒地
    #                  [0, 255, 0],  # 绿：森林
    #                  [128, 195, 255]])   # 橙：农田 # BGR格式

    return np.array([
        [255, 255, 255],  # 白色：道路
        [0, 0, 255],  # 深蓝色：建筑
        [0, 255, 255],  # 浅蓝色：植被
        [0, 255, 0],  # 绿色：树
        [255, 255, 0],  # 黄色：移动汽车
        [255, 0, 0]  # 红色：乱堆
    ])  # RGB

    # return np.array([
    #         [255, 255, 255],  # 白色：道路
    #         [255, 0, 0],  # 深蓝色：建筑
    #         [255, 255, 0],  # 浅蓝色：植被
    #         [0, 255, 0],  # 绿色：树
    #         [0, 255, 255],  # 黄色：移动汽车
    #         [0, 0, 255]  # 红色：乱堆
    #     ])  # BGR

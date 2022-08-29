from __future__ import print_function, division
import os
from PIL import Image
import numpy as np
from torch.utils.data import Dataset
from train.mypath import Path
from torchvision import transforms
from dataloaders import custom_transforms as tr

class VOCSegmentation(Dataset):
    """
    PascalVoc dataset
    """
    #NUM_CLASSES = 21
    NUM_CLASSES = 6

    def __init__(self,
                 args,
                 base_dir=Path.db_root_dir('pascal'),  # 在mypath.py里面有设置，为数据集的路径 'G:\\LoveDA\\'
                 split='train',
                 ):
        """
        :param base_dir: path to VOC dataset directory
        :param split: train/val
        :param transform: transform to apply
        """
        super().__init__()
        self._base_dir = base_dir  # 数据集的根目录 'G:\\LoveDA\\'
        self._image_dir = os.path.join(self._base_dir, 'JPEGImages')  # 'G:\\LoveDA\\JPEGImages'
        self._cat_dir = os.path.join(self._base_dir, 'SegmentationClass')  # 'G:\\LoveDA\\SegmentationClass'

        if isinstance(split, str):  # split为‘train’训练集，'val'验证集  isinstance（）：判断两个类型是否相同
            self.split = [split]
        else:
            split.sort()  # 对列表进行原址排序
            self.split = split

        self.args = args

        _splits_dir = os.path.join(self._base_dir, 'ImageSets', 'Segmentation')  # 存放着标签名称的文件夹路径
        # 'G:\\LoveDA\\ImageSets\\Segmentation'

        self.im_ids = []  # 存标签名的数组
        self.images = []  # 存图像地址的数组
        self.categories = []  # 存标签图片地址的数组

        for splt in self.split:  # 将图片信息读取到数组中
            with open(os.path.join(os.path.join(_splits_dir, splt + '.txt')), "r") as f:  # 'G:\\LoveDA\\ImageSets\\Segmentation\\train.txt'
                lines = f.read().splitlines()  # f.readlines()后面有加\n,f.read().splitlines()没有\n

            for ii, line in enumerate(lines):  # (ii=0,line='1366')  enumerate():返回 enumerate(枚举) 对象
                _image = os.path.join(self._image_dir, line + ".jpg")  # 'G:\\LoveDA\\JPEGImages\\1366.jpg'
                _cat = os.path.join(self._cat_dir, line + ".png")  # 'G:\\LoveDA\\SegmentationClass\\1366.png'
                # assert检查条件，不符合就终止程序
                assert os.path.isfile(_image)  # os.path.isfile 用于判断某一对象(需提供绝对路径)是否为文件
                assert os.path.isfile(_cat)
                self.im_ids.append(line)  # 存入标签名  1366
                self.images.append(_image)  # 存入图像地址  'G:\\LoveDA\\JPEGImages\\1366.jpg'
                self.categories.append(_cat)  # 存入标签图片地址  'G:\\LoveDA\\SegmentationClass\\1366.png'

        assert (len(self.images) == len(self.categories))  # 判断图像个数和标签数量是否一致

        # Display stats
        print('Number of images in {}: {:d}'.format(split, len(self.images)))  # 1156

    def __len__(self):
        return len(self.images)


    def __getitem__(self, index):
        _img, _target = self._make_img_gt_point_pair(index)  # 读取图像 img:RGB形式  (1024, 1024)
        sample = {'image': _img, 'label': _target}  # {}字典数据类型，每一个键值对

        for split in self.split:
            if split == "train":
                return self.transform_tr(sample)  # 返回训练集的图像处理
            elif split == 'val':
                return self.transform_val(sample)   # 返回验证集的图像处理


    def _make_img_gt_point_pair(self, index): # 仅仅是通过这个读就已经转成了0-7的标签图
        # Image.open()函数只是保持了图像被读取的状态，但是图像的真实数据并未被读取
        # Image.open()函数默认彩色图像读取通道的顺序为RGB
        # 如果不使用.convert(‘RGB’)进行转换的话，读出来的图像是RGBA四通道的，A通道为透明通道，该对深度学习模型训练来说暂时用不到，因此使用convert(‘RGB’)进行通道转换
        _img = Image.open(self.images[index]).convert('RGB')  # 原图像    RGB: 3x8位像素，真彩色
        _target = Image.open(self.categories[index])  # 标注图像

        # _img.show()
        # _target.show()

        return _img, _target

    def transform_tr(self, sample):   # 返回训练集的图像处理：翻转-切割-平滑-标准化-tensor形式保存  大小控制在[-1.5, 0.5] 区间为2
        composed_transforms = transforms.Compose([
            tr.RandomHorizontalFlip(),  # 以给定的概率随机水平翻转给定的PIL图像
            tr.RandomScaleCrop(base_size=self.args.base_size, crop_size=self.args.crop_size),  # 将原图片进行切割随机范围内裁剪
            tr.RandomGaussianBlur(),  # 用高斯滤波器（GaussianFilter）对图像进行平滑处理
            tr.Color(),  # 新增的一个函数，用于对遥感图像进行色彩的处理
            tr.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),  # 让数值大小变得可以比较
            tr.ToTensor()])  # tensor中是以 (c, h, w) 的格式来存储图片的

        return composed_transforms(sample)

    def transform_val(self, sample):  # 验证集的图片处理：切割-标准化-tensor形式保存 大小控制在[0，  7] 区间为8 因为图片的类别数为8

        composed_transforms = transforms.Compose([
            tr.FixScaleCrop(crop_size=self.args.crop_size),
            tr.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            tr.ToTensor()])

        return composed_transforms(sample)

    def __str__(self):
        return 'VOC2012(split=' + str(self.split) + ')'


if __name__ == '__main__':
    from dataloaders.utils import decode_segmap
    from torch.utils.data import DataLoader
    import matplotlib.pyplot as plt
    import argparse

    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    args.base_size = 513
    args.crop_size = 513

    voc_train = VOCSegmentation(args, split='train')    # 训练集的原图像

    dataloader = DataLoader(voc_train, batch_size=5, shuffle=True, num_workers=0)  # 只有主进程去加载batch数据

    for ii, sample in enumerate(dataloader):
        for jj in range(sample["image"].size()[0]):  # 字典sample  jj:0~5 5个
            img = sample['image'].numpy()  # (5, 3, 513, 513) 彩色 数值大小在[-1.5, 0.5]之间 [[[[-1.1075435  -1.1760424  -1.1246682  ...
            gt = sample['label'].numpy()   # 一张图： (5, 513, 513) float型   灰度图，5个通道，然后每一个都是（513，513）：r=g=b    错：掩码应该是单通道的，不应该上彩色
            plt.show()
            tmp = np.array(gt[jj]).astype(np.uint8)  # uint8无符号八位整数，数字范围[0,255]  unit8型 (513, 513) tmp是二维的 [[1 1 1 ... 0 0 0]
            segmap = decode_segmap(tmp, dataset='pascal')   # [513, 513，3]  绘制成彩色 [[[1. 0. 0.],
            img_tmp = np.transpose(img[jj], axes=[1, 2, 0])  # 黑白色 [513, 513，3]  np.transpose：求矩阵的转置 [[[-1.1075435 -0.897759 -0.688976 ],
            # 归一标准化的逆操作：恢复0-255之间的彩色值
            img_tmp *= (0.229, 0.224, 0.225)  # 在图像送入网络训练之前，减去图片的均值，算是一种归一化操作。
            img_tmp += (0.485, 0.456, 0.406)  # 图像其实是一种平稳的分布，减去数据对应维度的统计平均值，可以消除公共部分。
            img_tmp *= 255.0    # 以凸显个体之间的差异和特征。
            img_tmp = img_tmp.astype(np.uint8)  # 无符号八位整数，数字范围[0,255]
            plt.figure()
            plt.title('display')
            plt.subplot(211)
            plt.imshow(img_tmp)  # 显示原图像
            plt.subplot(212)
            plt.imshow(segmap)  # 显示彩色
            # print(segmap)

        if ii == 1:
            break

    plt.show(block=True)



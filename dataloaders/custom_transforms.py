import torch
import random
import numpy as np
# import tensorflow as tf
from PIL import Image, ImageOps, ImageFilter
import albumentations as A

class Normalize(object):
    """Normalize a tensor image with mean and standard deviation.
    Args:
        mean (tuple): means for each channel.
        std (tuple): standard deviations for each channel.
    """
    def __init__(self, mean=(0., 0., 0.), std=(1., 1., 1.)): # 当没有值的时候 mean std默认值
        self.mean = mean  # (0.485, 0.456, 0.406)
        self.std = std  # (0.229, 0.224, 0.225)

    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']
        img = np.array(img).astype(np.float32)
        mask = np.array(mask).astype(np.float32)
        img /= 255.0   # 图片保存都是0~255的数值范围  将数值大小降到[0, 1]
        img -= self.mean  # [0, 0.5]
        img /= self.std   # [1, 2]

        return {'image': img,
                'label': mask}


class ToTensor(object):  # 图像从 np.array 转换为 tensor
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        img = sample['image']
        mask = sample['label']
        img = np.array(img).astype(np.float32).transpose((2, 0, 1))   # 将图片的维度进行转换 在 tensor 中是以 (c, h, w) 的格式来存储图片的。
        mask = np.array(mask).astype(np.float32)

        img = torch.from_numpy(img).float()
        mask = torch.from_numpy(mask).float()

        return {'image': img,
                'label': mask}


class RandomHorizontalFlip(object):
    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']
        if random.random() < 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.FLIP_LEFT_RIGHT)

        return {'image': img,
                'label': mask}


class RandomRotate(object):
    def __init__(self, degree):
        self.degree = degree

    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']
        rotate_degree = random.uniform(-1*self.degree, self.degree)
        img = img.rotate(rotate_degree, Image.BILINEAR)
        mask = mask.rotate(rotate_degree, Image.NEAREST)

        return {'image': img,
                'label': mask}


class RandomGaussianBlur(object):
    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']
        if random.random() < 0.5:
            img = img.filter(ImageFilter.GaussianBlur(
                radius=random.random()))

        return {'image': img,
                'label': mask}


class RandomScaleCrop(object):  # 随机裁剪
    def __init__(self, base_size, crop_size, fill=0):
        self.base_size = base_size  # 512
        self.crop_size = crop_size  # 512
        self.fill = fill  # 0

    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']
        # random scale (short edge)
        short_size = random.randint(int(self.base_size * 1.0), int(self.base_size * 2.0))  # 返回 [256, 1024] 之间的任意整数  644
        w, h = img.size  # h:1024  w:1024 原图的尺寸大小
        if h > w:
            ow = short_size  # ow：644
            oh = int(1.0 * h * ow / w)  # oh:644
        else:
            oh = short_size  # oh 307
            ow = int(1.0 * w * oh / h)  # w = h so ow = oh = 644
        # resize函数来改变图像的大小
        # Image.NEAREST ：低质量  最邻近插值
        # Image.BILINEAR：双线性
        # Image.BICUBIC ：三次样条插值
        # Image.ANTIALIAS：高质量
        img = img.resize((ow, oh), Image.BILINEAR)  # 利用双线性插值公式的方法来进行图像的缩放
        mask = mask.resize((ow, oh), Image.NEAREST)

        # pad crop 填充
        if short_size < self.crop_size:  # 随机裁剪大小 小于 目标裁剪大小时
            padh = self.crop_size - oh if oh < self.crop_size else 0  # 513-307=206
            padw = self.crop_size - ow if ow < self.crop_size else 0
            img = ImageOps.expand(img, border=(0, 0, padw, padh), fill=0)  # ImageOps.expand在要调用或使用此函数的图像上添加边框  fill=0默认值为0，表示颜色为黑色
            # (513, 513) border的len为4，left, top, right, bottom右边和底部进行填充：0 黑色的像素值 然后图片大小变成（513，513）
            mask = ImageOps.expand(mask, border=(0, 0, padw, padh), fill=self.fill)
        # random crop crop_size
        w, h = img.size  # 随机裁剪的大小 （644，644）
        x1 = random.randint(0, w - self.crop_size)  # x1:44  (0, 644-513 = 131)之间的任意整数
        y1 = random.randint(0, h - self.crop_size)  # y1:34  (0,131)之间的任意整数
        img = img.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))   # 做切割 (513, 513)  左上右下<坐标轴>(44, 34, 557, 547)
        mask = mask.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))

        return {'image': img,
                'label': mask}


class FixScaleCrop(object):
    def __init__(self, crop_size):
        self.crop_size = crop_size

    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']
        w, h = img.size  # h:1024  w:1024 原图的尺寸大小
        if w > h:
            oh = self.crop_size
            ow = int(1.0 * w * oh / h)
        else:
            ow = self.crop_size  # 513
            oh = int(1.0 * h * ow / w)  # oh = ow = 513
        img = img.resize((ow, oh), Image.BILINEAR)  # 利用双线性插值公式的方法来进行图像的缩放(513,513)
        mask = mask.resize((ow, oh), Image.NEAREST)  # 利用最近邻插值公式的方法来进行图像的缩放   srcX=dstX*(srcWidth/dstWidth)  srcY=dstY*(srcHeight/dstHeight)
        # center crop
        w, h = img.size  # (513,513)
        x1 = int(round((w - self.crop_size) / 2.))  # x1=0
        y1 = int(round((h - self.crop_size) / 2.))  # y1=0
        img = img.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))  #(513, 513)  (0,0,513,513)
        mask = mask.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))

        return {'image': img,
                'label': mask}

class FixedResize(object):
    def __init__(self, size):
        self.size = (size, size)  # size: (h, w)

    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']

        assert img.size == mask.size

        img = img.resize(self.size, Image.BILINEAR)
        mask = mask.resize(self.size, Image.NEAREST)

        return {'image': img,
                'label': mask}


#   ***********************   新增  ********************    #

# 和图像翻转类似，调整图像的亮度、对比度、饱和度和色相在很多图像识别应用中都不会影响识别的结果。
# 所以在训练神经网络模型时，可以随机调整训练图像的这些属性，从而使得到的模型尽可能小地受到无关因素的影响。

class Color(object):  # 调整亮度、对比度、锐度
    def __init__(self):
        self.p = random.random()
        self.threshold = 0.3

    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']

        # 当随机值达到阈值时，才对数据进行色彩调整
        # if self.p > self.threshold:
        # 使用一种随机的顺序调整图像色彩。OneOf组合里面的变换是系统自动选择其中一个来做，而这里的概率参数p是指选定后的变换被做的概率
        # A.OneOf([
        #     # A.RandomContrast(limit=0.2, always_apply=False, p=0.5),  # 对比度
        #     # A.RandomBrightness(limit=0.2, always_apply=False, p=0.5),  # 亮度
        #     A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, brightness_by_max=True, always_apply=False, p=0.5),  # 这一个包含前面两个
        #     A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, always_apply=False, p=0.5)  # 随机更改输入图像的色调，饱和度和值
        # ], p=0.8)
        # 这里有2个增强方法，系统假设选择做A.HueSaturationValue，那么做A.HueSaturationValue的概率就是0.2。
        A.Compose([
            # A.RandomContrast(limit=0.2, always_apply=False, p=0.5),  # 对比度
            # A.RandomBrightness(limit=0.2, always_apply=False, p=0.5),  # 亮度
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, brightness_by_max=True,
                                       always_apply=False, p=0.5),  # 这一个包含前面两个
            A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, always_apply=False, p=0.5)
            # 随机更改输入图像的色调，饱和度和值
        ], p=0.8)

        return {'image': img,
                'label': mask}
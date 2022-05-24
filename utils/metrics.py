import numpy as np

# 包含全部分割指标的类
class Evaluator(object):
    def __init__(self, num_class):
        self.num_class = num_class  # 8
        self.confusion_matrix = np.zeros((self.num_class,)*2)  # (8, 8)
        # 用法：zeros(shape, dtype=float, order='C')
        # c = np.zeros((8,8)*2)->(8, 8, 8, 8) *** b = np.zeros((8)*2)->(16,)一行16列 *** a = np.zeros((8,)*2)->(8, 8)8行8列

    # 像素准确率 PA
    def Pixel_Accuracy(self):  # 所有分类正确的像素数占全部像素的比例
        Acc = np.diag(self.confusion_matrix).sum() / self.confusion_matrix.sum()  # np.diag(self.confusion_matrix).sum() 分类正确（对角线）
        return Acc

    # 平均像素准确率 MPA
    def Pixel_Accuracy_Class(self):  # 分别计算每个类别分类正确的像素数占所有预测为该类别像素数比例的平均值
        Acc = np.diag(self.confusion_matrix) / self.confusion_matrix.sum(axis=1)  # 1:行，0：列
        # b = np.array([[0,0,0,2],[0,0,2,1],[1,1,1,0],[1,0,1,2]])
        # np.sum(b, axis=1)
        Acc = np.nanmean(Acc)
        return Acc

    # 平均Iou MIoU均交并比 mIOU=TP/(FP+FN+TP) 真实值和预测值两个集合的交集和并集之比
    def Mean_Intersection_over_Union(self):
        MIoU = np.diag(self.confusion_matrix) / (  # np.diag(self.confusion_matrix)：预测为真的总和（对角线）：预测值和真实值的交集
                    np.sum(self.confusion_matrix, axis=1) + np.sum(self.confusion_matrix, axis=0) -
                    np.diag(self.confusion_matrix))  # 分母为混淆矩阵的和：预测值和真实值的并集
        # pij、pji则分别表示假正和假负
        MIoU = np.nanmean(MIoU)  # 平均
        return MIoU

    # 频权IOu FWIoU
    def Frequency_Weighted_Intersection_over_Union(self):
        freq = np.sum(self.confusion_matrix, axis=1) / np.sum(self.confusion_matrix)
        iu = np.diag(self.confusion_matrix) / (
                    np.sum(self.confusion_matrix, axis=1) + np.sum(self.confusion_matrix, axis=0) -
                    np.diag(self.confusion_matrix))

        FWIoU = (freq[freq > 0] * iu[freq > 0]).sum()
        return FWIoU

    # 生成混淆矩阵
    def _generate_matrix(self, gt_image, pre_image):
        mask = (gt_image >= 0) & (gt_image < self.num_class)
        label = self.num_class * gt_image[mask].astype('int') + pre_image[mask]
        count = np.bincount(label, minlength=self.num_class**2)
        confusion_matrix = count.reshape(self.num_class, self.num_class)
        return confusion_matrix

    # 为真值和预测值生成混淆矩阵
    def add_batch(self, gt_image, pre_image):
        assert gt_image.shape == pre_image.shape
        self.confusion_matrix += self._generate_matrix(gt_image, pre_image)

    # 重置混淆矩阵
    def reset(self):
        self.confusion_matrix = np.zeros((self.num_class,) * 2)
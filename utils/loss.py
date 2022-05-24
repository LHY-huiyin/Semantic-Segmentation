import torch
import torch.nn as nn
import torch.nn.functional as F


class SegmentationLosses(object):
    def __init__(self, weight=None, size_average=True, batch_average=True, ignore_index=255, cuda=False):
        self.ignore_index = ignore_index  # 255
        self.weight = weight  # None
        self.size_average = size_average  # true
        self.batch_average = batch_average  # true
        self.cuda = cuda  # true

    def build_loss(self, mode='ce'):
        """Choices: ['ce' or 'focal']"""
        if mode == 'ce':
            return self.CrossEntropyLoss  # 交叉熵损失函数
        elif mode == 'focal':
            return self.FocalLoss
        else:
            raise NotImplementedError

    def CrossEntropyLoss(self, logit, target):  # logit:[1,3,7,7]  target:[1,7,7]
        n, c, h, w = logit.size()  # n:batch:1 c:channel:3
        criterion = nn.CrossEntropyLoss(weight=self.weight, ignore_index=self.ignore_index,
                                        size_average=self.size_average)  # ignore_index:计算交叉熵时，自动忽略的标签值，
        if self.cuda:
            criterion = criterion.cuda()

        loss = criterion(logit, target.long())  # 交叉熵函数
        # print(loss.shape)
        if self.batch_average:
            loss /= n

        return loss

    def FocalLoss(self, logit, target, gamma=2, alpha=0.5):
        n, c, h, w = logit.size()
        criterion = nn.CrossEntropyLoss(weight=self.weight, ignore_index=self.ignore_index,
                                        size_average=self.size_average)
        if self.cuda:
            criterion = criterion.cuda()

        logpt = -criterion(logit, target.long())
        pt = torch.exp(logpt)
        if alpha is not None:
            logpt *= alpha
        loss = -((1 - pt) ** gamma) * logpt

        if self.batch_average:
            loss /= n

        return loss


class ECELoss(nn.Module):
    def __init__(self,  n_classes=8, alpha=1, radius=1, beta=0.5, ignore_lb=255, mode='ohem', batch_average=True, *args, **kwargs):
        super(ECELoss, self).__init__()
        self.ignore_lb = ignore_lb
        self.n_classes = n_classes
        self.alpha = alpha
        self.radius = radius
        self.beta = beta
        self.batch_average = batch_average
        if mode == 'ce':
            self.criteria = nn.CrossEntropyLoss(ignore_index=ignore_lb)
        else:
            raise Exception('No %s loss, plase choose form ohem and ce' % mode)

        self.edge_criteria = EdgeLoss(self.n_classes, self.radius, self.alpha)


    def forward(self, logits, target):
        n, c, h, w = logits.size()  # n:4 c:8 h=256 w=256
        if self.beta > 0:
            # 函数输入为long类型
            loss = self.criteria(logits, target.long()) + self.beta*self.edge_criteria(logits, target.long())
        else:
            loss = self.criteria(logits, target)

        # loss = criterion(logits, target.long())  # 交叉熵函数
        # print(loss.shape)

        if self.batch_average:
            loss /= n

        return loss

class EdgeLoss(nn.Module):
    def __init__(self, n_classes=19, radius=1, alpha=0.01):
        super(EdgeLoss, self).__init__()
        self.n_classes = n_classes
        self.radius = radius
        self.alpha = alpha


    def forward(self, logits, label):
        prediction = F.softmax(logits, dim=1)  # 取最大值 [4, 8, 256, 256]
        ks = 2 * self.radius + 1  # 3
        filt1 = torch.ones(1, 1, ks, ks)  # 生成一个全1矩阵[1,1,3,3]
        filt1[:, :, self.radius:2*self.radius, self.radius:2*self.radius] = -8  # 赋值：在第三维和第四维（三行三列）上的第二行第二列的值赋值为-8
        filt1.requires_grad = False
        filt1 = filt1.cuda()
        label = label.unsqueeze(1)  # 增加一维，由[4,256,256]变成[4,1,256,256]
        lbedge = F.conv2d(label.float(), filt1, bias=None, stride=1, padding=self.radius)  # 进行卷积，filt1相当于权重
        lbedge = 1 - torch.eq(lbedge, 0).float()  # 对两个张量Tensor进行逐元素的比较，若相同位置的两个元素相同，则返回True；若不同，返回False  0，1

        filt2 = torch.ones(self.n_classes, 1, ks, ks)  # 生成一个全1矩阵[8,1,3,3]
        filt2[:, :, self.radius:2*self.radius, self.radius:2*self.radius] = -8  # 赋值：在每个第三维和第四维（三行三列）上的第二行第二列的值赋值为-8
        filt2.requires_grad = False
        filt2 = filt2.cuda()
        prededge = F.conv2d(prediction.float(), filt2, bias=None,
                            stride=1, padding=self.radius, groups=self.n_classes)  # 进行卷积，filt2相当于权重  [4,8,256,256]

        # torch.pow(prededge, 2) 等价于 y=a^x  y=a^2= a*a 计算两个张量或者一个张量与一个标量的指数计算结果，返回一个张量。
        norm = torch.sum(torch.pow(prededge, 2), 1).unsqueeze(1)  # 先计算指数函数，再加1，最后增加一个维度  [4, 1, 256, 256]
        prededge = norm/(norm + self.alpha)


        # mask = lbedge.float()
        # num_positive = torch.sum((mask==1).float()).float()
        # num_negative = torch.sum((mask==0).float()).float()

        # mask[mask == 1] = 1.0 * num_negative / (num_positive + num_negative)
        # mask[mask == 0] = 1.5 * num_positive / (num_positive + num_negative)

        # cost = torch.nn.functional.binary_cross_entropy(
            # prededge.float(),lbedge.float(), weight=mask, reduce=False)
        # return torch.mean(cost)
        return BinaryDiceLoss()(prededge.float(),lbedge.float())


class BinaryDiceLoss(nn.Module):
    """Dice loss of binary class
    Args:
        smooth: A float number to smooth loss, and avoid NaN error, default: 1
        p: Denominator value: \sum{x^p} + \sum{y^p}, default: 2
        predict: A tensor of shape [N, *]
        target: A tensor of shape same with predict
        reduction: Reduction method to apply, return mean over batch if 'mean',
            return sum if 'sum', return a tensor of shape [N,] if 'none'
    Returns:
        Loss tensor according to arg reduction
    Raise:
        Exception if unexpected reduction
    """
    def __init__(self, smooth=1, p=2):
        super(BinaryDiceLoss, self).__init__()
        self.smooth = smooth  # 1
        self.p = p  # 2
    def forward(self, predict, target):  #   target:[4,1,256,256]
        assert predict.shape[0] == target.shape[0], "predict & target batch size don't match"
        # contiguous方法改变了多维数组在内存中的存储顺序，以便配合view方法使用
        predict = predict.contiguous().view(predict.shape[0], -1)  # predict:[4, 65536]
        target = target.contiguous().view(target.shape[0], -1)  # target:[4, 65536]

        num = 2*torch.sum(torch.mul(predict, target), dim=1) + self.smooth  # 先把预测值和标签相乘，再在一维上相加，乘2
        den = torch.sum(predict.pow(self.p) + target.pow(self.p), dim=1) + self.smooth  # 将预测值和标签值分别做指数函数，再相加

        loss = 1 - num / den
        return loss.sum()

if __name__ == "__main__":
    loss = SegmentationLosses(cuda=True)
    a = torch.rand(1, 3, 7, 7).cuda()
    b = torch.rand(1, 7, 7).cuda()
    print(loss.CrossEntropyLoss(a, b).item())
    print(loss.FocalLoss(a, b, gamma=0, alpha=None).item())
    print(loss.FocalLoss(a, b, gamma=2, alpha=0.5).item())

"""原本：
class SegmentationLosses(object):
    def __init__(self, weight=None, size_average=True, batch_average=True, ignore_index=255, cuda=False):
        self.ignore_index = ignore_index  # 255
        self.weight = weight  # None
        self.size_average = size_average  # true
        self.batch_average = batch_average  # true
        self.cuda = cuda  # true

    def build_loss(self, mode='ce'):
        # Choices: ['ce' or 'focal']
        if mode == 'ce':
            return self.CrossEntropyLoss  # 交叉熵损失函数
        elif mode == 'focal':
            return self.FocalLoss
        else:
            raise NotImplementedError

    def CrossEntropyLoss(self, logit, target):  # logit:[1,3,7,7]  target:[1,7,7]
        n, c, h, w = logit.size()  # n:batch:1 c:channel:3
        criterion = nn.CrossEntropyLoss(weight=self.weight, ignore_index=self.ignore_index,
                                        size_average=self.size_average)  # ignore_index:计算交叉熵时，自动忽略的标签值，
        if self.cuda:
            criterion = criterion.cuda()

        loss = criterion(logit, target.long())  # 交叉熵函数
        # print(loss.shape)
        if self.batch_average:
            loss /= n

        return loss

    def FocalLoss(self, logit, target, gamma=2, alpha=0.5):
        n, c, h, w = logit.size()
        criterion = nn.CrossEntropyLoss(weight=self.weight, ignore_index=self.ignore_index,
                                        size_average=self.size_average)
        if self.cuda:
            criterion = criterion.cuda()

        logpt = -criterion(logit, target.long())
        pt = torch.exp(logpt)
        if alpha is not None:
            logpt *= alpha
        loss = -((1 - pt) ** gamma) * logpt

        if self.batch_average:
            loss /= n

        return loss
"""



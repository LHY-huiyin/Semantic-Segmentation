import argparse
import os
import numpy as np
from tqdm import tqdm

from mypath import Path
from dataloaders import make_data_loader
from modeling.sync_batchnorm.replicate import patch_replication_callback
from modeling.deeplab_ghostnet import *
from utils.loss import SegmentationLosses
from utils.calculate_weights import calculate_weigths_labels
from utils.lr_scheduler import LR_Scheduler
from utils.saver import Saver
from utils.summaries import TensorboardSummary
from utils.metrics import Evaluator

import matplotlib.pyplot as plt
import cv2
import time

import codecs
from dataloaders.utils import decode_segmap

class Trainer(object):
    def __init__(self, args):
        self.args = args

        # Define Saver
        self.saver = Saver(args)  # 存储相关参数的文件定义
        self.saver.save_experiment_config()  # 保存实验的参数
        # Define Tensorboard Summary
        self.summary = TensorboardSummary(self.saver.experiment_dir)  # 'run\\pascal\\deeplab-resnet\\experiment_10'
        self.writer = self.summary.create_summary()  # 'run\\pascal\\deeplab-resnet\\experiment_10'

        # Define Dataloader
        kwargs = {'num_workers': args.workers, 'pin_memory': True}  # {'num_workers': 4, 'pin_memory': True}
        self.train_loader, self.val_loader, self.test_loader, self.nclass = make_data_loader(args, **kwargs)
        # 使用”**”调用函数,这种方式我们需要一个字典.注意:在函数调用中使用”*”，我们需要元组;

        # Define network
        model = DeepLab(num_classes=self.nclass,  # num_classes=8
                        backbone=args.backbone,  # backbone=resnet
                        output_stride=args.out_stride,  # output_stride=16
                        sync_bn=args.sync_bn,  # sync_bn= False
                        freeze_bn=args.freeze_bn)  # freeze_bn=False

        # 构建一个优化参数列表
        train_params = [{'params': model.get_1x_lr_params(), 'lr': args.lr},  # train_params[0]:args.lr = 0.0035
                        {'params': model.get_10x_lr_params(), 'lr': args.lr * 10}]  # train_params[0]:args.lr = 0.035

        # Define Optimizer  ******优化器***** 是算一个batch计算一次梯度，然后进行一次梯度更新
        optimizer = torch.optim.SGD(train_params, momentum=args.momentum,
                                    weight_decay=args.weight_decay, nesterov=args.nesterov)

        # 随机梯度下降SGD：nesterov具有预测能力，
        # momentum=0.9 综合考虑了梯度下降的方向和上次更新的方向  weight_decay=0.0005 nesterov=false
        # params(iterable)：可用于迭代优化的参数或者定义参数组的dicts。
        # weight_decay (float, optional)：权重衰减(如L2惩罚)(默认: 0)，针对最后更新参数的时候，给损失函数中的加的一个惩罚参数，更新参数使用

        # Define Criterion 损失函数
        # whether to use class balanced weights 类平衡权重
        if args.use_balanced_weights:  # false
            classes_weights_path = os.path.join(Path.db_root_dir(args.dataset),
                                                args.dataset + '_classes_weights.npy')  # 'G:\\LoveDA\\pascal_classes_weights.npy'
            if os.path.isfile(classes_weights_path):
                weight = np.load(classes_weights_path)
            else:
                weight = calculate_weigths_labels(args.dataset, self.train_loader, self.nclass)
            weight = torch.from_numpy(weight.astype(np.float32))
        else:
            weight = None  # None
        self.criterion = SegmentationLosses(weight=weight, cuda=args.cuda).build_loss(
            mode=args.loss_type)  # weight：None cuda:true loss_type='ce'(交叉熵损失函数)
        self.model, self.optimizer = model, optimizer

        # Define Evaluator
        self.evaluator = Evaluator(self.nclass)  # 指标miou
        # Define lr scheduler
        self.scheduler = LR_Scheduler(args.lr_scheduler, args.lr,  # 学习率策略lr_scheduler:'poly' lr=0.0035
                                      args.epochs, len(
                self.train_loader))  # epochs:50  train_loader:578(训练集有1156张图片，由于batch_size=2，所以目前一个batch)
        # 一个epoch意味着训练集中每一个样本都参与训练了一次。epoch：整个数据集训练了多少次.epoch由一个或多个Ｂａｔｃｈ组成
        # 有1156长图片，batch为2，意味着每两个样本才更新一次参数，也就意味着一个epoch里会提取1156/2=578次bach，这样才会把每个样本都提取一边，更新了578次此参数。这是一个epoch做的
        # 每个epoch要训练的图片数量：1156
        # 训练集具有的Batch个数：1156/2=578
        # 每个epoch需要完成的Batch个数：578  每个epoch具有iteration个数：578 每个epoch中发生模型权重更新的次数：578

        # Using cuda
        if args.cuda:
            self.model = torch.nn.DataParallel(self.model,
                                               device_ids=self.args.gpu_ids)  # gpu_ids=0 第0块gpu训练  torch.nn.DataParallel 进行多GPU训练
            # module即表示你定义的模型；device_ids表示你训练的device；
            # output_device这个参数表示输出结果的device；而这最后一个参数output_device一般情况下是省略不写的，那么默认就是在device_ids[0]，也就是第一块卡上，也就解释了为什么第一块卡的显存会占用的比其他卡要更多一些。
            patch_replication_callback(self.model)
            self.model = self.model.cuda()

        # Resuming checkpoint
        self.best_pred = 0.0
        if args.resume is not None:
            if not os.path.isfile(args.resume):
                raise RuntimeError("=> no checkpoint found at '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)  # 读取预训练模型
            args.start_epoch = checkpoint['epoch']
            if args.cuda:
                self.model.module.load_state_dict(checkpoint['state_dict'])  # 加载模型参数
            else:
                self.model.load_state_dict(checkpoint['state_dict'])
            if not args.ft:
                self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.best_pred = checkpoint['best_pred']
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))

        # Clear start epoch if fine-tuning
        if args.ft:
            args.start_epoch = 0

    def training(self, epoch):  # 一个epoch
        train_loss = 0.0
        self.model.train()
        tbar = tqdm(self.train_loader)  # tqdm=578  python写的一种进度条可视化工具
        num_img_tr = len(self.train_loader)
        for i, sample in enumerate(tbar):
            image, target = sample['image'], sample['label']
            # image [4,3,512,512] label[4,512,512]
            if self.args.cuda:
                image, target = image.cuda(), target.cuda()  # image torch.Size([4, 3, 513, 513])
            self.scheduler(self.optimizer, i, epoch, self.best_pred)  # lr_scheduler中的call_调用
            self.optimizer.zero_grad()  # 清空过往梯度；  把梯度置零，也就是把loss关于weight的导数变成0.
            # SGD随机梯度下降法：是算一个batch计算一次梯度，然后进行一次梯度更新
            # 这里梯度值就是对应偏导数的计算结果。显然，我们进行下一次batch梯度计算的时候，前一个batch的梯度计算结果，没有保留的必要了。
            # 所以在下一次梯度更新的时候，先使用optimizer.zero_grad把梯度信息设置为0。
            output = self.model(image)
            # Target 7 is out of bounds. softmax输出的节点数与标签数不一致  已经经过模型 解决：类别数设置为8
            # print(output.shape)  # torch.Size([4, 8, 513, 513])
            # print(target.shape)  # torch.Size([4, 513, 513])
            loss = self.criterion(output, target)  # 交叉熵损失函数 一个epoch之后：0.6823
            loss.backward()  # 反向传播，计算当前梯度；
            # with amp.scale_loss(loss, self.optimizer) as scaled_loss:
            #     scaled_loss.backward()
            self.optimizer.step()  # 根据梯度更新网络参数
            train_loss += loss.item()  # 4.322970390319824递增
            # item()返回loss的值，叠加之后算出总loss，最后再除以mini-batches的数量，取loss平均值。
            tbar.set_description('Train loss: %.3f' % (
                        train_loss / (i + 1)))  # 显示，前面时损失值：train_loss,后面是进度条：Train loss: 0.682:  31%|███       |
            self.writer.add_scalar('train/total_loss_iter', loss.item(), i + num_img_tr * epoch)
            # 数据保存在文件里面供可视化使用，代码运行结束后，会在当前的工作目录下自动生一个 runs 目录

            # Show 10 * 3 inference results each epoch
            if i % (num_img_tr // 10) == 0:  # num_img_tr：578
                global_step = i + num_img_tr * epoch
                self.summary.visualize_image(self.writer, self.args.dataset, image, target, output, global_step)

        self.writer.add_scalar('train/total_loss_epoch', train_loss, epoch)
        print('[Epoch: %d, numImages: %5d]' % (
        epoch, i * self.args.batch_size + image.data.shape[0]))  # [Epoch: 0, numImages:  1156]
        print('Loss: %.3f' % train_loss)  # Loss: 340.250

        if self.args.no_val:
            # save checkpoint every epoch
            is_best = False
            self.saver.save_checkpoint({  # 预训练模型参数
                'epoch': epoch + 1,
                'state_dict': self.model.module.state_dict(),  # 返回一个包含模型状态信息的字典 将每层与层的参数张量之间一一映射
                'optimizer': self.optimizer.state_dict(),  # 包含的是关于优化器状态的信息和使用到的超参数
                'best_pred': self.best_pred,
            }, is_best)

    def validation(self, epoch):
        self.model.eval()  # 测试状态
        self.evaluator.reset()

        # 计算参数量  1000ms/50ms = 20fps  注意：一定要使用ms单位！
        total_num = sum(p.numel() for p in self.model.parameters())
        trainable_num = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        # print('val total参数量：', total_num)
        # print('val Trainable参数量：', trainable_num)

        tbar = tqdm(self.val_loader, desc='\r')
        test_loss = 0.0
        reult_pred = []
        steps_plot = 20
        min_time_sum = 1000
        for i, sample in enumerate(tbar):
            image, target = sample['image'], sample['label']
            if self.args.cuda:
                image, target = image.cuda(), target.cuda()
            with torch.no_grad():
                # 计算FPS
                time_start = time.time()
                output = self.model(image)
                time_end = time.time()
                time_sum = time_end - time_start
                if min_time_sum > time_sum:
                    min_time_sum = time_sum
                # print(time_sum)
            loss = self.criterion(output, target)
            test_loss += loss.item()
            tbar.set_description('Test loss: %.3f' % (test_loss / (i + 1)))
            # 因为是tensor，需要将它从GPU拿出来
            pred = output.data.cpu().numpy()  # (5, 8, 512, 512)
            target = target.cpu().numpy()  # (5, 512, 512)
            pred = np.argmax(pred, axis=1)  # (5, 512, 512)
            # Add batch sample into evaluator
            self.evaluator.add_batch(target, pred)

        # Fast test during the training
        Acc = self.evaluator.Pixel_Accuracy()
        Acc_class = self.evaluator.Pixel_Accuracy_Class()
        mIoU = self.evaluator.Mean_Intersection_over_Union()
        FWIoU = self.evaluator.Frequency_Weighted_Intersection_over_Union()
        self.writer.add_scalar('val/total_loss_epoch', test_loss, epoch)
        self.writer.add_scalar('val/mIoU', mIoU, epoch)
        self.writer.add_scalar('val/Acc', Acc, epoch)
        self.writer.add_scalar('val/Acc_class', Acc_class, epoch)
        self.writer.add_scalar('val/fwIoU', FWIoU, epoch)
        print('Validation:')
        print('[Epoch: %d, numImages: %5d]' % (epoch, i * self.args.batch_size + image.data.shape[0]))
        print("Acc:{}, Acc_class:{}, mIoU:{}, fwIoU: {}".format(Acc, Acc_class, mIoU, FWIoU))
        print('Loss: %.3f' % test_loss)

        new_pred = mIoU  # 交并比
        if new_pred > self.best_pred:
            is_best = True
            self.best_pred = new_pred
            self.saver.save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': self.model.module.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'best_pred': self.best_pred,
            }, is_best)

        # 保存文件
        with codecs.open('实验记录ghostnet.txt', 'a', 'utf-8') as f:
            f.write("训练集：" + str(Path.db_root_dir) + "\n")
            f.write("epoch : " + str(epoch) + "\n")
            # f.write("lr : " + str(lr) + "\n")
            f.write("准确率Acc: " + str(Acc) + "\n")
            f.write("类准确率Acc_class: " + str(Acc_class) + "\n")
            f.write("均交并比mIoU： " + str(mIoU) + "\n")
            f.write("频率权重交并比FWIoU： " + str(FWIoU) + "\n")
            f.write("损失函数Loss： " + str(test_loss) + "\n")
            f.write("最佳值best_pred： " + str(self.best_pred) + "\n")
            f.write("Val total参数量： " + str(total_num) + "\n")
            f.write("Val Trainable参数量： " + str(trainable_num) + "\n")
            f.write("模型推理时间： " + str(min_time_sum) + "\n")
            f.write("-----------------------------------------------------" + "\n")

        # 展示图片 最后一张 output的可视化  output:[5, 8, 512, 512] 彩色，数值在[-0.5,0.5]之间  output是tensor类型  shape是属性,a.shape[0]  size是函数,a.size(0)
        # 数据加载器中数据的维度是[B, C, H, W]，我们每次只拿一个数据出来就是[C, H, W]，而matplotlib.pyplot.imshow要求的输入维度是[H, W, C]，
        # 所以我们需要交换一下数据维度，把通道数放到最后面，这里用到pytorch里面的permute方法（transpose方法也行，不过要交换两次，没这个方便，numpy中的transpose方法倒是可以一次交换完成）
        # 将tensor的维度换位。RGB->BGR  permute(1, 2, 0)
        if new_pred >= 0.54:  # MIOU
            for i, sample in enumerate(tbar):
                image, target = sample['image'], sample['label']
                if self.args.cuda:
                    image, target = image.cuda(), target.cuda()
                with torch.no_grad():
                    output = self.model(image)
                # 将tensor数据集转为数组
                pred = output.data.cpu().numpy()  # (12, 8, 512, 512)
                target = target.cpu().numpy()  # [12, 512, 512]
                pred = np.argmax(pred, axis=1)  # (12, 512, 512)
                # pred = decode_segmap(pred, dataset='pascal')

                # 从这里开始
                plt.figure(figsize=(25, 25))
                for j in range(4):
                    # print(pred[i].shape)     #(512, 512)
                    plt.subplot(2, 2, j + 1)  # 需要注意的是所有的数字不能超过10
                    tmp = np.array(pred[j]).astype(np.uint8)  # (512,512)
                    segmap = decode_segmap(tmp, dataset='pascal')  # (3, 512, 512)
                    plt.imshow(segmap)  # ([256, 256, 1])
                    plt.axis('off')
                plt.show()


def main():
    parser = argparse.ArgumentParser(description="PyTorch DeeplabV3Plus Training")  # 创建解析器
    # ArgumentParser:对象包含将命令行解析成 Python 数据类型所需的全部信息。 description：在参数帮助文档之前显示的文本
    # 给一个 ArgumentParser 添加程序参数信息是通过调用 add_argument() 方法完成的
    parser.add_argument('--backbone', type=str, default='ghostnet',
                        choices=['resnet', 'xception', 'drn', 'mobilenet', 'ghostnet', 'unet'],
                        help='backbone name (default: resnet)')  # 主干网络，用来做特征提取的网络，代表网络的一部分，一般是用于前端提取图片信息，生成特征图feature map,供后面的网络使用。
    parser.add_argument('--out-stride', type=int, default=16,
                        help='network output stride (default: 8)')  # 步长
    parser.add_argument('--dataset', type=str, default='pascal',
                        choices=['pascal', 'coco', 'cityscapes'],
                        help='dataset name (default: pascal)')
    # parser.add_argument('--use-sbd', action='store_true', default=True,
    #                     help='whether to use SBD dataset (default: True)')
    parser.add_argument('--use-sbd', action='store_true', default=False,
                        help='whether to use SBD dataset (default: True)')  # pascal补充的数据集，由于未下载，我设置为false
    parser.add_argument('--workers', type=int, default=4,
                        metavar='N', help='dataloader threads')
    # 设置多线程（threads）进行数据读取时，其实是假的多线程，他是开了N个子进程（PID是连续的）进行模拟多线程工作
    parser.add_argument('--base-size', type=int, default=384,  # 768->384->192->96->48->24
                        help='base image size')
    parser.add_argument('--crop-size', type=int, default=384,
                        help='crop image size')  # 裁剪大小
    parser.add_argument('--sync-bn', type=bool, default=None,
                        help='whether to use sync bn (default: auto)')
    parser.add_argument('--freeze-bn', type=bool, default=False,
                        help='whether to freeze bn parameters (default: False)')
    parser.add_argument('--loss-type', type=str, default='ce',
                        choices=['ce', 'focal'],
                        help='loss func type (default: ce)')  # 损失函数类型：交叉熵损失函数
    # training hyper params
    parser.add_argument('--epochs', type=int, default=None, metavar='N',
                        help='number of epochs to train (default: auto)')
    parser.add_argument('--start_epoch', type=int, default=0,
                        metavar='N', help='start epochs (default:0)')
    parser.add_argument('--batch-size', type=int, default=8,  # 2,4,8,12,14
                        metavar='N', help='input batch size for \
                                training (default: auto)')  # 每批数据量的大小。一次（1个iteration）一起训练batchsize个样本，计算它们的平均损失函数值，来更新参数
    # batchsize越小，一个batch中的随机性越大，越不易收敛。
    parser.add_argument('--test-batch-size', type=int, default=None,
                        metavar='N', help='input batch size for \
                                testing (default: auto)')
    parser.add_argument('--use-balanced-weights', action='store_true', default=False,
                        help='whether to use balanced weights (default: False)')
    # optimizer params
    parser.add_argument('--lr', type=float, default=None, metavar='LR',
                        help='learning rate (default: auto)')  # 学习率
    parser.add_argument('--lr-scheduler', type=str, default='poly',
                        choices=['poly', 'step', 'cos'],
                        help='lr scheduler mode: (default: poly)')  # 调整学习率
    parser.add_argument('--momentum', type=float, default=0.9,
                        metavar='M', help='momentum (default: 0.9)')  # 0.9 动量梯度下降法
    parser.add_argument('--weight-decay', type=float, default=5e-4,
                        metavar='M', help='w-decay (default: 5e-4)')  # 0.0005 L2正则化的目的就是为了让权重衰减到更小的值，在一定程度上减少模型过拟合的问题
    parser.add_argument('--nesterov', action='store_true', default=False,
                        help='whether use nesterov (default: False)')  # Nesterov先用当前的速度v更新一遍参数，在用更新的临时参数计算梯度

    # cuda, seed and logging
    parser.add_argument('--no-cuda', action='store_true', default=
    False, help='disables CUDA training')  # 使用gpu
    parser.add_argument('--gpu-ids', type=str, default='0',
                        help='use which gpu to train, must be a \
                        comma-separated list of integers only (default=0)')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')  # 1 其只对下一次调用生成随机数有效，为了避免二次调用产生不同的随机数据集
    # 能够确保每次抽样的结果一样。而random.seed()括号里的数字，相当于一把钥匙，对应一扇门，同样的数值能够使得抽样的结果一致。

    # checking point
    parser.add_argument('--resume', type=str, default=None,
                        help='put the path to resuming file if needed')  # 断点续训主要保存的是网络模型的参数以及优化器optimizer的状态
    parser.add_argument('--checkname', type=str, default=None,
                        help='set the checkpoint name')  # 保存模型参数，优化器参数，轮数
    # finetuning pre-trained models
    parser.add_argument('--ft', action='store_true', default=False,
                        help='finetuning on a different dataset')
    # evaluation option
    parser.add_argument('--eval-interval', type=int, default=1,
                        help='evaluuation interval (default: 1)')
    parser.add_argument('--no-val', action='store_true', default=False,
                        help='skip validation during training')

    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()  # true and true torch.cuda.is_available()判断GPU是否可用
    if args.cuda:  # true
        try:
            args.gpu_ids = [int(s) for s in args.gpu_ids.split(',')]  # 0
        except ValueError:
            raise ValueError('Argument --gpu_ids must be a comma-separated list of integers only')

    if args.sync_bn is None:  # sync_bn=False
        if args.cuda and len(args.gpu_ids) > 1:
            args.sync_bn = True
        else:
            args.sync_bn = False

    # default settings for epochs, batch_size and lr
    if args.epochs is None:
        epoches = {  # epoches={'coco': 30, 'cityscapes': 200, 'pascal': 50}
            'coco': 30,
            'cityscapes': 200,
            'pascal': 1000,  # 50
        }
        args.epochs = epoches[args.dataset.lower()]  # epochs = 50
        # args.dataset='pascal'  lower()转换字符串中所有大写字符为小写  Pascal -- pascal

    if args.batch_size is None:
        args.batch_size = 4 * len(args.gpu_ids)  # batch_size = 2 不为None

    if args.test_batch_size is None:
        args.test_batch_size = args.batch_size

    if args.lr is None:  # 学习率
        lrs = {  # lrs:{'coco': 0.1, 'cityscapes': 0.01, 'pascal': 0.007}
            'coco': 0.1,
            'cityscapes': 0.01,
            'pascal': 0.007,
        }
        args.lr = lrs[args.dataset.lower()] / (4 * len(args.gpu_ids)) * args.batch_size  # lr=0.0035  【0.007/4】*2

    if args.checkname is None:  # None
        args.checkname = 'deeplab-' + str(args.backbone)  # 'deeplab-resnet'
    print(args)
    torch.manual_seed(args.seed)  # 为特定GPU设置种子，生成随机数  为了确保每次生成固定的随机数
    trainer = Trainer(args)  # 初始化参数
    print('Starting Epoch:', trainer.args.start_epoch)  # Starting Epoch: 0
    print('Total Epoches:', trainer.args.epochs)  # Total Epoches: 50
    for epoch in range(trainer.args.start_epoch, trainer.args.epochs):
        trainer.training(epoch)  # 开始训练
        if not trainer.args.no_val and epoch % args.eval_interval == (args.eval_interval - 1):
            trainer.validation(epoch)

    trainer.writer.close()


if __name__ == "__main__":
    main()
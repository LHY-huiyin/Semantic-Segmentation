##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
## Created by: Hang Zhang
## ECE Department, Rutgers University
## Email: zhang.hang@rutgers.edu
## Copyright (c) 2017
##
## This source code is licensed under the MIT-style license found in the
## LICENSE file in the root directory of this source tree
##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

import math

class LR_Scheduler(object):  # 学习率调整策略
    """Learning Rate Scheduler

    Step mode: ``lr = baselr * 0.1 ^ {floor(epoch-1 / lr_step)}`` *** 固定步长衰减

    Cosine mode: ``lr = baselr * 0.5 * (1 + cos(iter/maxiter))`` ***Cosine余弦退火策略

    Poly mode: ``lr = baselr * (1 - iter/maxiter) ^ 0.9`` *** Poly是一种指数变换的策略
    # 学习率是优化时非常重要的一个因子，通常情况下，在训练过程中学习率都是要动态调整的，通常学习率会逐渐衰减。
    # lr:新的学习率  base_lr:基准学习率  iter:epoch迭代次数  maxiter:最大迭代次数  0.9:power控制曲线的形状

    Args:
        args:
          :attr:`args.lr_scheduler` lr scheduler mode (`cos`, `poly`),
          :attr:`args.lr` base learning rate, :attr:`args.epochs` number of epochs,
          :attr:`args.lr_step`

        iters_per_epoch: number of iterations per epoch 每个epoch中迭代的数量
    """
    def __init__(self, mode, base_lr, num_epochs, iters_per_epoch=0,
                 lr_step=0, warmup_epochs=0):  # num_epochs=50  iters_per_epoch=578
        self.mode = mode  # 动态调整学习率模式:poly指数衰减
        print('Using {} LR Scheduler!'.format(self.mode))
        self.lr = base_lr  # 基准学习率 0.0035
        if mode == 'step':  # 固定步长衰减
            assert lr_step
        self.lr_step = lr_step  # 步长为0
        self.iters_per_epoch = iters_per_epoch  # 578
        self.N = num_epochs * iters_per_epoch  # 28900 最大迭代次数
        self.epoch = -1
        self.warmup_iters = warmup_epochs * iters_per_epoch  # 0

    def __call__(self, optimizer, i, epoch, best_pred):
        T = epoch * self.iters_per_epoch + i  # 当epoch为0时，T=0  当epoch=50时：T = 50*578 + i 迭代次数
        if self.mode == 'cos':
            lr = 0.5 * self.lr * (1 + math.cos(1.0 * T / self.N * math.pi))
        elif self.mode == 'poly':
            lr = self.lr * pow((1 - 1.0 * T / self.N), 0.9)  # lr=0.0035 返回 x^y（x 的 y 次方） 的值
            # 指数函数:y=a^x (0<a<1为下降趋势的)
        elif self.mode == 'step':
            lr = self.lr * (0.1 ** (epoch // self.lr_step))
        else:
            raise NotImplemented
        # warm up lr schedule
        if self.warmup_iters > 0 and T < self.warmup_iters:
            lr = lr * 1.0 * T / self.warmup_iters
        if epoch > self.epoch:  # 在epoch=50之前
            print('\n=>Epoches %i, learning rate = %.4f, \
                previous best = %.4f' % (epoch, lr, best_pred))
            self.epoch = epoch
        assert lr >= 0
        self._adjust_learning_rate(optimizer, lr)

    def _adjust_learning_rate(self, optimizer, lr):
        if len(optimizer.param_groups) == 1:
            optimizer.param_groups[0]['lr'] = lr
        else:  # len == 2
            # enlarge the lr at the head
            optimizer.param_groups[0]['lr'] = lr  # 0.0035
            for i in range(1, len(optimizer.param_groups)):
                optimizer.param_groups[i]['lr'] = lr * 10  # 0.035

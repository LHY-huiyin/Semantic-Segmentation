import os

class Path(object):
    @staticmethod
    def db_root_dir(dataset):
        if dataset == 'pascal':
            # return '/root/autodl-tmp/loveda_a/'  # AutoDL云服务器跑
            # return 'G:\\loveda_aft\\'
            # return 'C:\\Remote sensing semantic segmentation\\loveda_a\\'  #'G:\\Loveda_a\\'  数据集大小为512*512
            # return 'C:\\Remote sensing semantic segmentation\\loveda_last\\'  # 错误的设置：训练集348 验证集512
            # return 'X:\\luo\\loveda_a\\'  # 3080师弟电脑的路径
            # return 'C:\\luo\\loveda_a\\'  # 3090服务器的路径

            # return 'H:\\datasets\\Potsdam\\'  # 3060的路径 Potsdam公共数据集
            # return 'C:\\luo\\Vaihingen\\'  # 3090服务器的路径 Vaihingen数据集
            return 'C:\\Remote sensing semantic segmentation\\Vaihingen\\'  # 3060的路径 Vaihingen数据集
        # elif dataset == 'sbd':
        #     return '/path/to/datasets/benchmark_RELEASE/'  # folder that contains dataset/.
        # elif dataset == 'cityscapes':
        #     return '/path/to/datasets/cityscapes/'     # foler that contains leftImg8bit/
        # elif dataset == 'coco':
        #     return '/path/to/datasets/coco/'
        # else:
        #     print('Dataset {} not available.'.format(dataset))
        #     raise NotImplementedError

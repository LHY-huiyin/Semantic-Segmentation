import os

class Path(object):
    @staticmethod
    def db_root_dir(dataset):
        if dataset == 'pascal':
            # return '/root/autodl-tmp/loveda_a/'  # AutoDL云服务器跑
            # return 'G:\\loveda_aft\\'
            return 'G:\\Loveda_a\\'
            # return 'X:\\luo\\loveda_a\\'
        # elif dataset == 'sbd':
        #     return '/path/to/datasets/benchmark_RELEASE/'  # folder that contains dataset/.
        # elif dataset == 'cityscapes':
        #     return '/path/to/datasets/cityscapes/'     # foler that contains leftImg8bit/
        # elif dataset == 'coco':
        #     return '/path/to/datasets/coco/'
        # else:
        #     print('Dataset {} not available.'.format(dataset))
        #     raise NotImplementedError

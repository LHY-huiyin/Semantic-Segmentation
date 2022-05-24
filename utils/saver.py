import os
import shutil
import torch
from collections import OrderedDict
import glob

class Saver(object):

    def __init__(self, args):
        self.args = args  # 接受任意个数的参数
        self.directory = os.path.join('run', args.dataset, args.checkname)  # 存放运行文件的路径 'run\\pascal\\deeplab-resnet'
        self.runs = sorted(glob.glob(os.path.join(self.directory, 'experiment_*'))) #['run\\pascal\\deeplab-resnet\\experiment_0',
        # 'run\\pascal\\deeplab-resnet\\experiment_1', 'run\\pascal\\deeplab-resnet\\experiment_10', 'run\\pascal\\deeplab-resnet\\experiment_2',
        # 'run\\pascal\\deeplab-resnet\\experiment_3', 'run\\pascal\\deeplab-resnet\\experiment_4', 'run\\pascal\\deeplab-resnet\\experiment_5',
        # 'run\\pascal\\deeplab-resnet\\experiment_6', 'run\\pascal\\deeplab-resnet\\experiment_7', 'run\\pascal\\deeplab-resnet\\experiment_8', 'run\\pascal\\deeplab-resnet\\experiment_9']
        # glob.glob查找符合自己目的的文件
        # run_id = int(self.runs[-1].split('_')[-1]) + 1 if self.runs else 0  # 把experiment_*中的“*”提取出来：10

        if self.runs:  # 如果runs不为0
            max = 0
            for i in range(len(self.runs)):
                run_id = int(self.runs[i].split('_')[-1])
                if run_id > max:
                    max = run_id
            run_id = max + 1
        else:
            run_id = 0

        self.experiment_dir = os.path.join(self.directory, 'experiment_{}'.format(str(run_id))) # 'run\\pascal\\deeplab-resnet\\experiment_10'
        if not os.path.exists(self.experiment_dir):
            os.makedirs(self.experiment_dir)  # 递归创建目录

    def save_checkpoint(self, state, is_best, filename='checkpoint.pth.tar'):
        """Saves checkpoint to disk"""
        filename = os.path.join(self.experiment_dir, filename)  # 'run\\pascal\\deeplab-resnet\\experiment_10\\checkpoint.pth.tar'
        torch.save(state, filename)  # state:字典  将state存入filename
        if is_best:  # 验证时是true 控制量：是否更新参数
            best_pred = state['best_pred']
            with open(os.path.join(self.experiment_dir, 'best_pred.txt'), 'w') as f:
                f.write(str(best_pred))  # 存入新的最好的
            if self.runs:  # run为true
                previous_miou = [0.0]  # 建立一个数组来存放文件中之前的best_pred
                for run in self.runs:
                    run_id = run.split('_')[-1]
                    path = os.path.join(self.directory, 'experiment_{}'.format(str(run_id)), 'best_pred.txt')
                    # 'run\\pascal\\deeplab-resnet\\experiment_10\\best_pred.txt'
                    if os.path.exists(path):
                        with open(path, 'r') as f:
                            miou = float(f.readline())  # 从文件读取整行(之前的best_pred)，包括 "\n" 字符
                            previous_miou.append(miou) # 将文件中的数字存入数组previous_miou
                    else:#如果没有存在该文件
                        continue
                max_miou = max(previous_miou) # 取最大的
                if best_pred > max_miou: # 将当前的pred存入文件中
                    shutil.copyfile(filename, os.path.join(self.directory, 'model_best.pth.tar'))
                    #shutil.copyfile复制文件到另一个文件夹中
            else:
                shutil.copyfile(filename, os.path.join(self.directory, 'model_best.pth.tar'))
                # filename：'run\\pascal\\deeplab-resnet\\experiment_10\\checkpoint.pth.tar'
                # 'run\\pascal\\deeplab-resnet\\model_best.pth.tar

    def save_experiment_config(self): # 保存实验的参数
        logfile = os.path.join(self.experiment_dir, 'parameters.txt')  # 'run/pascal/deeplab-resnet/experiment_10/parameters.txt'
        log_file = open(logfile, 'w')
        p = OrderedDict()  #有序字典可以按字典中元素的插入顺序来输出 只是记住元素插入顺序并按顺序输出  有序字典一般用于动态添加并需要按添加顺序输出的时候
        #如果有序字典中的元素一开始就定义好了，后面没有插入元素这一动作，那么遍历有序字典，其输出结果仍然是无序的　＃
        p['datset'] = self.args.dataset
        p['backbone'] = self.args.backbone
        p['out_stride'] = self.args.out_stride
        p['lr'] = self.args.lr
        p['lr_scheduler'] = self.args.lr_scheduler
        p['loss_type'] = self.args.loss_type
        p['epoch'] = self.args.epochs
        p['base_size'] = self.args.base_size
        p['crop_size'] = self.args.crop_size

        for key, val in p.items():
            log_file.write(key + ':' + str(val) + '\n')  # 写入文件中
        log_file.close()
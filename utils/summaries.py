import os
import torch
from torchvision.utils import make_grid
from tensorboardX import SummaryWriter
from dataloaders.utils import decode_seg_map_sequence

class TensorboardSummary(object):
    def __init__(self, directory):
        self.directory = directory  # 'run\\pascal\\deeplab-resnet\\experiment_10'

    def create_summary(self):
        writer = SummaryWriter(log_dir=os.path.join(self.directory))  # SummaryWriter记录日志信息
        # log_dir:'run\\pascal\\deeplab-resnet\\experiment_10'
        # os.path.join():连接两个或更多的路径名组件
        return writer

    def visualize_image(self, writer, dataset, image, target, output, global_step):
        grid_image = make_grid(image[:3].clone().cpu().data, 3, normalize=True)
        writer.add_image('Image', grid_image, global_step)
        grid_image = make_grid(decode_seg_map_sequence(torch.max(output[:3], 1)[1].detach().cpu().numpy(),
                                                       dataset=dataset), 3, normalize=False, range=(0, 255))
        writer.add_image('Predicted label', grid_image, global_step)
        grid_image = make_grid(decode_seg_map_sequence(torch.squeeze(target[:3], 1).detach().cpu().numpy(),
                                                       dataset=dataset), 3, normalize=False, range=(0, 255))
        writer.add_image('Groundtruth label', grid_image, global_step)

    def visualize_image_u2net(self, writer, dataset, image, target, output, global_step):
        # 复制图片的值
        grid_image = make_grid(image[:3].clone().cpu().data, 3, normalize=True)
        writer.add_image('Image', grid_image, global_step)

        # 将tensor数据转为numpy数组：去除output数据的一维的数值，然后复制detach()，再放在cpu上运行，并转为numpy数组
        # torch.max(output[:3],1)从维度1上获取output的最大值；torch.max(b[:3],1)[1]获取值[2,512,512]
        grid_image = make_grid(decode_seg_map_sequence(torch.max(output[:3], 1)[1].detach().cpu().numpy(),
                                                       dataset=dataset), 3, normalize=False, range=(0, 255))
        writer.add_image('Predicted label', grid_image, global_step)

        # 使用原来的target，因为处理后的target1全是0,1，选择最大值后全1
        grid_image = make_grid(decode_seg_map_sequence(torch.squeeze(target[:3], 1).detach().cpu().numpy(),
                                                       dataset=dataset), 3, normalize=False, range=(0, 255))
        writer.add_image('Groundtruth label', grid_image, global_step)
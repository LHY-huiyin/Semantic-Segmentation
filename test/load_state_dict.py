from collections import OrderedDict

import torch

from modeling.backbone.ghostnet import my_ghostnet
from modeling.backbone.ghostnet_rectify import GhostNet

# model1 = GhostNet()
# model2 = my_ghostnet()
checkpoint = torch.load('state_dict_73.98.pth')
checkpoint1 = torch.load('ghostnet_1x-f97d70db.pth')

print(checkpoint)

# ******  首先说一下“魔改”的定义：网络结构得到了不同的封装，
# 比如原网络结构是很多卷积层被放到同一个nn.sequential当中，而现在这些卷积层被放到了不同的nn.sequential当中（顺序没变），
# 这使得储放网络参数的有序字典的键值发生变化，但实际上如果网络结构没有变的话，第二种情况就没法用了。
# 但通过仔细的对前面Pytorch加载预训练参数的观察下，我们发现，我们只要将要加载的预训练参数放到一个有序字典当中就行了
# net_path_PRE = 'state_dict_73.98.pth'
# pretrained_model = pretrain_model()
# pretrained_model.load_state_dict(torch.load(net_path_PRE))
# pretrained_dict = pretrained_model.state_dict()
#
# model_dict = model.state_dict()
# model_keys = []
#
# for k, v in model_dict.items():
#     model_keys.append(k)
#
# # 我的网络只在原网络结构的后面添加了一些卷积层，而前面的网络被nn.sequential重新封装，
# # 所以前面的预训练参数完封不动的加载到我的网络当中
# i = 0
# for k, v in pretrained_dict.items():
#     model_dict[model_keys[i]] = v
#     i += 1
#
# self.net.load_state_dict(model_dict)


def load_model(model, pretrain_dir):
    state_dict_ = torch.load(pretrain_dir, map_location='cuda:0')
    print('loaded pretrained weights form %s !' % pretrain_dir)
    state_dict = OrderedDict()  # 对字典对象中的元素排序

    # convert data_parallal to model 去掉module字符
    for key in state_dict_:
        if key.startswith('module') and not key.startswith('module_list'):  # 判断当前字符串是否是以另外一个给定的子字符串“开头”的，根据判断结果返回 true 或 false。
            state_dict[key[7:]] = state_dict_[key]
        else:
            state_dict[key] = state_dict_[key]

    # check loaded parameters and created model parameters  去掉module字符
    model_state_dict_ = model.state_dict()
    model_state_dict = OrderedDict()
    for key in model_state_dict_:
        if key.startswith('module') and not key.startswith('module_list'):
            model_state_dict[key[7:]] = model_state_dict_[key]
        else:
            model_state_dict[key] = model_state_dict_[key]
    # 检查权重格式
    for key in state_dict:
        if key in model_state_dict:
            if state_dict[key].shape != model_state_dict[key].shape:
                print('Skip loading parameter {}, required shape{}, loaded shape{}.'.format(
                    key, model_state_dict[key].shape, state_dict[key].shape))
                state_dict[key] = model_state_dict[key]
        else:
            state_dict.pop(key)
            print('Drop parameter {}.'.format(key))

    for key in model_state_dict:
        if key not in state_dict:
            print('No param {}.'.format(key))
            state_dict[key] = model_state_dict[key]
    # 将权重的key与model的key统一
    model_key = list(model_state_dict_.keys())
    pretrained_key = list(state_dict.keys())
    pre_state_dict = OrderedDict()
    for k in range(len(model_key)):
        # if model_key[k] != pretrained_key[k]:
        pre_state_dict[model_key[k]] = state_dict[pretrained_key[k]]

    model.load_state_dict(pre_state_dict, strict=True)

    return model



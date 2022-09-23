from newmodeling.backbone import resnet101_5, resnet101_4, resnet101_2

def build_backbone(backbone, output_stride, BatchNorm):  # 'resnet' 16
    if backbone == 'resnet':  # 16
        return resnet101_2.ResNet101(output_stride, BatchNorm)
        # return resnet101_5.ResNet101(output_stride, BatchNorm)
        # return resnet101_4.ResNet101(output_stride, BatchNorm)

        # return resnet101.ResNet101(output_stride, BatchNorm)
        # return resnet50_pointflow.ResNet50(output_stride, BatchNorm)  # 获取4各输出
        # return resnetxt.resnext101()
        # return resnet50_deeplabunet.ResNet50(output_stride, BatchNorm)  # 获取5个输出
        # return resnet50_bifpn.ResNet50(output_stride, BatchNorm)
        # return resnet101_msapp.ResNet101(output_stride, BatchNorm)  # 获取4各输出
        # return resnet101_bifpn.ResNet101(output_stride, BatchNorm)  # 获取5个输出
    # elif backbone == 'xception':
    #     return xception.AlignedXception(output_stride, BatchNorm)
    # elif backbone == 'drn':
    #     return drn.drn_d_54(BatchNorm)
    # elif backbone == 'mobilenet':
    #     return mobilenet_origin.MobileNetV2(output_stride, BatchNorm)
    # elif backbone == 'ghostnet':
    #     return ghostnet_origin.GhostNet()
    else:
        raise NotImplementedError

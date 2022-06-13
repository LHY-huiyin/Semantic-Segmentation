from modeling.backbone import resnet, xception, drn, mobilenet, ghostnet_rectify, resnetxt_cifar, resnetxt, mobilenet_origin

def build_backbone(backbone, output_stride, BatchNorm):  # 'resnet' 16
    if backbone == 'resnet':  # 16
        return resnet.ResNet101(output_stride, BatchNorm)
        # return resnet.ResNet50(output_stride, BatchNorm)
        # return resnetxt.resnext101()
    elif backbone == 'xception':
        return xception.AlignedXception(output_stride, BatchNorm)
    elif backbone == 'drn':
        return drn.drn_d_54(BatchNorm)
    elif backbone == 'mobilenet':
        return mobilenet_origin.MobileNetV2(output_stride, BatchNorm)
    elif backbone == 'ghostnet':
        return ghostnet_rectify.GhostNet()
    else:
        raise NotImplementedError

from modeling.backbone import resnet101, xception, drn, mobilenet, ghostnet_rectify, resnetxt_cifar, resnetxt,\
    mobilenet_origin, ghostnet_origin, resnet50, resnet50_pointflow, resnet50_deeplabunet

def build_backbone(backbone, output_stride, BatchNorm):  # 'resnet' 16
    if backbone == 'resnet':  # 16
        # return resnet.ResNet101(output_stride, BatchNorm)
        # return resnet50_pointflow.ResNet50(output_stride, BatchNorm)
        # return resnetxt.resnext101()
        return resnet50_deeplabunet.ResNet50(output_stride, BatchNorm)
    elif backbone == 'xception':
        return xception.AlignedXception(output_stride, BatchNorm)
    elif backbone == 'drn':
        return drn.drn_d_54(BatchNorm)
    elif backbone == 'mobilenet':
        return mobilenet_origin.MobileNetV2(output_stride, BatchNorm)
    elif backbone == 'ghostnet':
        return ghostnet_origin.GhostNet()
    else:
        raise NotImplementedError

import torch
import torch.nn as nn
import torch.nn.functional as F
# from network.nn.mynn import Norm2d


def point_sample(input, point_coords, **kwargs):
    """
    A wrapper around :function:`torch.nn.functional.grid_sample` to support 3D point_coords tensors.
    Unlike :function:`torch.nn.functional.grid_sample` it assumes `point_coords` to lie inside
    [0, 1] x [0, 1] square.

    Args:
        input (Tensor): A tensor of shape (N, C, H, W) that contains features map on a H x W grid.
        point_coords (Tensor): A tensor of shape (N, P, 2) or (N, Hgrid, Wgrid, 2) that contains
        [0, 1] x [0, 1] normalized point coordinates.

    Returns:
        output (Tensor): A tensor of shape (N, C, P) or (N, C, Hgrid, Wgrid) that contains
            features for points in `point_coords`. The features are obtained via bilinear
            interplation from `input` the same way as :function:`torch.nn.functional.grid_sample`.
    """
    """
    input：[B, C, H_in, W_in]
    grid: [B, H_out, W_out, 2]   
    output: [B, C, H_out, W_out]   grid的大小指定了输出大小，每个grid的位置是一个(x,y)坐标，其值来自于：输入input的(x,y)的双线性插值得到
    grid中最后一个维度的2表示在input中的相对索引位置（offset），函数的内部主要执行几件事：
        遍历output图像的所有像素坐标
        比如现在要求output中的（5, 5）坐标的特征向量，若通过查找grid中（5, 5）位置中的offset值为（0.1, 0.2）
            根据(0.1*W_in， 0.2*H_in)得到对应input图像上的位置坐标
            通过双线性插值得到该点的特征向量。
            将该特征向量copy到output图像的(5, 5)位置
        底层代码： 
        ix = ((ix + 1) / 2) * (IW-1);
        iy = ((iy + 1) / 2) * (IH-1);       ix,iy是对应grid的值 [-1,1]->[0,1]  IW,IH是输入特征图的大小
        注：grid中的offset坐标必须是归一化的: x = x / (W_in - 1); y = y / (H_in -1)
    """
    device = input.device
    dtype = input.dtype
    add_dim = False
    if point_coords.dim() == 3:
        add_dim = True
        point_coords = point_coords.unsqueeze(2)  # 在[2]处增加一个维度 torch.Size([8, 32, 1, 2])

    # 进行插值：从point_coords获取x,y的坐标，然后从特征图中取32个值  [8, 40, 32, 32]
    # 第二个参数是要插值的点，输入的是2.0*point_coords-1.0，其中point_coords是之前随机生成的k*N （一张图片）个二维坐标点
    # grid_sample()输入的插值点的坐标是相对坐标，相对于mask的位置，其中左上角坐标是(-1, -1), 右下角坐标是(1, 1)。所以传入的坐标范围要在[-1, 1]之间
    output = F.grid_sample(input, 2.0 * point_coords - 1.0, **kwargs)  # torch.Size([8, 40, 32, 1])
    # M = 2.0 * point_coords - 1.0   # torch.Size([8, 32, 1, 2])
    if add_dim:
        output = output.squeeze(3)  # 去掉一维[3]:  [8, 40, 32]

    # if output.device == device:
    #     # 将数据从GPU中取出，加载到cpu，并转为numpy类型
    #     output = output.cpu().detach().numpy()
    #     # numpy转换数据类型
    #     output.dtype = 'float16'  # 强转：普适性不强
    #     # output.dtype = dtype  错
    #     # 将numpy数据取出加载到torch(tensor)上，并加载到gpu上
    #     output = torch.from_numpy(output).cuda()  # GPU的情况
    # else:
    #     output = output.cpu().detach().numpy()
    #     output.dtype = 'float16'
    #     output = torch.from_numpy(output)

    return output


def get_uncertain_point_coords_on_grid(uncertainty_map, num_points):
    """
    Find `num_points` most uncertain points from `uncertainty_map` grid.

    Args:
        uncertainty_map (Tensor): A tensor of shape (N, 1, H, W) that contains uncertainty
            values for a set of points on a regular H x W grid.
        num_points (int): The number of points P to select.

    Returns:
        point_indices (Tensor): A tensor of shape (N, P) that contains indices from
            [0, H x W) of the most uncertain points.
        point_coords (Tensor): A tensor of shape (N, P, 2) that contains [0, 1] x [0, 1] normalized
            coordinates of the most uncertain points from the H x W grid.
    """
    # 推理时选点，直接选出N个不确定的点，所选的点的坐标都是相对坐标
    R, _, H, W = uncertainty_map.shape  # H：32 W：32 R：2
    h_step = 1.0 / float(H)  # 0.03125
    w_step = 1.0 / float(W)  # 0.03125

    num_points = min(H * W, num_points)  # 32
    # torch.topk()：对于给定的输入张量input，沿着给定的维度，返回k个最大元素==>一个命名元组(values,indices)将会被返回，这里的indices是返回的元素在原始的input张量中的indices。
    # 得到边界信息中最大值的索引（位置）：不确定性越大，越可能是边界点   从1024中选择32个点（最大的），得到它的索引
    point_indices = torch.topk(uncertainty_map.view(R, H * W), k=num_points, dim=1)[1]  # size:[8, 32]
    # 随机生成num_point个二维坐标，batch_size为R，所以随机生成的点的尺寸为[R, num_point, 2]
    # point_coords = torch.zeros(R, num_points, 2, dtype=torch.float16,
    #                            device=uncertainty_map.device)  # Size[8, 32, 2]  uncertainty_map.device=cpu
    point_coords = torch.zeros(R, num_points, 2, dtype=torch.float32,
                               device=uncertainty_map.device)
    # 方便后期的插值grid_sample,坐标是相对坐标，相对于mask的位置，传入的坐标范围在[-1,1]之间
    # [0]维上选点：point_indices % W得到所在列   要插值的点的位置
    point_coords[:, :, 0] = w_step / 2.0 + (point_indices % W).to(torch.float) * w_step
    # [1]维上选点：point_indices// W得到所在行
    point_coords[:, :, 1] = h_step / 2.0 + (point_indices // W).to(torch.float) * h_step

    return point_indices, point_coords


class PointMatcher(nn.Module):
    """
        # 将高层特征与低层特征进行融合A，这一步的操作：将低层特征进行上采样后B与融合后的特征A进行融合C
        Simple Point Matcher
    """

    def __init__(self, dim, kernel_size=3):  # dim：高级语义特征的通道数    dim:64 kernel_size=3
        super(PointMatcher, self).__init__()

        self.match_conv = nn.Conv2d(dim * 2, 1, kernel_size, padding=1)  # conv2d(128, 1, 3, padding=1)
        self.sigmoid = nn.Sigmoid()  # 激活函数:把函数映射到[0,1]
        """
        import torch.nn as nn
        m = nn.Sigmoid()
        input = torch.randn(2)   # tensor([-2.2669,  0.3081])
        output = m(input)   # tensor([0.0939, 0.5764])
        """

    def forward(self, x):
        x_high, x_low = x  # 高层语义信息：x_high:[8, 64, 32, 32]  低层语义信息x_low:[8, 64, 64, 64]

        # 先将低层语义信息x_low进行下采样，双线性插值--w,h变成高层语义的w,h
        x_low = F.upsample(x_low, size=x_high.size()[2:], mode='bilinear', align_corners=True)  # [8, 64, 32, 32]

        # 将高层语义信息与低层语义信息进行融合，64+64=128，输入通道数为两个的叠加，输出通道数为1
        # 3*3卷积，输入通道为dim的2倍，输出通道为1
        certainty = self.match_conv(torch.cat([x_high, x_low], dim=1))  # [8, 1, 32, 32]

        # 激活函数，返回一个概率值
        return self.sigmoid(certainty)


class PointFlowModuleWithMaxAvgpool(nn.Module):
    def __init__(self, in_planes, dim=40, maxpool_size=8, avgpool_size=8, matcher_kernel_size=3,
                 edge_points=64):
        super(PointFlowModuleWithMaxAvgpool, self).__init__()
        self.dim = dim  # 高级语义特征图的通道数   40
        self.point_matcher = PointMatcher(dim, matcher_kernel_size)

        self.down_h = nn.Conv2d(in_planes, dim, 1)  # 1*1卷积  Conv2d(40, 40, kernel_size=(1, 1), stride=(1, 1))
        self.down_l = nn.Conv2d(in_planes, dim, 1)  # Conv2d(40, 40, kernel_size=(1, 1), stride=(1, 1))

        self.softmax = nn.Softmax(dim=-1)
        self.maxpool_size = maxpool_size
        self.avgpool_size = avgpool_size
        self.edge_points = edge_points
        self.max_pool = nn.AdaptiveMaxPool2d((maxpool_size, maxpool_size), return_indices=True)
        self.avg_pool = nn.AdaptiveAvgPool2d((avgpool_size, avgpool_size))

        # 对高层语义信息”Element-wise subtraction“后的结果（边界信息）做3*3卷积
        self.edge_final = nn.Sequential(
            nn.Conv2d(in_channels=in_planes, out_channels=in_planes, kernel_size=3, padding=1, bias=False),
            # Norm2d(in_planes),
            nn.BatchNorm2d(in_planes),
            nn.ReLU(),
            nn.Conv2d(in_channels=in_planes, out_channels=1, kernel_size=3, padding=1, bias=False)
        )

    def forward(self, x):
        # 高层语义信息：x_high（已经进行1*1卷积，改变了通道数，输出为512）  获取“上一层”解码器的低层信息x_low,   x_high的分辨率是x_low的的一半
        x_high, x_low = x  # x_high:[8, 40, 32, 32]  x_low:[8, 40, 64, 64]
        stride_ratio = x_low.shape[2] / x_high.shape[2]  # w的比值，放大倍数  2.0
        # 分别进行1*1卷积
        x_high_embed = self.down_h(x_high)  # torch.Size([8, 64, 32, 32])
        x_low_embed = self.down_l(x_low)  # torch.Size([8, 64, 64, 64])
        # 获取通道数、特征图的长、宽
        N, C, H, W = x_low.shape  # c=40 h=64  n=8  w=64
        N_h, C_h, H_h, W_h = x_high.shape  # C_h:40  H_h:32  N_h:8  W_h:32

        # 得到Ml：将低层语义信息x_low=Fl-1下采样 再将高层语义信息与低层语义信息进行拼接等操作，得到saliency map Ml
        certainty_map = self.point_matcher([x_high_embed, x_low_embed])  # torch.Size([8, 1, 32, 32])
        # 对Ml进行平均池化
        avgpool_grid = self.avg_pool(certainty_map)  # [8, 1, 8, 8]
        _, _, map_h, map_w = certainty_map.size()  # map_h, map_w = 32
        # 对平均池化后的Ml进行上采样，
        avgpool_grid = F.interpolate(avgpool_grid, size=(map_h, map_w), mode="bilinear", align_corners=True)  # [8, 1, 32, 32]


        # edge part      Dual Index Generator左边+Point Flow Module底边到Propagation
        # 对高层语义信息x_high进行”Element-wise subtraction“（左边）：利用显著性图对Fl进行平滑，然后用Fl减去平滑后的图像就是边界信息s
        x_high_edge = x_high - x_high * avgpool_grid  # [8, 40, 32, 32]
        # 将边界信息s进行3*3卷积,获得边界信息m
        edge_pred = self.edge_final(x_high_edge)  # [8, 1, 32, 32]
        # 用边界信息m监督学习，得到边界序号I(b)：point_coords是二维坐标点 32个[x,y]
        point_indices, point_coords = get_uncertain_point_coords_on_grid(edge_pred, num_points=self.edge_points)  # 索引：torch.Size([2, 32])  值：torch.Size([2, 32, 2])
        # 索引所在的列、行，再乘放大倍数2
        sample_x = point_indices % W_h * stride_ratio  # torch.Size([8, 32])
        sample_y = point_indices // W_h * stride_ratio  # torch.Size([8, 32])
        # 对行进行相乘，再加上列:所在位置，展平后的位置
        low_edge_indices = sample_x + sample_y * W  # torch.Size([8, 32])
        # 增加通道数
        low_edge_indices = low_edge_indices.unsqueeze(1).expand(-1, C, -1).long()  # torch.Size([8, 8, 32])
        # 对高层语义信息进行取样:   torch.Size([8, 40, 32])
        high_edge_feat = point_sample(x_high, point_coords)  # [8, 40, 32]  精度32
        # 对低层语义信息进行取样
        low_edge_feat = point_sample(x_low, point_coords)  # [8, 40, 32]  精度32
        #    # ********     Propagation   **************      #
        # torch.bmm():计算两个矩阵的乘法   transpose():数组转置  精度降低
        affinity_edge = torch.bmm(high_edge_feat.transpose(2, 1), low_edge_feat).transpose(2, 1)  # torch.Size([8, 32, 32])
        # 激活函数，[0，1]之间  得到公式（6）
        affinity = self.softmax(affinity_edge)  # torch.Size([8, 32, 32])  精度32
        # 其中fusion_edge_feat( fl-1^r)以实值的形式，给出了C个channel上p个采样点的关系分数
        high_edge_feat = torch.bmm(affinity, high_edge_feat.transpose(2, 1)).transpose(2, 1)  # torch.Size([8, 40, 32])
        fusion_edge_feat = high_edge_feat + low_edge_feat  # torch.Size([8, 40, 32])

        """
           gather可以对一个Tensor进行聚合，声明为：torch.gather(input, dim, index, out=None) → Tensor
           一般来说有三个参数：输入的变量input、指定在某一维上聚合的dim、聚合的使用的索引index，输出为Tensor类型的结果（index必须为LongTensor类型
           dim=0为第0个维度，代表行   dim=1为第一个维度，代表列
            import torch
            a = torch.tensor([[10,5,7],[2,7,6],[31,43,9]])
            a
            tensor([[10,  5,  7],
                    [ 2,  7,  6],
                    [31, 43,  9]])
            idx = torch.tensor([1,0,2])
            idx = idx.unsqueeze(0)
            a.gather(0,idx)   # 在行方向上选择所需：第一个值是1->对[1]=2  第二个值是0->[0]:5
            tensor([[2, 5, 9]])
       """

        # residual part    Dual Index Generator右边+Point Flow Module上边到Propagation  certainty_map：[8,1,32,32]
        # 最大池化得到I(s)：maxpool_grid  有两个作用：一个继续做运算maxpool_grid，一个用去做索引maxpool_grid
        maxpool_grid, maxpool_indices = self.max_pool(certainty_map)  # maxpool_grid:[8, 1, 32, 32]  maxpool_indices:[8, 8, 8]
        maxpool_indices = maxpool_indices.expand(-1, C, -1, -1)  # torch.Size([8, 40, 8, 8]) 复制了一份
        # 对最大池化后的I(s)进行上采样得到A，用A去和高层语义信息Fl点乘
        maxpool_grid = F.interpolate(maxpool_grid, size=(map_h, map_w), mode="bilinear", align_corners=True)  # torch.Size([8, 1, 32, 32])
        # 获得行和列  放大2倍  ？？为什么乘2
        x_indices = maxpool_indices % W_h * stride_ratio   # torch.Size([8, 40, 8, 8])
        y_indices = maxpool_indices // W_h * stride_ratio   # torch.Size([8, 40, 8, 8])
        # 将列数乘64（低层语义信息的W），再加上行数
        low_indices = x_indices + y_indices * W   # torch.Size([8, 40, 8, 8])
        low_indices = low_indices.long()
        # 对高层语义信息进行”Element-wise dot product“-”Element-wise addition“点乘后相加，得到Fl^s
        x_high = x_high + maxpool_grid * x_high  # torch.Size([8, 40, 32, 32])
        # 对高层语义信息Fl^s进行sample，，
        flattened_high = x_high.flatten(start_dim=2)   # torch.Size([8, 40, 1024])
        # 使用I(s)在Fl^s上取点  index=maxpool_indices.flatten(start_dim=2)=>[2,2,64]  对第二维进行聚合，
        high_features = flattened_high.gather(dim=2, index=maxpool_indices.flatten(start_dim=2)).view_as(maxpool_indices)  # torch.Size([8, 40, 8, 8])
        # 对低级语义信息进行sample，，  64*64
        flattened_low = x_low.flatten(start_dim=2)  # torch.Size([8, 40, 4096])
        # 使用I(s)在低级语义信息Fl-1上取点
        low_features = flattened_low.gather(dim=2, index=low_indices.flatten(start_dim=2)).view_as(low_indices)   # torch.Size([8, 40, 8, 8])
        # propagation
        feat_n, feat_c, feat_h, feat_w = high_features.shape  # [8, 40, 8, 8]
        # 调整特征图：降维
        high_features = high_features.view(feat_n, -1, feat_h * feat_w)  # torch.Size([8, 40, 64])
        low_features = low_features.view(feat_n, -1, feat_h * feat_w)  # torch.Size([8, 40, 64])
        # 高层特征与低层特征相乘得到A
        affinity = torch.bmm(high_features.transpose(2, 1), low_features).transpose(2, 1)
        # 激活函数
        affinity = self.softmax(affinity)  # b, n, n
        # 将A与高层特征进行相乘
        high_features = torch.bmm(affinity, high_features.transpose(2, 1)).transpose(2, 1)
        # 融合
        fusion_feature = high_features + low_features   # torch.Size([8, 40, 64])
        # 调整
        mp_b, mp_c, mp_h, mp_w = low_indices.shape  # [8,40,8,8]
        low_indices = low_indices.view(mp_b, mp_c, -1)  # [8,40,64]

        """ Scatter过程：
            import torch
            src = torch.arange(1,11).reshape((2,5))
            src
            tensor([[ 1,  2,  3,  4,  5],
                    [ 6,  7,  8,  9, 10]])
            index = torch.tensor([[0,1,2,0]])
            torch.zeros(3,5,dtype=src.dtype).scatter_(0,index,src)
            tensor([[1, 0, 0, 4, 0],
                    [0, 2, 0, 0, 0],
                    [0, 0, 3, 0, 0]])
        """
        # Scatter过程就是依次按I(b)和I(s)将fl-1^r中的p个点的值回代入这些点在Fl-1的原位置上的特征值，得到Fl-1^r
        # 从I(b)阶段过来的低层语义信息[8, 40, 4096]进行scatter   low_edge_indices:[8, 8, 32]   fusion_edge_feat:[8, 40, 32]
        # 类型不一致：fusion_edge_feat是float32  x_low是float16
        # Scatter过程:从fusion_edge_feat中选择low_edge_indices位置的点，填入x_low中
        # 将 src 中数据根据 index 中的索引按照 dim 的方向填进 input 中
        final_features = x_low.reshape(N, C, H * W).scatter(2, low_edge_indices, fusion_edge_feat)
        # 从I(s)阶段过来的低层语义信息
        final_features = final_features.scatter(2, low_indices, fusion_feature).view(N, C, H, W)

        # edge_pred:用于损失函数
        return final_features, edge_pred

if __name__ == "__main__":
    import torch

    # model = ResNet101(BatchNorm=nn.BatchNorm2d, pretrained=True, output_stride=8)
    model = PointFlowModuleWithMaxAvgpool(in_planes=3)
    input1 = torch.rand(1, 3, 512, 512)  # loveda的图片大小为：1024*1024
    input2 = torch.rand(1, 3, 512, 512)
    # print('input`1',input)
    output, low_level_feat = model(input1, input2)
    print(output.size())  # torch.Size([1, 2048, 32, 32])
    print(low_level_feat.size())  # torch.Size([1, 256, 128, 128])
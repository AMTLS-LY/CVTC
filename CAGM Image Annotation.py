import torch
from torch import nn,einsum
import torch.nn.functional as F
import cv2
import numpy as np
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
def cast_tuple(val, length = 1):
    return val if isinstance(val, tuple) else ((val,) * length)
# 如果 val 已经是一个元组，则直接返回 val。
# 如果 val 不是元组，则将 val 转换为一个长度为 length 的元组。具体来说，它创建一个包含 val 的元组，并将其重复 length 次。
# 例如：
# cast_tuple(5, 3) 会返回 (5, 5, 5)。
# cast_tuple((1, 2), 3) 会返回 (1, 2)，因为输入已经是一个元组

# 多尺度卷积嵌入
class CrossEmbedLayer(nn.Module):
    def __init__(self,dim_in,
                 dim_out,
                 kernel_size, #可迭代对象，如列表
                 stride=2
                 ):
        super(CrossEmbedLayer,self).__init__()

        kernel_size = sorted(kernel_size)
        num_scales = len(kernel_size)

        #计算每个尺度的输出维度，输出维度按 dim_out 的一部分分配，确保总和等于 dim_out
        dim_scales=[int(dim_out/(2**i)) for i in range(1,num_scales)]  # 长度为 num_scales-1
        # 假设 dim_out 是 128，num_scales 是 4，那么计算过程如下:
        # i=1 dim/(2**1)=64--> i=2 dim/(2**2)=32--> i=3 dim/(2**3)=16 ......
        dim_scales = [*dim_scales, dim_out - sum(dim_scales)]  # 加1后长度刚好等于num_scales
        # 将一个额外的元素添加到 dim_scales 列表中，以确保所有尺度的维度总和等于 dim_out

        self.conv=nn.ModuleList([])
        for kernel,dim_scale in zip(kernel_size,dim_scales):
            self.conv.append(
                nn.Conv2d(in_channels=dim_in,out_channels=dim_scale,kernel_size=kernel,stride=stride,padding=(kernel-stride)//2)
            )

    def forward(self,x):
        f=tuple(map(lambda conv: conv(x),self.conv)) # 对每个self.conv的元素都应用卷积
        # for i in f:
        #      print(i.size())
        return torch.cat(f,dim=1)

test=torch.randn(12,3,32,32)
kernel_size=[2,4,6]
model=CrossEmbedLayer(dim_in=3,dim_out=256,kernel_size=kernel_size,stride=2)
print(model(test).size())

# 计算动态位置偏置
def DynamicPositionBias(dim):
    return nn.Sequential(
        nn.Linear(2,dim),
        nn.LayerNorm(dim),
        nn.ReLU(),
        nn.Linear(dim,dim),
        nn.LayerNorm(dim),
        nn.ReLU (),
        nn.Linear(dim,1),
        nn.Flatten(start_dim=0)
    )

class LayerNorm(nn.Module):
    def __init__(self,dim,eps=1e-5):
        super(LayerNorm,self).__init__()

        self.eps=eps
        self.g=nn.Parameter(torch.ones(1,dim,1,1))
        self.b=nn.Parameter(torch.zeros(1,dim,1,1))

    def forward(self,x):
        var=torch.var(x,dim=1,unbiased=False,keepdim=True)
        # 计算在dim=1上的方差
        # unbiased = False 表示使用有偏估计， keepdim = True 保持输出的维度与输入相同
        mean=torch.mean(x,dim=1,keepdim=True)
        return ((x-mean)/(var+self.eps)) * self.g + self.b

# 前馈传播
def FeedForward(dim, mult = 4, dropout = 0.):
    return nn.Sequential(
        LayerNorm(dim),
        nn.Conv2d(dim, dim * mult, 1),
        nn.GELU(),
        nn.Dropout(dropout),
        nn.Conv2d(dim * mult, dim, 1)
    )

class Attention(nn.Module):
    def __init__(self,
                 dim,
                 attn_type, # 支持长距离(long)和短距离(short)
                 window_size, # 注意力机制中每个窗口的大小
                 dim_head=32, # 每个头在计算查询、键和值时的特征维度
                 dropout=0.):
        super(Attention,self).__init__()
        assert attn_type in {'short', 'long'}, 'attention type 必须是long或者short'

        heads = dim // dim_head # 头数
        # 确保 dim >= dim_head
        assert dim >= dim_head, 'dim 必须大于等于 dim_head'
        if heads==0:
            raise ValueError('heads 不能为零，请确保 dim >= dim_head')
        self.heads = heads
        self.dim_head=dim_head
        self.scale = dim_head ** -0.5
        inner_dim = dim_head * heads # 所有头的总维度

        self.attn_type = attn_type
        self.window_size = window_size
        self.norm = LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Conv2d(dim, inner_dim * 3, 1, bias=False)
        self.to_out = nn.Conv2d(inner_dim, dim, 1)

        #动态位置偏置
        self.dpb=DynamicPositionBias(dim//4)

        # 计算位置
        pos = torch.arange(window_size)
        grid = torch.stack(torch.meshgrid(pos, pos, indexing='ij'))
        # torch.meshgrid(pos, pos, indexing='ij') 生成两个二维网格张量，分别表示每个位置的行和列索引。
        # torch.stack 将这些网格张量沿新维度堆叠，形成一个形状为 (2, window_size, window_size) 的张量 grid。
        # 在0维度上 第一，第二维度就是两个表格
        _,w1,w2=grid.size()
        grid=grid.view(-1,w1*w2).permute(1,0).contiguous()
        real_pos=grid.view(w1*w2,1,2) - grid.view(1,w1*w2,2)
        # 计算每个位置对之间的相对位置，结果 rel_pos 的形状为 (window_size * window_size, window_size * window_size, 2)。
        real_pos=real_pos + window_size-1
        # 将相对位置偏移 window_size - 1，确保所有相对位置都是非负数。

        # 计算每一对位置之间的相对位置索引
        rel_pos_indices = (real_pos * torch.tensor([2 * window_size - 1, 1])).sum(dim=-1)
        # 这种转换方式的目的是将二维的相对位置 (dx, dy) 映射到一个唯一的一维索引，以便在后续操作中可以方便地查找和使用这些相对位置索引。
        # torch.tensor([2 * window_size - 1, 1]) 是一个形状为 (2,) 的一维张量。
        # 相乘时，广播机制会将 torch.tensor([2 * window_size - 1, 1]) 扩展为形状 (1, 1, 2)，使其与 rel_pos 的形状匹配。
        # 最后一个维度上求和，得到一维索引 rel_pos_indices，形状为 (window_size * window_size, window_size * window_size)。

        self.register_buffer('rel_pos_indices', rel_pos_indices, persistent=False)
        # 将 rel_pos_indices 注册为一个缓冲区，使其成为模型的一部分，但不会被视为模型的参数。

    def forward(self,x,return_attention=False):
        b,dim,h,w,heads, wsz, device = *x.shape, self.heads, self.window_size, x.device
        x=self.norm(x)
        if self.attn_type == 'short':
            x=x.view(b,dim,h//wsz,wsz,w//wsz,wsz)
            x=x.permute(0,2,4,1,3,5).contiguous()
            x=x.view(-1,dim,wsz,wsz)
        elif self.attn_type == 'long':
            x = x.view(b, dim, h // wsz, wsz, w // wsz, wsz)
            x = x.permute(0, 2, 4, 1, 3, 5).contiguous()
            x = x.view(-1, dim, wsz, wsz)
        # [batch * (height // wsz) * (width // wsz), dim, wsz, wsz]
        # print('x.size',x.size())
        # print(f"to_qkv 权重形状: {self.to_qkv.weight.shape}")
        q, k, v = self.to_qkv(x).chunk(3, dim = 1) # [batch * (height // wsz) * (width // wsz), heads*dim_head, wsz, wsz]
        q,k,v=map(lambda x: x.view(-1,self.heads,wsz*wsz,self.dim_head),(q,k,v))
        # 分离头部: 将 heads * dim_head 分离成 heads 和 dim_head 两个维度，使得每个头部的特征维度独立出来。
        # 展平空间维度: 将 wsz 和 wsz 两个空间维度展平为一个维度 wsz * wsz，方便后续的矩阵乘法操作。
        # [batch * (height // wsz) * (width // wsz), heads, wsz * wsz, dim_head]
        q = q * self.scale

        sim = einsum('b h i d, b h j d -> b h i j', q, k) # 沿着d维度点积
        # ==>[batch * (height // wsz) * (width // wsz), heads , wsz * wsz , wsz * wsz]
        # 沿着 d 维度进行点积是因为 d 代表每个头的维度（dim_head），即查询和键向量的特征维度。点积操作的目的是计算两个向量之间的相似度或相关性。

        # 为相似度矩阵 sim 添加动态位置偏差
        pos = torch.arange(-wsz, wsz + 1, device=device) # 生成一个从 -wsz 到 wsz 的整数序列，表示相对位置索引。
        rel_pos = torch.stack(torch.meshgrid(pos, pos, indexing='ij'))
        # torch.stack 将这些网格堆叠在一起，形成一个形状为 [2, (2 * wsz + 1), (2 * wsz + 1)] 的张量。
        # 如果 wsz = 2，那么 torch.arange(-2, 3) 生成的序列是 [-2, -1, 0, 1, 2]，总共有 5 个元素，即 2 * 2 + 1。
        _,size1,size2=rel_pos.size()
        rel_pos=rel_pos.permute(1,2,0).view(size1*size2,2)
        # 计算动态位置偏差
        biases = self.dpb(rel_pos.float())  # [(2 * wsz + 1) * (2 * wsz + 1), 2]==>[(2 * wsz + 1) * (2 * wsz + 1)]
        # 检索相对位置偏差：
        # 使用预先计算的相对位置索引 self.rel_pos_indices 从偏差值中检索相应的偏差。
        rel_pos_bias = biases[self.rel_pos_indices]
        # print('rel_pos_bias:',rel_pos_bias.size())
        # print('biases:',biases.size())
        # print('sim:',sim.size())
        sim = sim + rel_pos_bias  # 添加位置偏差到相似度矩阵

        attn = sim.softmax(dim=-1)
        attn = self.dropout(attn)
        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        # [batch * (height // wsz) * (width // wsz), heads , wsz * wsz , wsz * wsz] * [batch * (height // wsz) * (width // wsz), heads , wsz * wsz, dim_head]
        # ==>[batch * (height // wsz) * (width // wsz), heads , wsz * wsz , dim_head]
        out=out.permute(0,1,3,2).contiguous().view(-1 , self.heads*self.dim_head , wsz , wsz)
        # [batch * (height // wsz) * (width // wsz), heads * dim_head, wsz, wsz]
        out = self.to_out(out)
        # [batch * (height // wsz) * (width // wsz), dim, wsz, wsz]

        if self.attn_type == 'short':
            b, d, h, w = b, dim, h // wsz, w // wsz
            out = out.view(b, h, w, d, wsz, wsz)
            out = out.permute(0, 3, 1, 4, 2, 5).contiguous()
            out = out.view(b, d, h * wsz, w * wsz)
        elif self.attn_type == 'long':
            b, d, l1, l2 = b, dim, h // wsz, w // wsz
            out = out.view(b, l1, l2, d, wsz, wsz)
            out = out.permute(0, 3, 1, 4, 2, 5).contiguous()
            out = out.view(b, d, l1 * wsz, l2 * wsz)

        if return_attention:
            return out, attn
        return out

test=torch.randn(12,64,32,32)
model=Attention(dim=64,attn_type='short',window_size=8,dim_head=8)
print(model(test).size())

class Transformer(nn.Module):
    def __init__(self,
                 dim,
                 local_window_size,
                 global_window_size,
                 depth=4,
                 dim_head=32,
                 attn_dropout=0.,
                 ff_dropout=0. #前馈层的 dropout 率
                 ):
        super(Transformer,self).__init__()
        self.layers=nn.ModuleList([])

        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim=dim,attn_type='short',window_size=local_window_size,dim_head=dim_head,dropout=attn_dropout),
                FeedForward(dim=dim,dropout=ff_dropout),
                Attention(dim=dim,attn_type='long',window_size=global_window_size,dim_head=dim_head,dropout=attn_dropout),
                FeedForward(dim=dim,dropout=ff_dropout)
            ]))

    def forward(self, x):
        for short_attn, short_ff, long_attn, long_ff in self.layers:
            x = short_attn(x) + x
            x = short_ff(x) + x
            x = long_attn(x) + x
            x = long_ff(x) + x

        return x

# 确保dim>=dim_head  因为heads = dim // dim_head 否则heads就为0
test=torch.randn(12,8,32,32)
model=Transformer(dim=8,local_window_size=8,global_window_size=16,depth=2,dim_head=8)
print(model(test).size())

class CrossFormer(nn.Module):
    def __init__(
        self,
        dim = (64, 128, 256, 512),
        depth = (2, 2, 8, 2),
        global_window_size = (8, 4, 2, 1),
        local_window_size = 16,
        cross_embed_kernel_sizes = ((4, 8, 16, 32), (2, 4), (2, 4), (2, 4)),
        cross_embed_strides = (4, 2, 2, 2),
        num_classes = 10,
        attn_dropout = 0.,
        ff_dropout = 0.,
        channels = 3,
        feature_dim=20
    ):
        super(CrossFormer,self,).__init__()

        dim = cast_tuple(dim, 4)
        depth = cast_tuple(depth, 4)
        global_window_size = cast_tuple(global_window_size, 4)
        local_window_size = cast_tuple(local_window_size, 4)
        cross_embed_kernel_sizes = cast_tuple(cross_embed_kernel_sizes, 4)
        cross_embed_strides = cast_tuple(cross_embed_strides, 4)

        assert len(dim) == 4
        assert len(depth) == 4
        assert len(global_window_size) == 4
        assert len(local_window_size) == 4
        assert len(cross_embed_kernel_sizes) == 4
        assert len(cross_embed_strides) == 4

        last_dim=dim[-1]
        dims = [channels, *dim]
        # 如果 channels = 3 且 dim = (64, 128, 256, 512)，那么 dims 将是 [3, 64, 128, 256, 512]
        dim_in_and_out = tuple(zip(dims[:-1], dims[1:]))
        # 如果 dims = [3, 64, 128, 256, 512]，那么 dims[:-1] 是 [3, 64, 128, 256]，
        # dims[1:] 是 [64, 128, 256, 512]，zip(dims[:-1], dims[1:])
        # 将生成 [(3, 64), (64, 128), (128, 256), (256, 512)]，
        # 最终 dim_in_and_out 将是 ((3, 64), (64, 128), (128, 256), (256, 512))。

        self.layers=nn.ModuleList([])
        for (dim_in, dim_out), layers, global_wsz, local_wsz, cel_kernel_sizes, cel_stride in zip(dim_in_and_out, depth,
                                                                                                  global_window_size,
                                                                                                  local_window_size,
                                                                                                  cross_embed_kernel_sizes,
                                                                                                  cross_embed_strides):
            self.layers.append(nn.ModuleList([
                CrossEmbedLayer(dim_in, dim_out, cel_kernel_sizes, stride=cel_stride),
                Transformer(dim_out, local_window_size=local_wsz, global_window_size=global_wsz, depth=layers,
                            attn_dropout=attn_dropout, ff_dropout=ff_dropout)
            ]))

        self.to_logits = nn.Sequential(
            nn.Linear(last_dim, num_classes)
        )
        self.feature_reducer = nn.Linear(last_dim, feature_dim)  # 添加线性层降维

    def forward(self, x, tracker):
        feature_maps = []
        for cel, transformer in self.layers:
            x=cel(x)
            coords_list = []
            for conv in cel.conv:
                new_coords = track_point_in_layer(conv, tracker.get_coords()[-1], x.shape)
                coords_list.append(new_coords)
                # 不更新 tracker.update_coords(new_coords)

            # 计算合并后的新坐标
            avg_coords = (
                sum(coord[0] for coord in coords_list) // len(coords_list),
                sum(coord[1] for coord in coords_list) // len(coords_list)
            )
            tracker.update_coords(avg_coords)
            print('cel',len(tracker.get_coords()))
            feature_maps.append(x)

            z=x
            for layer in transformer.layers:
                short_attn, short_ff, long_attn, long_ff = layer

                # Short Attention
                y ,attn= short_attn(z,return_attention=True)
                print('short attn',attn.size())
                new_coords = track_point_in_attention(attn, tracker.get_coords()[-1],y.shape)
                #new_coords = track_point_in_layer(short_attn.to_qkv, tracker.get_coords()[-1], y.shape)
                tracker.update_coords(new_coords)

                # Short FeedForward
                y = short_ff(y)
                print('short ff',y.size())
                new_coords = track_point_in_layer(short_ff[1], tracker.get_coords()[-1], y.shape)  # 追踪第一个卷积层
                tracker.update_coords(new_coords)
                new_coords = track_point_in_layer(short_ff[4], tracker.get_coords()[-1], y.shape)  # 追踪第二个卷积层
                tracker.update_coords(new_coords)

                # Long Attention
                y,attn = long_attn(y,return_attention=True)
                print('long attn',attn.size())
                new_coords = track_point_in_attention(attn, tracker.get_coords()[-1], y.shape)
                #new_coords = track_point_in_layer(long_attn.to_qkv, tracker.get_coords()[-1], y.shape)
                tracker.update_coords(new_coords)

                # Long FeedForward
                y = long_ff(y)
                print('long ff',y.size())
                new_coords = track_point_in_layer(long_ff[1], tracker.get_coords()[-1], y.shape)  # 追踪第一个卷积层
                tracker.update_coords(new_coords)
                new_coords = track_point_in_layer(long_ff[4], tracker.get_coords()[-1], y.shape)  # 追踪第二个卷积层
                tracker.update_coords(new_coords)
                z=y
            x = transformer(x)
            print('transformer',len(tracker.get_coords()))
            feature_maps.append(x)

        x = torch.einsum('b c h w -> b c', x) / (x.shape[2] * x.shape[3])

        return self.to_logits(x), feature_maps

    def extract_features(self, x):
        for cel, transformer in self.layers:
            x = cel(x)
            x = transformer(x)
        # 使用 einsum 进行平均池化
        features = torch.einsum('b c h w -> b c', x) / (x.shape[2] * x.shape[3])
        features = self.feature_reducer(features)  # 降维
        return features


model=CrossFormer(
        dim = (32, 64, 128, 256),
        depth = (2, 2, 2, 2),
        global_window_size = (8, 4, 2, 1),
        local_window_size = 16,
        cross_embed_kernel_sizes = ((2, 4, 6, 8), (2, 4), (2, 4), (2, 4)),
        cross_embed_strides = (2, 2, 2, 2),
        num_classes = 10,
        attn_dropout = 0.2,
        ff_dropout = 0.2,
        channels = 3).to(device)

# 加载状态字典
state_dict = torch.load(r"E:\dataset\mut_best_model.pth")

# 将状态字典加载到模型中
model.load_state_dict(state_dict)


# Coordinate-Aware Gradient Mapping (CAGM)
class TrackPoint:
    def __init__(self, initial_coords):
        self.coords = [initial_coords]

    def update_coords(self, new_coords,ap=True):
        if ap==True:
            self.coords.append(new_coords)

    def get_coords(self):
        return self.coords

def track_point_in_layer(layer, point_coords, input_shape):
    batch_size, channels, height, width = input_shape
    x, y = point_coords

    # 计算新的坐标
    new_x = (x + layer.padding[0] - layer.kernel_size[0] // 2) // layer.stride[0]
    new_y = (y + layer.padding[1] - layer.kernel_size[1] // 2) // layer.stride[1]

    return new_x, new_y

def track_point_in_attention(attention_weights, point_coords,input_shape):
    # attention_weights: [batch_size, num_heads, seq_len, seq_len]
    # point_coords: (x, y) in the input sequence
    batch_size, channels, height, width = input_shape
    # 获取目标位置的注意力权重
    batch_size, num_heads, seq_len, _ = attention_weights.shape
    x, y = point_coords  # 确保 point_coords 是一个元组 (x, y)

    # 确保索引在有效范围内
    if 0 <= y < seq_len:
        print('未超出索引')
        # 获取目标位置的注意力权重
        target_weights = attention_weights[0, :, y, :]  # [num_heads, seq_len]

        # 计算每个头的加权平均坐标
        avg_coords = target_weights.mean(dim=0).argmax(dim=-1)  # [seq_len]

        return avg_coords.item(), y
    else:
        print('超出索引')
        # 如果索引超出范围，使用默认值
        # 使用最近的有效值作为默认值
        return min(max(x, 0), seq_len - 1), min(max(y, 0), seq_len - 1)

model.eval()
# 初始化
initial_coords = (82, 104)  # 假设初始点在 (10, 10)
tracker = TrackPoint(initial_coords)
# 输入数据
from PIL import Image
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
image_path = r"E:\dataset\OriginalDataset\MildDemented\mildDem16.jpg"  # 替换为你的图片路径
image = Image.open(image_path).convert('RGB')


preprocess = transforms.Compose([
    transforms.Resize((256,256)),
])

input_tensor = preprocess(image)
input_tensor = transforms.ToTensor()(input_tensor)
input_data = input_tensor.unsqueeze(0).to(device)  # 添加批次维度

output, feature_maps = model(input_data, tracker)
# 获取追踪的坐标
tracked_coords = tracker.get_coords()



def find_important_regions(feature_maps, tracker, logits, target_class):
    """
    找出特征图中对于分类的关键部分，并评估原始图像中的追踪点是否对分类起关键作用
    :param feature_maps: 特征图列表
    :param tracker: 追踪器对象，包含特征图中的坐标
    :param logits: 分类器输出的logits
    :param target_class: 目标分类
    :return: 关键部分的掩码和原始追踪点的重要性评估
    """
    # 计算目标分类的概率
    probs = F.softmax(logits, dim=-1)
    target_prob = probs[:, target_class].sum()  # 确保 target_prob 是一个标量

    # 初始化重要性掩码
    importance_masks = []

    for feature_map in feature_maps:
        # 计算特征图中每个位置对目标分类的贡献
        grad = torch.autograd.grad(target_prob, feature_map, retain_graph=True)[0]
        importance_mask = torch.sum(grad * feature_map, dim=1, keepdim=True)
        importance_masks.append(importance_mask)

    # 评估原始追踪点的重要性
    importance_weights = []

    for i, coords in enumerate(tracker):
        x, y = coords
        # 确保索引在有效范围内
        if 0 <= x < importance_masks[i].shape[3] and 0 <= y < importance_masks[i].shape[2]:
            importance = importance_masks[i][0, 0, y, x].item()
        else:
            importance = 0  # 或者其他默认值
        importance_weights.append(importance)

    # 归一化重要性
    max_importance = max(importance_weights)
    epsilon = 1e-10  # 一个非常小的数值
    normalized_importance = [imp / (max_importance ) for imp in importance_weights]

    # 计算总重要性
    total_importance = sum(normalized_importance)

    return importance_masks, total_importance

# # 示例用法
# # 假设我们有特征图、追踪器对象、分类器输出的logits和目标分类
# feature_maps = feature_maps  # 特征图列表
# tracker = [tracked_coords[1],tracked_coords[13],tracked_coords[14],tracked_coords[26],tracked_coords[27],tracked_coords[39],tracked_coords[40],tracked_coords[47]]  # 追踪器对象
# logits = output  # 分类器输出的logits
# target_class = 2  # 目标类别的索引
#
# importance_masks, tracked_importance = find_important_regions(feature_maps, tracker, logits, target_class)
#
# print("Tracked Importance:", tracked_importance)
# # 最后的 Total Importance 是原始追踪点在图像分类中的整体重要性评分

import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog, Toplevel, StringVar
from PIL import Image, ImageDraw
import pandas as pd
from openpyxl import Workbook

def load_model():
    model = CrossFormer(
        dim=(32, 64, 128, 256),
        depth=(2, 2, 2, 2),
        global_window_size=(8, 4, 2, 1),
        local_window_size=16,
        cross_embed_kernel_sizes=((2, 4, 6, 8), (2, 4), (2, 4), (2, 4)),
        cross_embed_strides=(2, 2, 2, 2),
        num_classes=10,
        attn_dropout=0.2,
        ff_dropout=0.2,
        channels=3).to(device)

    # 加载状态字典
    state_dict = torch.load(r"E:\dataset\mut_best_model.pth")

    # 将状态字典加载到模型中
    model.load_state_dict(state_dict)
    model.eval()
    return model

def load_image():
    global image_path
    image_path = filedialog.askopenfilename(title="选择图片文件", filetypes=[("Image Files", "*.png;*.jpg;*.jpeg")])
    if image_path:
        img_path_entry.delete(0, tk.END)
        img_path_entry.insert(0, image_path)
        generate_excel()  # 加载图片后生成Excel文件

def load_excel():
    global exc_path
    exc_path = filedialog.askopenfilename(title="选择Excel文件", filetypes=[("Excel Files", "*.xlsx")])
    if exc_path:
        excel_path_entry.delete(0, tk.END)
        excel_path_entry.insert(0, exc_path)

def select_target_class():
    def set_target_class():
        global target_class
        target_class = int(target_class_var.get())
        target_class_window.destroy()

    global target_class_window
    target_class_window = Toplevel(root)
    target_class_window.title("选择目标类别")
    target_class_window.geometry("600x600")

    target_class_var = StringVar(value="0")

    tk.Label(target_class_window, text="请选择目标类别的索引:").pack(pady=10)
    tk.Radiobutton(target_class_window, text="glioma=0", variable=target_class_var, value="0").pack(anchor=tk.W)
    tk.Radiobutton(target_class_window, text="meningioma=1", variable=target_class_var, value="1").pack(anchor=tk.W)
    tk.Radiobutton(target_class_window, text="MildDemented=2", variable=target_class_var, value="2").pack(anchor=tk.W)
    tk.Radiobutton(target_class_window, text="ModerateDemented=3", variable=target_class_var, value="3").pack(anchor=tk.W)
    tk.Radiobutton(target_class_window, text="no-tumor=4", variable=target_class_var, value="4").pack(anchor=tk.W)
    tk.Radiobutton(target_class_window, text="NonDemented=5", variable=target_class_var, value="5").pack(anchor=tk.W)
    tk.Radiobutton(target_class_window, text="Nonstroke=6", variable=target_class_var, value="6").pack(anchor=tk.W)
    tk.Radiobutton(target_class_window, text="pituitary=7", variable=target_class_var, value="7").pack(anchor=tk.W)
    tk.Radiobutton(target_class_window, text="Stroke=8", variable=target_class_var, value="8").pack(anchor=tk.W)
    tk.Radiobutton(target_class_window, text="VeryMildDemented=9", variable=target_class_var, value="9").pack(anchor=tk.W)

    tk.Button(target_class_window, text="确定", command=set_target_class).pack(pady=20)

def generate_excel():
    try:
        model = load_model()
        image = Image.open(image_path).convert('RGB')

        preprocess = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor()
        ])

        input_tensor = preprocess(image).unsqueeze(0).to(device)

        # 获取用户输入的target_class
        select_target_class()
        root.wait_window(target_class_window)

        # 获取用户输入的重要性阈值
        importance_threshold = simpledialog.askfloat("输入", "请输入重要性阈值:(推荐为2,2为比较重要的数)")

        # 创建一个新的工作簿和工作表
        wb = Workbook()
        ws = wb.active
        ws.title = "Tracked Importance"

        # 写入表头
        ws.append(["Coordinate", "Tracked Importance"])

        for i in range(70, 80):
            for j in range(30, 40):
                initial_coords = (i, j)
                tracker = TrackPoint(initial_coords)
                output, feature_maps = model(input_tensor, tracker)
                tracked_coords = tracker.get_coords()
                tracker_points = [tracked_coords[1], tracked_coords[13], tracked_coords[14], tracked_coords[26],
                                  tracked_coords[27], tracked_coords[39], tracked_coords[40], tracked_coords[47]]
                importance_masks, total_importance = find_important_regions(feature_maps, tracker_points, output, target_class)

                # 如果总重要性大于用户输入的阈值，将坐标转换为字符串格式并逐条写入 Excel 文件
                if total_importance > importance_threshold:
                    ws.append([str(initial_coords), total_importance])

        save_path = filedialog.asksaveasfilename(defaultextension=".xlsx", filetypes=[("Excel Files", "*.xlsx")])
        if save_path:
            wb.save(save_path)
            global exc_path
            exc_path = save_path
            excel_path_entry.delete(0, tk.END)
            excel_path_entry.insert(0, save_path)
            messagebox.showinfo("完成", f"Excel文件已保存为：{save_path}")

    except Exception as e:
        messagebox.showerror("错误", f"发生错误: {e}")

def mark_image(exc_path, image_path):
    try:
        # 读取Excel文件和图像
        df = pd.read_excel(exc_path)
        image = Image.open(image_path).convert('RGB')

        # 预处理图像
        preprocess = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor()
        ])
        input_tensor = preprocess(image)
        image_resized = transforms.ToPILImage()(input_tensor)

        # 在图像上标注点
        draw = ImageDraw.Draw(image_resized)
        for coord in df.iloc[:, 0]:
            x, y = map(int, coord.strip('()').split(','))
            if 25 <= x < 226 and 18 <= y < 233:
                draw.point((x, y), fill='red')

        # 保存标注后的图像
        save_path = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG Files", "*.png")])
        if save_path:
            image_resized.save(save_path)
            messagebox.showinfo("完成", f"标注后的图片已保存为：{save_path}")

        # 使用OpenCV处理图像
        original_image = cv2.imread(image_path)
        annotated_image = cv2.imread(save_path)

        # 转换为灰度图像
        gray = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)

        # 应用阈值以创建二值图像
        _, binary = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)

        # 查找轮廓
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # 创建一个全黑的掩码
        mask = np.zeros_like(gray)

        # 绘制大脑区域的轮廓
        cv2.drawContours(mask, contours, -1, 255, thickness=cv2.FILLED)

        # 调整掩码的尺寸以匹配标注后的图像
        mask_resized = cv2.resize(mask, (annotated_image.shape[1], annotated_image.shape[0]))

        # 使用掩码将大脑区域复制到标注后的图像上
        brain_region = cv2.bitwise_and(annotated_image, annotated_image, mask=mask_resized)

        # 将大脑区域外的部分变为黑色
        final_image = np.where(mask_resized[:, :, np.newaxis] == 0, 0, brain_region)

        # 显示结果
        cv2.imshow('Result', final_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    except Exception as e:
        messagebox.showerror("错误", f"发生错误: {e}")

# 创建主窗口
root = tk.Tk()
root.title("CVTC 图片标注工具")
root.geometry("1000x500")

# 图片路径
tk.Label(root, text="图片路径:").pack(pady=5)
img_path_entry = tk.Entry(root, width=50)
img_path_entry.pack(pady=5)
tk.Button(root, text="选择图片文件", command=load_image).pack(pady=5)

# Excel文件路径
tk.Label(root, text="Excel路径:").pack(pady=5)
excel_path_entry = tk.Entry(root, width=50)
excel_path_entry.pack(pady=5)
tk.Button(root, text="选择Excel文件", command=load_excel).pack(pady=5)

# 标注按钮
tk.Button(root, text="标注图片", command=lambda: mark_image(exc_path, img_path_entry.get()), bg="lightblue").pack(pady=20)

# 提示信息标签
tk.Label(root, text="类别索引提示:\nglioma=0，meningioma=1，MildDemented=2，ModerateDemented=3，no-tumor=4，NonDemented=5，Nonstroke=6，pituitary=7，Stroke=8，VeryMildDemented=9").pack(pady=10)

# 运行主循环
root.mainloop()
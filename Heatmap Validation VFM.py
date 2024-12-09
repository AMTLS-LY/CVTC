from idlelib.pyparse import trans

import torch
from torch import nn,einsum,optim
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, recall_score
from torchvision import datasets,transforms



device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
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

kernel_size=[3,5,7]
model=CrossEmbedLayer(dim_in=3,dim_out=256,kernel_size=kernel_size,stride=2)

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
    def __init__(self, dim, attn_type, window_size, dim_head=32, dropout=0.):
        super(Attention, self).__init__()
        assert attn_type in {'short', 'long'}, 'attention type 必须是long或者short'

        heads = dim // dim_head
        assert dim >= dim_head, 'dim 必须大于等于 dim_head'
        if heads == 0:
            raise ValueError('heads 不能为零，请确保 dim >= dim_head')
        self.heads = heads
        self.dim_head = dim_head
        self.scale = dim_head ** -0.5
        inner_dim = dim_head * heads

        self.attn_type = attn_type
        self.window_size = window_size
        self.norm = LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Conv2d(dim, inner_dim * 3, 1, bias=False)
        self.to_out = nn.Conv2d(inner_dim, dim, 1)

        self.dpb = DynamicPositionBias(dim // 4)

        pos = torch.arange(window_size)
        grid = torch.stack(torch.meshgrid(pos, pos, indexing='ij'))
        _, w1, w2 = grid.size()
        grid = grid.view(-1, w1 * w2).permute(1, 0).contiguous()
        real_pos = grid.view(w1 * w2, 1, 2) - grid.view(1, w1 * w2, 2)
        real_pos = real_pos + window_size - 1

        rel_pos_indices = (real_pos * torch.tensor([2 * window_size - 1, 1])).sum(dim=-1)
        self.register_buffer('rel_pos_indices', rel_pos_indices, persistent=False)

    def forward(self, x, return_attention=False):
        b, dim, h, w, heads, wsz, device = *x.shape, self.heads, self.window_size, x.device
        x = self.norm(x)
        if self.attn_type == 'short':
            x = x.view(b, dim, h // wsz, wsz, w // wsz, wsz)
            x = x.permute(0, 2, 4, 1, 3, 5).contiguous()
            x = x.view(-1, dim, wsz, wsz)
        elif self.attn_type == 'long':
            x = x.view(b, dim, h // wsz, wsz, w // wsz, wsz)
            x = x.permute(0, 2, 4, 1, 3, 5).contiguous()
            x = x.view(-1, dim, wsz, wsz)

        q, k, v = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(lambda x: x.view(-1, self.heads, wsz * wsz, self.dim_head), (q, k, v))
        q = q * self.scale

        sim = einsum('b h i d, b h j d -> b h i j', q, k)

        pos = torch.arange(-wsz, wsz + 1, device=device)
        rel_pos = torch.stack(torch.meshgrid(pos, pos, indexing='ij'))
        _, size1, size2 = rel_pos.size()
        rel_pos = rel_pos.permute(1, 2, 0).view(size1 * size2, 2)
        biases = self.dpb(rel_pos.float())
        rel_pos_bias = biases[self.rel_pos_indices]

        sim = sim + rel_pos_bias

        attn = sim.softmax(dim=-1)
        attn = self.dropout(attn)
        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = out.permute(0, 1, 3, 2).contiguous().view(-1, self.heads * self.dim_head, wsz, wsz)
        out = self.to_out(out)

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


model=Attention(dim=64,attn_type='short',window_size=8,dim_head=8)

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
model=Transformer(dim=8,local_window_size=8,global_window_size=16,depth=2,dim_head=8)

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

    def forward(self, x):
        for cel, transformer in self.layers:
            x = cel(x)
            x = transformer(x)
        x = torch.einsum('b c h w -> b c', x) / (x.shape[2] * x.shape[3])
        return self.to_logits(x)

    def get_attention_weights(self, x, transformer_layer_index, attention_layer_index, attention_index):
        attentions = []
        num_layers = len(self.layers)
        # 处理负索引
        if transformer_layer_index < 0:
            transformer_layer_index += num_layers
        for i, (cel, transformer) in enumerate(self.layers):
            x = cel(x)
            x = transformer(x)
            if i == transformer_layer_index:
                # 获取指定 Transformer 层的指定 Attention 层的注意力权重
                attention_layer = transformer.layers[attention_layer_index][attention_index]
                _, attention_weights = attention_layer(x, return_attention=True)
                attentions.append(attention_weights)
                break  # 找到指定层后退出循环
        return attentions

    def extract_features(self, x):
        for cel, transformer in self.layers:
            x = cel(x)
            x = transformer(x)
        # 使用 einsum 进行平均池化
        features = torch.einsum('b c h w -> b c', x) / (x.shape[2] * x.shape[3])
        features = self.feature_reducer(features)  # 降维
        return features


import torch
from torch import nn
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

model = CrossFormer(
    dim=(32, 64, 128, 256),  # 确保这些参数与训练时一致
    depth=(2, 2, 2, 2),
    global_window_size=(8, 4, 2, 1),
    local_window_size=16,
    cross_embed_kernel_sizes=((2, 4, 6, 8), (2, 4), (2, 4), (2, 4)),
    cross_embed_strides=(2, 2, 2, 2),
    num_classes=4,
    attn_dropout=0.2,
    ff_dropout=0.2,
    channels=3
).to(device)

try:
    model.load_state_dict(torch.load(r"E:\dataset\best_model.pth", map_location=device))
except Exception as e:
    print(f"Error loading model: {e}")
    exit()


import matplotlib.pyplot as plt
import matplotlib.cm as cm
def visualize_attention(model, image_path, transformer_layer_index, attention_layer_index, attention_index):
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])

    image = Image.open(image_path).convert('RGB')  # 保持 RGB 通道
    image_tensor = transform(image).unsqueeze(0).float().to(device)

    model.eval()
    with torch.no_grad():
        attentions = model.get_attention_weights(image_tensor, transformer_layer_index, attention_layer_index, attention_index)
        if not attentions:
            print(f"No attention weights found for transformer layer {transformer_layer_index} and attention layer {attention_layer_index}")
            return
        attention_weights = attentions[0].mean(dim=1).cpu().numpy()

    attention_map = attention_weights.squeeze()
    attention_map = np.interp(attention_map, (attention_map.min(), attention_map.max()), (0, 255)).astype(np.uint8)

    if attention_map.ndim == 3:
        attention_map = attention_map[0]

    attention_map = np.array(Image.fromarray(attention_map).resize((image.size[0], image.size[1]), Image.BILINEAR))

    # 使用 'RdYlGn' 颜色映射
    cmap = cm.get_cmap('RdYlGn')
    heatmap = cmap(attention_map)[:, :, :3] * 255

    # 保留大脑MRI图像的形状
    mask = np.array(image.convert('L')) > 0  # 创建一个掩码，保留非黑色区域
    overlay = np.zeros_like(np.array(image))
    overlay[mask] = (0.5 * np.array(image)[mask] + 0.5 * heatmap[mask]).astype(np.uint8)  # 仅在掩码区域内叠加热图

    # 创建一个带颜色条的图像
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(overlay)
    ax.axis('off')

    # 添加颜色条
    cbar = plt.colorbar(cm.ScalarMappable(cmap=cmap), ax=ax, orientation='vertical')
    cbar.set_label('Attention Intensity')

    plt.show()

# def visualize_attention(model, image_path, transformer_layer_index, attention_layer_index, attention_index):
#     transform = transforms.Compose([
#         transforms.Resize((256, 256)),
#         transforms.ToTensor()
#     ])
#
#     image = Image.open(image_path).convert('RGB')  # 保持 RGB 通道
#     image_tensor = transform(image).unsqueeze(0).float().to(device)
#
#     model.eval()
#     with torch.no_grad():
#         attentions = model.get_attention_weights(image_tensor, transformer_layer_index, attention_layer_index, attention_index)
#         if not attentions:
#             print(f"No attention weights found for transformer layer {transformer_layer_index} and attention layer {attention_layer_index}")
#             return
#         attention_weights = attentions[0].mean(dim=1).cpu().numpy()
#
#     attention_map = attention_weights.squeeze()
#     attention_map = np.interp(attention_map, (attention_map.min(), attention_map.max()), (0, 255)).astype(np.uint8)
#
#     if attention_map.ndim == 3:
#         attention_map = attention_map[0]
#
#     attention_map = np.array(Image.fromarray(attention_map).resize((image.size[0], image.size[1]), Image.BILINEAR))
#
#     # 使用 'jet' 颜色映射
#     cmap = cm.get_cmap('jet')
#     heatmap = cmap(attention_map)[:, :, :3] * 255
#
#     # 保留大脑MRI图像的形状
#     mask = np.array(image.convert('L')) > 0  # 创建一个掩码，保留非黑色区域
#     overlay = np.zeros_like(np.array(image))
#     overlay[mask] = (0.5 * np.array(image)[mask] + 0.5 * heatmap[mask]).astype(np.uint8)  # 仅在掩码区域内叠加热图
#
#     # 创建一个带颜色条的图像
#     fig, ax = plt.subplots(figsize=(10, 10))
#     ax.imshow(overlay)
#     ax.axis('off')
#
#     # 添加颜色条
#     cbar = plt.colorbar(cm.ScalarMappable(cmap=cmap), ax=ax, orientation='vertical')
#     cbar.set_label('Attention Intensity')
#
#     plt.show()

# 局部注意
visualize_attention(model, r"E:\dataset\OriginalDataset\MildDemented\mildDem65.jpg", transformer_layer_index=-4, attention_layer_index=1,attention_index=0)
# 全局注意
visualize_attention(model, r"E:\dataset\OriginalDataset\MildDemented\mildDem65.jpg", transformer_layer_index=-4, attention_layer_index=1,attention_index=2)
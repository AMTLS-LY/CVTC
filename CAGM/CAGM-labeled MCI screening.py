import torch
from torch import nn, einsum
import torch.nn.functional as F
import cv2
import numpy as np
from PIL import Image, ImageDraw
import torchvision.transforms as transforms
import pandas as pd
from openpyxl import Workbook
import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog, Toplevel, StringVar
import os

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

# CrossFormer（支持批量坐标跟踪）
class CrossFormer(nn.Module):
    def __init__(self, dim=(64, 128, 256, 512), depth=(2, 2, 8, 2), global_window_size=(8, 4, 2, 1),
                 local_window_size=16, cross_embed_kernel_sizes=((4, 8, 16, 32), (2, 4), (2, 4), (2, 4)),
                 cross_embed_strides=(4, 2, 2, 2), num_classes=10, attn_dropout=0., ff_dropout=0., channels=3, feature_dim=20):
        super(CrossFormer, self).__init__()
        dim = cast_tuple(dim, 4)
        depth = cast_tuple(depth, 4)
        global_window_size = cast_tuple(global_window_size, 4)
        local_window_size = cast_tuple(local_window_size, 4)
        cross_embed_kernel_sizes = cast_tuple(cross_embed_kernel_sizes, 4)
        cross_embed_strides = cast_tuple(cross_embed_strides, 4)
        last_dim = dim[-1]
        dims = [channels, *dim]
        dim_in_and_out = tuple(zip(dims[:-1], dims[1:]))
        self.layers = nn.ModuleList([
            nn.ModuleList([
                CrossEmbedLayer(dim_in, dim_out, cel_kernel_sizes, stride=cel_stride),
                Transformer(dim_out, local_window_size=local_wsz, global_window_size=global_wsz, depth=layers,
                            attn_dropout=attn_dropout, ff_dropout=ff_dropout)
            ]) for (dim_in, dim_out), layers, global_wsz, local_wsz, cel_kernel_sizes, cel_stride in zip(
                dim_in_and_out, depth, global_window_size, local_window_size, cross_embed_kernel_sizes, cross_embed_strides)
        ])
        self.to_logits = nn.Sequential(nn.Linear(last_dim, num_classes))
        self.feature_reducer = nn.Linear(last_dim, feature_dim)

    def forward(self, x, tracker=None):
        feature_maps = []
        x = x.requires_grad_(True)
        for cel, transformer in self.layers:
            x = cel(x)
            if tracker:
                coords_list = [track_point_in_layer_batch(conv, tracker.get_coords()[-1], x.shape) for conv in cel.conv]
                # 调试：打印 coords_list 的结构
                print("coords_list 长度:", len(coords_list))
                print("coords_list[0] 长度:", len(coords_list[0]))
                # 计算每个点的平均坐标
                num_coords = len(coords_list[0])  # 应该是 43215
                num_convs = len(coords_list)      # 卷积层数量
                avg_coords = []
                for i in range(num_coords):
                    avg_x = sum(coords[i][0] for coords in coords_list) // num_convs
                    avg_y = sum(coords[i][1] for coords in coords_list) // num_convs
                    avg_coords.append((avg_x, avg_y))
                tracker.update_coords(avg_coords)
            feature_maps.append(x)
            z = x
            for layer in transformer.layers:
                short_attn, short_ff, long_attn, long_ff = layer
                y, attn = short_attn(z, return_attention=True)
                if tracker:
                    new_coords = track_point_in_attention_batch(attn, tracker.get_coords()[-1], y.shape)
                    tracker.update_coords(new_coords)
                y = short_ff(y)
                if tracker:
                    new_coords = track_point_in_layer_batch(short_ff[1], tracker.get_coords()[-1], y.shape)
                    tracker.update_coords(new_coords)
                    new_coords = track_point_in_layer_batch(short_ff[4], tracker.get_coords()[-1], y.shape)
                    tracker.update_coords(new_coords)
                y, attn = long_attn(y, return_attention=True)
                if tracker:
                    new_coords = track_point_in_attention_batch(attn, tracker.get_coords()[-1], y.shape)
                    tracker.update_coords(new_coords)
                y = long_ff(y)
                if tracker:
                    new_coords = track_point_in_layer_batch(long_ff[1], tracker.get_coords()[-1], y.shape)
                    tracker.update_coords(new_coords)
                    new_coords = track_point_in_layer_batch(long_ff[4], tracker.get_coords()[-1], y.shape)
                    tracker.update_coords(new_coords)
                z = y
            x = transformer(x)
            feature_maps.append(x)
        x = torch.einsum('b c h w -> b c', x) / (x.shape[2] * x.shape[3])
        return self.to_logits(x), feature_maps

# 批量坐标跟踪器
class BatchTrackPoint:
    def __init__(self, initial_coords_list):
        self.coords = [initial_coords_list]  # [num_coords, 2]

    def update_coords(self, new_coords_list, ap=True):
        if ap:
            self.coords.append(new_coords_list)

    def get_coords(self):
        return self.coords

# 批量跟踪函数
def track_point_in_layer_batch(layer, coords_batch, input_shape):
    batch_size, channels, height, width = input_shape
    coords_tensor = torch.tensor(coords_batch, device=device)
    x, y = coords_tensor[:, 0], coords_tensor[:, 1]
    # 确保 padding、kernel_size 和 stride 是元组
    padding = cast_tuple(layer.padding, 2)  # 转换为长度为 2 的元组
    kernel_size = cast_tuple(layer.kernel_size, 2)  # 转换为长度为 2 的元组
    stride = cast_tuple(layer.stride, 2)  # 转换为长度为 2 的元组
    new_x = (x + padding[0] - kernel_size[0] // 2) // stride[0]
    new_y = (y + padding[1] - kernel_size[1] // 2) // stride[1]
    return torch.stack([new_x, new_y], dim=1).tolist()

def track_point_in_attention_batch(attention_weights, coords_batch, input_shape):
    batch_size, channels, height, width = input_shape
    batch_size, num_heads, seq_len, _ = attention_weights.shape
    coords_tensor = torch.tensor(coords_batch, device=device)  # [num_coords, 2]
    x, y = coords_tensor[:, 0], coords_tensor[:, 1]
    # 确保 y 在有效范围内
    y = y.clamp(0, seq_len - 1)  # 限制 y 的范围，避免越界
    valid_mask = (y >= 0) & (y < seq_len)
    target_weights = attention_weights[0, :, y.long(), :]  # [num_heads, num_coords, seq_len]
    avg_coords = target_weights.mean(dim=0).argmax(dim=-1)  # [num_coords]
    new_x = torch.where(valid_mask, avg_coords, x)
    return torch.stack([new_x, y], dim=1).tolist()

# 计算全局重要性掩码（修复：添加 detach()）
def get_global_importance_mask(feature_maps, logits, target_class):
    probs = F.softmax(logits, dim=-1)
    target_prob = probs[:, target_class].sum()
    global_mask = None
    for feature_map in feature_maps:
        feature_map.retain_grad()
        grad = torch.autograd.grad(target_prob, feature_map, retain_graph=True, create_graph=True)[0]
        mask = torch.sum(grad * feature_map, dim=1, keepdim=True)
        mask = F.interpolate(mask, size=(256, 256), mode='bilinear', align_corners=False)
        if global_mask is None:
            global_mask = mask
        else:
            global_mask += mask
    # 使用 detach() 分离张量后再转换为 NumPy 数组
    return global_mask.squeeze().detach().cpu().numpy()

# 批量计算重要性
def find_important_regions_batch(feature_maps, tracker_coords_batch, logits, target_class):
    probs = F.softmax(logits, dim=-1)
    target_prob = probs[:, target_class].sum()
    importance_masks = []
    for feature_map in feature_maps:
        feature_map.retain_grad()  # 确保 feature_map 保留梯度
        grad = torch.autograd.grad(target_prob, feature_map, retain_graph=True, create_graph=True)[0]
        importance_mask = torch.sum(grad * feature_map, dim=1, keepdim=True)
        importance_masks.append(importance_mask)
    coords_tensor = torch.tensor(tracker_coords_batch, device=device)
    importance_weights = []
    for i, mask in enumerate(importance_masks):
        x = coords_tensor[:, i, 0].long()
        y = coords_tensor[:, i, 1].long()
        valid_mask = (x >= 0) & (x < mask.shape[3]) & (y >= 0) & (y < mask.shape[2])
        weights = torch.zeros(len(x), device=device)
        weights[valid_mask] = mask[0, 0, y[valid_mask], x[valid_mask]]
        importance_weights.append(weights)
    importance_weights = torch.stack(importance_weights, dim=1)
    max_importance = importance_weights.max(dim=1)[0] + 1e-10
    normalized_importance = importance_weights / max_importance.unsqueeze(1)
    total_importance = normalized_importance.sum(dim=1)
    return importance_masks, total_importance.tolist()

# GUI相关函数
def load_model():
    model = CrossFormer(
        dim=(32, 64, 128, 256), depth=(2, 2, 2, 2), global_window_size=(8, 4, 2, 1), local_window_size=16,
        cross_embed_kernel_sizes=((2, 4, 6, 8), (2, 4), (2, 4), (2, 4)), cross_embed_strides=(2, 2, 2, 2),
        num_classes=3, attn_dropout=0.2, ff_dropout=0.2, channels=3).to(device)
    state_dict = torch.load(r"E:\dataset-CVTC\权重\MCI(1).pth")
    model.load_state_dict(state_dict)
    model.eval()
    return model

def load_image():
    global image_path
    image_path = filedialog.askopenfilename(title="选择图片文件", filetypes=[("Image Files", "*.png;*.jpg;*.jpeg")])
    if image_path:
        img_path_entry.delete(0, tk.END)
        img_path_entry.insert(0, image_path)
        generate_excel()

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
    # 修改为3个类别
    for i, label in enumerate(["AlzheimersDisease=0", "CognitivelyNormal=1", "MildCognitiveImpairment=2"]):
        tk.Radiobutton(target_class_window, text=label, variable=target_class_var, value=str(i)).pack(anchor=tk.W)
    tk.Button(target_class_window, text="确定", command=set_target_class).pack(pady=20)


def generate_excel():
    model = load_model()
    image = Image.open(image_path).convert('RGB')
    preprocess = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])
    input_tensor = preprocess(image).unsqueeze(0).to(device)

    # 弹出类别选择窗口并等待
    select_target_class()
    root.wait_window(target_class_window)

    # 将图像转换为灰度图并检测大脑轮廓
    img_np = np.array(image.resize((256, 256)))  # 转换为numpy数组
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)  # 转换为灰度图
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)  # 二值化
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  # 查找轮廓

    # 选择最大的轮廓（假设是大脑区域）
    if contours:
        brain_contour = max(contours, key=cv2.contourArea)
        # 创建掩码，只保留轮廓内的区域
        mask = np.zeros((256, 256), dtype=np.uint8)
        cv2.drawContours(mask, [brain_contour], -1, (255), thickness=cv2.FILLED)
    else:
        # 如果未检测到轮廓，使用整个图像
        mask = np.ones((256, 256), dtype=np.uint8) * 255

    # 生成所有初始坐标并筛选出轮廓内的点
    coords = [(i, j) for i in range(25, 226) for j in range(18, 233) if mask[i, j] > 0]
    tracker = BatchTrackPoint(coords)
    output, feature_maps = model(input_tensor, tracker)

    # 计算全局重要性掩码并筛选高重要性区域（仅在轮廓内）
    global_mask = get_global_importance_mask(feature_maps, output, target_class)
    threshold = np.percentile(global_mask, 80)  # 初步筛选前20%区域
    high_importance_coords = [(i, j) for i, j in coords if global_mask[i, j] > threshold]

    # 批量计算重要性得分
    tracker_coords_batch = []
    coord_to_idx = {coord: idx for idx, coord in enumerate(coords)}
    for coord in high_importance_coords:
        idx = coord_to_idx[coord]
        tracker_coords_batch.append([
            tracker.get_coords()[1][idx], tracker.get_coords()[13][idx],
            tracker.get_coords()[14][idx], tracker.get_coords()[26][idx],
            tracker.get_coords()[27][idx], tracker.get_coords()[39][idx],
            tracker.get_coords()[40][idx], tracker.get_coords()[47][idx]
        ])

    total_importance_batch = []
    if tracker_coords_batch:
        _, total_importance_batch = find_important_regions_batch(feature_maps, tracker_coords_batch, output, target_class)
        scores = np.array(total_importance_batch)
        valid_scores = scores[scores > 0]  # 过滤零值
    else:
        valid_scores = np.array([])

    # 多种阈值计算方法
    threshold_methods = {}
    best_threshold = 0
    best_method = "no_data"

    if len(valid_scores) > 0:
        threshold_methods = {}
        target_ratio = 0.25

        try:
            from skimage.filters import threshold_otsu
            if valid_scores.max() > valid_scores.min():
                otsu_thresh = threshold_otsu(valid_scores)
                threshold_methods['otsu_adj'] = otsu_thresh * 0.9
        except (ImportError, Exception) as e:
            print(f"Otsu阈值不可用: {str(e)}")

        median = np.median(valid_scores)
        mad = 1.4826 * np.median(np.abs(valid_scores - median))
        threshold_methods['mad_2sigma'] = median + 2 * mad

        q75 = np.percentile(valid_scores, 75)
        q25 = np.percentile(valid_scores, 25)
        iqr = q75 - q25
        dynamic_percentile = 85 if iqr > (q75 * 0.2) else 92
        threshold_methods[f'percentile_{dynamic_percentile}'] = np.percentile(valid_scores, dynamic_percentile)

        try:
            from sklearn.mixture import GaussianMixture
            if len(valid_scores) >= 10:
                scores_reshaped = valid_scores.reshape(-1, 1)
                gmm = GaussianMixture(n_components=3, tol=1e-4, random_state=0).fit(scores_reshaped)
                means = sorted(gmm.means_.flatten())
                if len(means) >= 2:
                    threshold_methods['gmm_mid'] = np.mean(means[-2:])
        except (ImportError, Exception) as e:
            print(f"GMM不可用: {str(e)}")

        if len(valid_scores) >= 2:
            mean_score = np.mean(valid_scores)
            std_score = np.std(valid_scores)
            threshold_methods['mean_1std'] = mean_score + std_score

        max_quality = -np.inf
        for method_name, thresh in threshold_methods.items():
            selected = valid_scores[valid_scores > thresh]
            if len(selected) == 0:
                continue
            area_ratio = len(selected) / len(valid_scores)
            mean_score = np.mean(selected)
            std_score = np.std(selected)
            quality_score = (
                mean_score * 0.4 +
                (1 - abs(area_ratio - target_ratio)) * 0.5 +
                (1 / (std_score + 1e-7)) * 0.1
            )
            if quality_score > max_quality:
                max_quality = quality_score
                best_threshold = thresh
                best_method = method_name

        if best_method == "no_data" or len(valid_scores) == 0:
            fallback_percentile = 80
            best_threshold = np.percentile(valid_scores, fallback_percentile) if len(valid_scores) > 0 else 0
            best_method = f"percentile_fallback_{fallback_percentile}"

        final_selected = valid_scores[valid_scores > best_threshold]
        if len(final_selected) / len(valid_scores) < 0.1:
            best_threshold = np.percentile(valid_scores, 80)
            best_method = "force_80_percentile"

    # 生成Excel文件
    wb = Workbook()
    ws = wb.active
    ws.title = "Tracked Importance"
    headers = ["Coordinate", "Importance", "Threshold Method", "Global Mask Value"]
    ws.append(headers)

    for coord, importance in zip(high_importance_coords, total_importance_batch):
        if importance > best_threshold:
            global_val = global_mask[coord[0], coord[1]]
            ws.append([
                str(coord),
                round(importance, 4),
                best_method,
                round(float(global_val), 4)
            ])

    # 保存文件
    save_path = filedialog.asksaveasfilename(
        defaultextension=".xlsx",
        filetypes=[("Excel Files", "*.xlsx")]
    )
    if save_path:
        wb.save(save_path)
        global exc_path
        exc_path = save_path
        excel_path_entry.delete(0, tk.END)
        excel_path_entry.insert(0, save_path)

        stats_msg = [
            f"自动选择方法: {best_method}",
            f"最终阈值: {best_threshold:.4f}",
            f"高重要性区域: {len(high_importance_coords)}",
            f"筛选后区域: {ws.max_row - 1}",
            f"平均重要性: {np.mean(valid_scores):.4f}" if len(valid_scores) > 0 else ""
        ]
        messagebox.showinfo("分析结果", "\n".join(stats_msg))

def mark_image(exc_path, image_path):
    # 读取Excel文件和图像
    df = pd.read_excel(exc_path)
    image = Image.open(image_path).convert('RGB')

    # 预处理图像
    preprocess = transforms.Compose([transforms.Resize((256, 256)), transforms.ToTensor()])
    input_tensor = preprocess(image)
    image_resized = transforms.ToPILImage()(input_tensor)

    # 转换为OpenCV格式
    image_cv = cv2.cvtColor(np.array(image_resized), cv2.COLOR_RGB2BGR)

    # 创建一个空白图像用于热图
    heatmap = np.zeros((256, 256), dtype=np.float32)

    # 在热图上标注点
    for coord in df.iloc[:, 0]:
        x, y = map(int, coord.strip('()').split(','))
        if 25 <= x < 226 and 18 <= y < 233:
            heatmap[x, y] = 255  # 将点的值设为最大值

    # 应用高斯模糊使热图平滑
    heatmap = cv2.GaussianBlur(heatmap, (15, 15), 0)

    # 将热图转换为彩色
    heatmap_color = cv2.applyColorMap(heatmap.astype(np.uint8), cv2.COLORMAP_JET)

    # 创建一个透明图层
    transparent_layer = np.zeros_like(image_cv, dtype=np.uint8)
    transparent_layer[heatmap > 0] = heatmap_color[heatmap > 0]

    # 将热图叠加到原始图像上
    overlay = cv2.addWeighted(image_cv, 1.0, transparent_layer, 0.6, 0)

    # 保存热图叠加后的图像
    save_path = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG Files", "*.png")])
    if save_path:
        cv2.imwrite(save_path, overlay)  # 保存热图叠加图像
        if not os.path.exists(save_path):
            raise FileNotFoundError(f"保存文件失败：{save_path}")
        messagebox.showinfo("完成", f"标注后的图片已保存为：{save_path}")

    # 显示热图叠加结果
    cv2.imshow('Result', overlay)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# 主窗口
root = tk.Tk()
root.title("CVTC 图片标注工具")
root.geometry("1000x500")
tk.Label(root, text="图片路径:").pack(pady=5)
img_path_entry = tk.Entry(root, width=50)
img_path_entry.pack(pady=5)
tk.Button(root, text="选择图片文件", command=load_image).pack(pady=5)
tk.Label(root, text="Excel路径:").pack(pady=5)
excel_path_entry = tk.Entry(root, width=50)
excel_path_entry.pack(pady=5)
tk.Button(root, text="选择Excel文件", command=load_excel).pack(pady=5)
tk.Button(root, text="标注图片", command=lambda: mark_image(exc_path, img_path_entry.get()), bg="lightblue").pack(pady=20)
# 修改类别提示为3类
tk.Label(root, text="类别索引提示:\nAlzheimersDisease=0, CognitivelyNormal=1, MildCognitiveImpairment=2").pack(pady=10)
root.mainloop()
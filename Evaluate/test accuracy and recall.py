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

    def forward(self,x):
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
        num_classes = 4,
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
            # print(f'After CrossEmbedLayer: {x.size()}')
            x = transformer(x)
            # print(f'After Transformer: {x.size()}')
        # 使用 einsum 进行平均池化
        x = torch.einsum('b c h w -> b c', x) / (x.shape[2] * x.shape[3])
        return self.to_logits(x)

    def extract_features(self, x):
        for cel, transformer in self.layers:
            x = cel(x)
            x = transformer(x)
        # 使用 einsum 进行平均池化
        features = torch.einsum('b c h w -> b c', x) / (x.shape[2] * x.shape[3])
        features = self.feature_reducer(features)  # 降维
        return features


import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


# 定义数据预处理
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])

new_test_data = datasets.ImageFolder(root=r'D:\6666\CVT\OriginalDataset', transform=transform)
new_test_loader = DataLoader(new_test_data, batch_size=32, shuffle=False)

# 初始化模型
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
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
#
# 加载最佳模型权重
try:
    model.load_state_dict(torch.load(r'D:\6666\CVT\1.pth', map_location=device))
except Exception as e:
    print(f"Error loading model: {e}")
    exit()
#
model.eval()
with torch.no_grad():  # 对于分类问题，神经网络的预测结果并不是标签，而是不同标签的概率分布
    total_correct = 0
    total_sum = 0
    total_true_positive = 0
    total_actual_positive = 0

    for x, label in new_test_loader:
        x, label = x.to(device), label.to(device)
        logits = model(x)
        pred = logits.argmax(dim=1)

        total_correct += torch.eq(pred, label).float().sum().item()  # 使用item转换为numpy
        total_sum += x.size(0)

        # 计算召回率
        true_positive = torch.eq(pred, label) & torch.eq(label, 1)  # 假设正类标签为1
        total_true_positive += true_positive.float().sum().item()
        total_actual_positive += torch.eq(label, 1).float().sum().item()

    acc = total_correct / total_sum
    recall = total_true_positive / total_actual_positive if total_actual_positive > 0 else 0

    print(f'Accuracy: {acc:.4f}')
    print(f'Recall: {recall:.4f}')
#
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve
import numpy as np

model.eval()
with torch.no_grad():
    total_correct = 0
    total_sum = 0
    all_labels = []
    all_preds = []
    for x, label in new_test_loader:
        x, label = x.to(device), label.to(device)
        logits = model(x)
        probs = F.softmax(logits, dim=1)  # Apply softmax to get probabilities
        pred = logits.argmax(dim=1)
        total_correct += torch.eq(pred, label).float().sum().item()
        total_sum += x.size(0)
        all_labels.extend(label.cpu().numpy())
        all_preds.append(probs.cpu().numpy())

    all_labels = np.array(all_labels)
    all_preds = np.vstack(all_preds)  # Ensure all_preds is a 2D array

    acc = total_correct / total_sum
    auc = roc_auc_score(all_labels, all_preds, multi_class='ovr')
    print(f'Accuracy: {acc}, AUC: {auc}')

    # Plot ROC curve for each class
    fpr = {}
    tpr = {}
    for i in range(probs.shape[1]):
        fpr[i], tpr[i], _ = roc_curve(all_labels == i, all_preds[:, i])
        plt.plot(fpr[i], tpr[i], lw=2,
                 label=f'Class {i} (area = {roc_auc_score(all_labels == i, all_preds[:, i]):.2f})')

    plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.show()





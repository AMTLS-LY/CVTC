import torch
import torch.nn as nn
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt


class Block(nn.Module):
    def __init__(self, in_channels, out_channels, downsample_state=False,padding_0=1,stride_0=1,padding_3=1,stride_3=1,padding_down=1,stride_down=1,add=False):
        super(Block, self).__init__()
        self.add=add
        self.double_conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=padding_0, bias=False,stride=stride_0),
            nn.InstanceNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=padding_3, bias=False,stride=stride_3),
            nn.InstanceNorm3d(out_channels)
        )
        self.act = nn.ReLU(inplace=True)
        self.downsample = None
        if downsample_state:
            self.downsample = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=1, padding=padding_down,bias=False,stride=stride_down),
                nn.InstanceNorm3d(out_channels)
            )

    def forward(self, x):
        if self.downsample is not None:
            residual = self.downsample(x)
            x = self.double_conv(x)
            x = self.act(x + residual)  # 修改位置：将 act 放在 downsample 之后
        elif self.add==True:
            x1=x
            x = self.double_conv(x)
            x = self.act(x+x1)
        else:
            x = self.double_conv(x)
        return x


class Down(nn.Module):
    def __init__(self, in_channels, out_channels, downsample_state=False,b1_padding_0=1,b1_stride_0=1,b1_padding_3=1,b1_stride_3=1,b1_padding_down=1,b1_stride_down=1,b2_padding_0=1,b2_stride_0=1,b2_padding_3=1,b2_stride_3=1,b2_padding_down=1,b2_stride_down=1,add2=False,add1=False):
        super(Down, self).__init__()
        self.add1,self.add2=add1,add2
        self.block1 = Block(in_channels, out_channels, downsample_state=downsample_state,padding_0=b1_padding_0,padding_3=b1_padding_3,padding_down=b1_padding_down,stride_0=b1_stride_0,stride_3=b1_stride_3,stride_down=b1_stride_down,add=add1)
        self.block2 = Block(out_channels, out_channels,padding_0=b2_padding_0,padding_3=b2_padding_3,padding_down=b2_padding_down,stride_0=b2_stride_0,stride_3=b2_stride_3,stride_down=b2_stride_down,add=add2)
    def forward(self, x):
        x1=x = self.block1(x)
        x = self.block2(x)
        x = x + x1  # 添加相加操作
        x=nn.functional.relu(x)
        return x


class Up(nn.Module):
    def __init__(self, in_channels, out_channels,last=False,up_stride1=1,up_padding1=1,up_stride_tp=1,up_padding_tp=1,up_stride2=1,up_padding2=1,output_padding=1):
        super(Up, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=1, padding=up_padding1, bias=False,stride=up_stride1),
            nn.InstanceNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.tp_conv = nn.Sequential(
            nn.ConvTranspose3d(out_channels, out_channels, kernel_size=3, stride=up_stride_tp, padding=up_padding_tp, bias=False,output_padding=output_padding),
            nn.InstanceNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )
        if last==False:
            self.conv2 = nn.Sequential(
                nn.Conv3d(out_channels, out_channels*2, kernel_size=1, padding=up_padding2, bias=False,stride=up_stride2),
                nn.InstanceNorm3d(out_channels*2),
                nn.ReLU(inplace=True)
            )
        else:
            self.conv2 = nn.Sequential(
                nn.Conv3d(out_channels, out_channels *4, kernel_size=1,padding=up_padding2, bias=False,stride=up_stride2),
                nn.InstanceNorm3d(out_channels*4),
                nn.ReLU(inplace=True)
            )

    def forward(self, x):
        x = self.conv1(x)
        x = self.tp_conv(x)
        x = self.conv2(x)
        return x


class LinkNet3d(nn.Module):
    def __init__(self):
        super(LinkNet3d, self).__init__()
        self.init = nn.Sequential(
            nn.Conv3d(1, 16, kernel_size=7, stride=2, padding=3, bias=False),
            nn.InstanceNorm3d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=3, stride=2, padding=1,dilation=1)
        )
        self.down0 = Down(16, 16,add1=True,add2=False)
        self.down1 = Down(16, 32, downsample_state=True,b1_stride_down=2,b1_padding_down=0,b1_stride_0=2,b1_padding_0=1,b1_stride_3=1,b1_padding_3=1)
        self.down2 = Down(32, 64, downsample_state=True,b1_stride_down=2,b1_padding_down=0,b1_stride_0=2,b1_padding_0=1,b1_stride_3=1,b1_padding_3=1)
        self.down3 = Down(64, 128,downsample_state=True,b1_stride_down=2,b1_padding_down=0,b1_stride_0=2,b1_padding_0=1,b1_stride_3=1,b1_padding_3=1)
        self.up0 = Up(128, 32,up_stride1=1,up_padding1=0,up_stride_tp=2,up_padding_tp=1,up_stride2=1,up_padding2=0)  # 修改 in_channels 为 128，out_channels 为 64
        self.up1 = Up(64, 16,up_stride1=1,up_padding1=0,up_stride_tp=2,up_padding_tp=1,up_stride2=1,up_padding2=0)  # 修改 in_channels 为 64，out_channels 为 32
        self.up2 = Up(32, 8,up_stride1=1,up_padding1=0,up_stride_tp=2,up_padding_tp=1,up_stride2=1,up_padding2=0)  # 修改 in_channels 为 32，out_channels 为 16
        self.up3 = Up(16, 4,last=True,up_stride1=1,up_padding1=0,up_stride_tp=1,up_padding_tp=1,up_stride2=1,up_padding2=0,output_padding=0)  # 修改 in_channels 为 16，out_channels 为 8
        self.tp_conv1 = nn.Sequential(
            nn.ConvTranspose3d(16, 8, kernel_size=3, stride=2,padding=1,output_padding=1),
            nn.InstanceNorm3d(8),
            nn.ReLU(inplace=True)
        )
        self.conv = nn.Sequential(
            nn.Conv3d(8, 8, kernel_size=3, padding=1,stride=1),
            nn.InstanceNorm3d(8),
            nn.ReLU(inplace=True)
        )
        self.tp_conv2 = nn.ConvTranspose3d(8, 2, kernel_size=2, stride=2,padding=0,output_padding=1)

    def forward(self, x):
        y1=x = self.init(x)
        x = nn.functional.relu(x)
        x0 = self.down0(x)
        x1 = self.down1(x0)
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        # 打印尺寸
        #print(f"x2 size: {x2.size()}")
        #print(f"x3 size: {x3.size()}")
        # Update the order of up layers
        x = self.up0.conv1(x3)  # up0.conv1.0.weight
        x = self.up0.tp_conv(x)  # up0.tp_conv.0.weight
        x = self.up0.conv2(x)  # up0.conv2.0.weight
        #print(f"x size after up0.conv2(x): {x.size()}")
        x = torch.add(x, x2)

        # 打印尺寸
        #print(f"x size after up0: {x.size()}")
        x = self.up1.conv1(x)  # up1.conv1.0.weight
        x = self.up1.tp_conv(x)  # up1.tp_conv.0.weight
        x = self.up1.conv2(x)  # up1.conv2.0.weight
        x = torch.add(x, x1)

        x = self.up2.conv1(x)  # up2.conv1.0.weight
        x = self.up2.tp_conv(x)  # up2.tp_conv.0.weight
        x = self.up2.conv2(x)  # up2.conv2.0.weight
        x = torch.add(x, x0)

        x = self.up3.conv1(x)  # up3.conv1.0.weight
        x = self.up3.tp_conv(x)  # up3.tp_conv.0.weight
        x = self.up3.conv2(x)  # up3.conv2.0.weight
        x = torch.add(x, y1)

        x = self.tp_conv1(x)
        x = self.conv(x)
        x = self.tp_conv2(x)

        x = torch.softmax(x, dim=1)

        return x


# # 创建自定义模型实例并加载到设备
# model_recreated = LinkNet3d()
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# model_recreated.to(device)
#
# # # 加载原始模型
# model_path = r"E:\dataset\nfbs\bbox_model.pt"
# pre_trained_model = torch.jit.load(model_path)
#
# # 将预训练模型的权重复制到自定义模型中
# model_recreated.load_state_dict(pre_trained_model.state_dict())
# torch.save(model_recreated.state_dict(), 'bbox.pth')
#
# print("模型已成功保存为 model.pth 文件。")
# print("预训练模型的权重已成功加载到自定义模型中。")
#
#
# # 加载 NIfTI 文件
# nii_file_path = r"E:\dataset\subjects\OAS1_0001_MR1_1.nii"
# nii_data = nib.load(nii_file_path)
# nii_array = nii_data.get_fdata()
#
# # 将 NIfTI 数据转换为 PyTorch 张量
# input_tensor = torch.tensor(nii_array, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)  # 添加批次和通道维度
#
# # 使用模型进行推理
# model_recreated.eval()
# with torch.no_grad():
#     output_tensor = model_recreated(input_tensor)
#
# # 获取分割结果并显示其中一个切片
# output_array = output_tensor.squeeze().cpu().numpy()  # 先将张量移动到 CPU
# slice_index = output_array.shape[1] // 2 + 10  # 显示中间切片
#
# # 显示原图和预测结果
# fig, axes = plt.subplots(1, 2)
#
# # 原图
# axes[0].imshow(input_tensor.cpu().numpy()[0][0][slice_index], cmap='gray')
# axes[0].set_title('Original Image')
#
# # 分割结果
# axes[1].imshow(output_array[0][slice_index], cmap='gray')
# axes[1].set_title('Segmentation Result')
#
# plt.show()
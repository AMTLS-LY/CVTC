import os
import torch
from torch.utils.data import Dataset, DataLoader
import nibabel as nib
import numpy as np
from torch import nn,optim

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
class Attention_block(nn.Module):
    def __init__(self,ch_in):
        super(Attention_block,self).__init__()

        self.ch_in=ch_in
        self.to_qkv=nn.Conv2d(ch_in,ch_in*3,kernel_size=1,stride=1,padding=0)
        self.to_out=nn.Conv2d(ch_in,ch_in,kernel_size=1)

        self.norm=nn.Sequential(
            nn.BatchNorm2d(ch_in)
        )

    def forward(self,x):
        b,ch,h,w=x.size()
        x_norm=self.norm(x)
        x_qkv=self.to_qkv(x_norm)
        q,k,v=torch.split(x_qkv,self.ch_in,dim=1)
        q=q.permute(0,2,3,1).view(b,h*w,ch)
        k=k.view(b,ch,h*w)
        v=v.permute(0,2,3,1).view(b,h*w,ch)

        dot=torch.bmm(q,k)*(ch**-0.5)
        attention=torch.softmax(dot,dim=-1)
        out=torch.bmm(attention,v).view(b,h,w,ch).permute(0,3,1,2)
        return self.to_out(out)+x

test=torch.randn(12,3,32,32)
model=Attention_block(ch_in=3)
print(model(test).size())

class ResBlk(nn.Module):
    def __init__(self,ch_in,ch_out,stride=1):
        super(ResBlk,self).__init__()

        self.ch_in=ch_in

        self.conv=nn.Sequential(
            nn.Conv2d(ch_in,ch_out,kernel_size=3,stride=stride,padding=1),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(),
            nn.Conv2d(ch_out,ch_out,kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(ch_out),
            nn.ReLU()
        )

        self.attention=Attention_block(ch_out)
        self.shortcut=nn.Sequential(
            nn.Conv2d(ch_in,ch_out,kernel_size=3,stride=stride,padding=1),
            nn.BatchNorm2d(ch_out),
            nn.ReLU()
        )

    def forward(self,x):
        out=self.conv(x)
        out=self.attention(out)
        x=out+self.shortcut(x)
        return x

test=torch.randn(12,3,32,32)
model=ResBlk(ch_in=3,ch_out=16)
print(model(test).size())
class up_sample(nn.Module):
    def __init__(self,ch_in):
        super(up_sample,self).__init__()

        self.up_sample=nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(ch_in,ch_in,kernel_size=3,stride=1,padding=1)
        )

    def forward(self,x):
        x=self.up_sample(x)
        return x

class down_sample(nn.Module):
    def __init__(self,ch_in):
        super(down_sample,self).__init__()

        self.down_sample=nn.Sequential(
            nn.Conv2d(ch_in,ch_in,kernel_size=3,stride=2,padding=1),
            nn.BatchNorm2d(ch_in),
            nn.ReLU()
        )

    def forward(self,x):
        x=self.down_sample(x)
        return x

class Unet(nn.Module):
    def __init__(self,ch_in,dim):
        super(Unet,self).__init__()

        self.conv1=nn.Conv2d(ch_in,dim,kernel_size=3,stride=1,padding=1)

        #下采样 每一次加入两个短接层
        down=[]
        for i in range(4):
            for _ in range(2):
                down.append(ResBlk(ch_in=dim,ch_out=dim))
            down.append(down_sample(ch_in=dim))

        self.down_sample=nn.Sequential(*down)

        self.mid_layer=nn.Sequential(
            ResBlk(ch_in=dim,ch_out=dim),
            ResBlk(ch_in=dim,ch_out=dim)
        )

        #上采样 每一次加入两个短接层
        up=[]
        for i in range(4):
            for _ in range(2):
                up.append(ResBlk(ch_in=dim,ch_out=dim))
            up.append(up_sample(ch_in=dim))

        self.up_sample=nn.Sequential(*up)

        self.change_ch = nn.Conv2d(dim * 2, dim, kernel_size=3, stride=1, padding=1)
        self.out_conv = nn.Sequential(
            nn.Conv2d(dim, ch_in, 3, padding=1, stride=1),
            nn.BatchNorm2d(ch_in),
            nn.ReLU()
        )

    def forward(self,x):
        x=self.conv1(x)
        #下采样
        down_output=[]
        for layer in self.down_sample:
            x=layer(x)
            down_output.append(x)

        x=self.mid_layer(x)
        #上采样
        for layer in self.up_sample:
            skip_connection = down_output.pop()
            if skip_connection.size() != x.size():
                skip_connection = torch.nn.functional.interpolate(skip_connection, size=x.size()[2:])
            x = torch.cat((x, skip_connection), dim=1)  # [b,ch,x,x]==>[b,2ch,x,x]
            x = self.change_ch(x)
            # print('s',x.size())
            # print(x.size())
            x = layer(x)

        x=self.out_conv(x)
        return x

from torchvision.transforms import Resize
import torchvision.transforms as transforms

class MRIDataset(Dataset):
    def __init__(self, root_dir, target_size=(208, 208), slice_index=0):
        self.image_slices = []
        self.mask_slices = []
        self.target_size = target_size  # 目标尺寸
        self.resize_image = Resize(self.target_size)  # 初始化图像调整大小的 transform
        self.resize_mask = Resize(self.target_size, interpolation=transforms.InterpolationMode.NEAREST)  # 使用最近邻插值调整掩码
        self.slice_index = slice_index  # 特定切片索引

        for subdir in os.listdir(root_dir):
            subdir_path = os.path.join(root_dir, subdir)
            if os.path.isdir(subdir_path):
                image_path = os.path.join(subdir_path, f'sub-{subdir}_ses-NFB3_T1w.nii.gz')
                mask_path = os.path.join(subdir_path, f'sub-{subdir}_ses-NFB3_T1w_brainmask.nii.gz')
                if os.path.exists(image_path) and os.path.exists(mask_path):
                    image = nib.load(image_path).get_fdata()
                    mask = nib.load(mask_path).get_fdata()

                    # 只加载特定索引的切片
                    if self.slice_index < image.shape[2]:
                        self.image_slices.append(image[:, :, self.slice_index])
                        self.mask_slices.append(mask[:, :, self.slice_index])
                    else:
                        print(f"Slice index {self.slice_index} out of range for {subdir_path}")

                else:
                    print(f"Missing files in {subdir_path}")
                    print(f"Expected image path: {image_path}")
                    print(f"Expected mask path: {mask_path}")

    def __len__(self):
        return len(self.image_slices)

    def __getitem__(self, idx):
        image = self.image_slices[idx]
        mask = self.mask_slices[idx]

        # Normalize the image
        image = (image - np.min(image)) / (np.max(image) - np.min(image))

        # Convert to torch tensors
        image = torch.tensor(image, dtype=torch.float32).unsqueeze(0)
        mask = torch.tensor(mask, dtype=torch.float32).unsqueeze(0)

        # 调整尺寸为 target_size
        image = self.resize_image(image)
        mask = self.resize_mask(mask)  # 使用最近邻插值法调整掩码

        return image, mask

def dice_coefficient(pred, target, smooth=1e-6):
    pred = pred.contiguous()
    target = target.contiguous()

    intersection = (pred * target).sum(dim=2).sum(dim=2)
    dice = (2. * intersection + smooth) / (pred.sum(dim=2).sum(dim=2) + target.sum(dim=2).sum(dim=2) + smooth)

    return dice.mean().item()


def iou(pred, target, smooth=1e-6):
    pred = pred.contiguous()
    target = target.contiguous()

    intersection = (pred * target).sum(dim=2).sum(dim=2)
    union = pred.sum(dim=2).sum(dim=2) + target.sum(dim=2).sum(dim=2) - intersection
    iou = (intersection + smooth) / (union + smooth)

    return iou.mean().item()


def test_segmentation_results(unet, data_loader):
    unet.eval()
    dice_scores = []
    iou_scores = []
    with torch.no_grad():
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = unet(images)
            preds = torch.sigmoid(outputs) > 0.5

            dice = dice_coefficient(preds, labels)
            iou_score = iou(preds, labels)

            dice_scores.append(dice)
            iou_scores.append(iou_score)

    avg_dice = np.mean(dice_scores)
    avg_iou = np.mean(iou_scores)

    print(f'Average Dice Coefficient: {avg_dice:.4f}')
    print(f'Average IoU: {avg_iou:.4f}')

import matplotlib.pyplot as plt
def show_segmentation_results(unet, data_loader, num_images=4):
    unet.eval()
    images_shown = 0

    with torch.no_grad():
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = unet(images)
            # 清理显存
            torch.cuda.empty_cache()
            preds = torch.sigmoid(outputs) > 0.5

            for i in range(images.size(0)):
                if images_shown >= num_images:
                    return

                image = images[i].cpu().numpy().squeeze()
                label = labels[i].cpu().numpy().squeeze()
                pred = preds[i].cpu().numpy().squeeze()

                # 创建一个只显示分割部分的图像
                segmented_image = np.where(pred, image, 0)  # 只保留分割部分的图像像素

                fig, axes = plt.subplots(1, 3, figsize=(15, 5))
                axes[0].imshow(image, cmap='gray')
                axes[0].set_title('Input Image')
                axes[1].imshow(label, cmap='gray')
                axes[1].set_title('Ground Truth')
                axes[2].imshow(segmented_image, cmap='gray')
                axes[2].set_title('Segmented Image')

                for ax in axes:
                    ax.axis('off')

                plt.show()

                images_shown += 1

from PIL import Image
def load_image(image_path, target_size=(208, 208)):
    image = Image.open(image_path).convert('L')  # 加载灰度图像
    transform = transforms.Compose([
        transforms.Resize(target_size),
        transforms.ToTensor()
    ])
    return transform(image).unsqueeze(0)  # 添加批次维度
def calculate_iou(pred, target):
    intersection = np.logical_and(pred, target)
    union = np.logical_or(pred, target)
    iou = np.sum(intersection) / np.sum(union)
    return iou

def calculate_dice(pred, target):
    intersection = np.logical_and(pred, target)
    dice = 2 * np.sum(intersection) / (np.sum(pred) + np.sum(target))
    return dice

def show_segmentation_results_with_custom_images(unet, image_paths, target_size=(208, 208)):
    unet.eval()
    images_shown = 0

    with torch.no_grad():
        for image_path in image_paths:
            image = load_image(image_path, target_size).to(device)
            output = unet(image)
            # 清理显存
            torch.cuda.empty_cache()
            pred = torch.sigmoid(output) > 0.5

            image_np = image.cpu().numpy().squeeze()
            pred_np = pred.cpu().numpy().squeeze()

            # 创建一个只显示分割部分的图像
            segmented_image = np.where(pred_np, image_np, 0)  # 只保留分割部分的图像像素

            # 假设有可用于比较的真实掩码
            # 这里使用一个虚拟的真实掩码进行演示
            ground_truth_mask = np.zeros_like(pred_np)  # 替换为实际的真实掩码

            iou = calculate_iou(pred_np, ground_truth_mask)
            dice = calculate_dice(pred_np, ground_truth_mask)

            fig, axes = plt.subplots(1, 2, figsize=(10, 5))
            axes[0].imshow(image_np, cmap='gray')
            axes[0].set_title('Input Image')
            axes[1].imshow(segmented_image, cmap='gray')
            axes[1].set_title(f'Segmented Image\nIoU: {iou:.4f}, Dice: {dice:.4f}')

            for ax in axes:
                ax.axis('off')

            plt.show()

            images_shown += 1
            if images_shown >= len(image_paths):
                break


# 加载训练好的 U-Net 模型
unet = Unet(ch_in=1, dim=64)
unet.load_state_dict(torch.load(r"E:\dataset\best_Unet_model(2).pth"))
unet.to(device)

# 创建测试数据集的 DataLoader
test_dataset = MRIDataset(root_dir=r"E:\dataset\nfbs",slice_index=60)
print(len(test_dataset))
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

# 测试分割结果
test_segmentation_results(unet, test_loader)
# 显示分割结果的图片
show_segmentation_results(unet, test_loader)
show_segmentation_results_with_custom_images(unet, [r"E:\dataset\帕金森\非帕金森患者-冠状\sagittal_slice_47_4.png"])

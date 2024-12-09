import torch
from torch import nn,optim
from torchvision import datasets, transforms

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

test=torch.randn(12,3,32,32)
model=Unet(ch_in=3,dim=64)
print(model(test).size())



#预训练uUet

def pretrain_Unet(Unet, data_loader, epochs=1, lr=0.001):
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(Unet.parameters(), lr=lr)

    for epoch in range(epochs):
        Unet.train()
        running_loss = 0.0
        for images, labels in data_loader:
            optimizer.zero_grad()
            images, labels = images.to(device), labels.to(device)
            outputs = Unet(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        print(f'Epoch [{epoch + 1}/{epochs}], Loss: {running_loss / len(data_loader):.4f}')

        # 模型评估
        Unet.eval()
        with torch.no_grad():
            val_loss = 0.0
            for images, labels in data_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = Unet(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
            print(f'Validation Loss: {val_loss / len(data_loader):.4f}')

        # 保存模型
        torch.save(Unet.state_dict(), f'Unet_epoch_{epoch + 1}.pth')



import os
import torch
from torch.utils.data import Dataset, DataLoader
import nibabel as nib
import numpy as np
from torchvision.transforms import Resize, RandomHorizontalFlip, RandomVerticalFlip, RandomCrop, ColorJitter, \
    RandomRotation


class MRIDataset(Dataset):
    def __init__(self, root_dir, target_size=(208, 208), crop_size=(180, 180)):
        self.image_slices = []
        self.mask_slices = []
        self.target_size = target_size  # 目标尺寸
        self.crop_size = crop_size  # 随机裁剪尺寸
        self.resize_image = Resize(self.target_size)  # 初始化图像调整大小的 transform
        self.resize_mask = Resize(self.target_size, interpolation=transforms.InterpolationMode.NEAREST)  # 使用最近邻插值调整掩码

        for subdir in os.listdir(root_dir):
            subdir_path = os.path.join(root_dir, subdir)
            if os.path.isdir(subdir_path):
                image_path = os.path.join(subdir_path, f'sub-{subdir}_ses-NFB3_T1w.nii.gz')
                mask_path = os.path.join(subdir_path, f'sub-{subdir}_ses-NFB3_T1w_brainmask.nii.gz')
                if os.path.exists(image_path) and os.path.exists(mask_path):
                    image = nib.load(image_path).get_fdata()
                    mask = nib.load(mask_path).get_fdata()

                    # 轴状切片
                    for i in range(image.shape[2]):
                        self.image_slices.append(image[:, :, i])
                        self.mask_slices.append(mask[:, :, i])

                    # 冠状切片
                    for i in range(image.shape[1]):
                        self.image_slices.append(image[:, i, :])
                        self.mask_slices.append(mask[:, i, :])

                    # 矢状切片
                    for i in range(image.shape[0]):
                        self.image_slices.append(image[i, :, :])
                        self.mask_slices.append(mask[i, :, :])
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

        # 数据增强
        if torch.rand(1).item() > 0.5:
            image = RandomHorizontalFlip(p=1.0)(image)
            mask = RandomHorizontalFlip(p=1.0)(mask)
        if torch.rand(1).item() > 0.5:
            image = RandomVerticalFlip(p=1.0)(image)
            mask = RandomVerticalFlip(p=1.0)(mask)

        # 随机裁剪
        i, j, h, w = RandomCrop.get_params(image, output_size=self.crop_size)
        image = image[:, i:i + h, j:j + w]
        mask = mask[:, i:i + h, j:j + w]

        # 调整尺寸为 target_size
        image = self.resize_image(image)
        mask = self.resize_mask(mask)

        # 更多数据增强
        if torch.rand(1).item() > 0.5:
            color_jitter = ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2)
            image = color_jitter(image)

        if torch.rand(1).item() > 0.5:
            rotation_angle = torch.randint(-30, 30, (1,)).item()
            image = RandomRotation((rotation_angle, rotation_angle))(image)
            mask = RandomRotation((rotation_angle, rotation_angle))(mask)

        return image, mask


# 示例数据路径
root_dir = r"E:\dataset\NFBS_Dataset"
dataset = MRIDataset(root_dir)
print(f"Found {len(dataset)} samples in the dataset.")
dataloader = DataLoader(dataset, batch_size=1, shuffle=True)



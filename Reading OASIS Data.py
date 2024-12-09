import os
import pandas as pd
import nibabel as nib
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import numpy as np
from torchvision import transforms

# 定义输入和输出文件夹
input_folder = r"E:\dataset\subjects"
output_folder = r'E:\dataset\分割后的图像'

# 确保输出文件夹存在
os.makedirs(output_folder, exist_ok=True)

# 读取Excel文件，假设Excel文件名为 "file_info.xlsx"
excel_file = r"E:\dataset\csv\final_oasis_FahimIsrat.csv"
df = pd.read_csv(excel_file)

# 创建四分类文件夹
categories = {
    0: 'NonDemented',
    0.5: 'VeryMildDemented',
    1: 'MildDemented',
    2: 'ModerateDemented'
}

for category in categories.values():
    os.makedirs(os.path.join(output_folder, category), exist_ok=True)

# 加载Unet
class UNet(nn.Module):
    pass

unet_model = UNet()
unet_model.load_state_dict(torch.load(r"E:\dataset\best_Unet_model(2).pth"))
unet_model.eval()

# 定义一个函数来进行分割
def segment_brain(image, model):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((256, 256))
    ])
    image_tensor = transform(image).unsqueeze(0)  # 增加批次维度
    with torch.no_grad():
        mask_tensor = model(image_tensor)
    mask = mask_tensor.squeeze().numpy()  # 去除批次维度并转换为numpy数组
    mask = np.resize(mask, image.shape)  # 调整回原始尺寸
    return mask

# 遍历Excel中的每一行，根据文件名和CDR评分进行处理和分类存储
for index, row in df.iterrows():
    file_name = row.iloc[0]  # 使用 iloc 访问
    cdr_score = row.iloc[2]  # 使用 iloc 访问

    # 确保文件名包含 .nii 扩展名
    if not file_name.endswith('.nii'):
        file_name += '.nii'

    input_path = os.path.join(input_folder, file_name)
    brain_path = os.path.join(output_folder, categories[cdr_score], file_name.replace('.nii', '_brain.nii'))
    mask_path = os.path.join(output_folder, categories[cdr_score], file_name.replace('.nii', '_mask.nii'))

    # 检查输入文件是否存在
    if not os.path.exists(input_path):
        print(f"文件未找到: {input_path}")
        continue

    # 加载输入图像
    img = nib.load(input_path)
    img_data = img.get_fdata()

    # 使用Unet模型进行分割
    mask_data = segment_brain(img_data, unet_model)

    # 保存分割后的脑部图像和掩码
    brain_img = nib.Nifti1Image(img_data, img.affine)
    mask_img = nib.Nifti1Image(mask_data, img.affine)
    nib.save(brain_img, brain_path)
    nib.save(mask_img, mask_path)

    # 检查文件是否生成
    if not os.path.exists(brain_path):
        print(f"文件未生成: {brain_path}")
        continue

    # 加载并保存分割后的脑部图像到对应的分类文件夹
    brain_data = brain_img.get_fdata()

    num_slices = brain_data.shape[2]
    for i in range(num_slices):
        plt.figure(figsize=(6, 6))  # 调整图像尺寸以避免过大错误
        plt.imshow(brain_data[:, :, i], cmap='gray')
        plt.axis('off')  # 去除坐标轴

        # 保存图像到对应的分类文件夹，去除白色边框
        output_image_path = os.path.join(output_folder, categories[cdr_score], f'{file_name}_slice_{i + 1}.png')
        plt.savefig(output_image_path, bbox_inches='tight', pad_inches=0)
        plt.close()

print("所有切片已成功保存到对应的分类文件夹。")
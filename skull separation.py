import torch
import cc3d
from LinkNet3d import LinkNet3d as model
from bbox import LinkNet3d as box
import nibabel as nib
import numpy as np
import pandas as pd
from tqdm import tqdm
import fill_voids
import torch.nn.functional as F
# 加载 .pth 文件中的权重
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model().to(device)
box=box().to(device)
model_state_dict = torch.load('model.pth')
box_state_dict=torch.load('bbox.pth')
# 将权重加载到模型中
model.load_state_dict(model_state_dict)
box.load_state_dict(box_state_dict)

def normalize(x, low, high):
    x = (x - low) / (high - low)
    x = x.clamp(min=0, max=1)
    x = (x - x.mean()) / x.std()
    return 0.226 * x + 0.449

def dilate(x, n_layer):
    for _ in range(abs(int(n_layer))):
        graph = cc3d.voxel_connectivity_graph(x.astype(int) + 1, connectivity=6)
        x[graph != 63] = 1 if n_layer > 0 else 0
    return x    # 膨胀或腐蚀

def keep_largest_connected_component(mask):
    labels = cc3d.connected_components(mask)
    largest_label = np.bincount(labels[labels != 0]).argmax()
    mask[labels != largest_label] = 0.
    return mask
def apply_mask(x, mask):
    x = x.flatten('F')
    x[~mask.flatten('F')] = 0.
    return x.reshape(mask.shape, order='F')

def reoriented_nifti(array, affine, header):
    ornt_ras = [[0, 1], [1, 1], [2, 1]]
    ornt = nib.io_orientation(affine)
    ornt_inv = nib.orientations.ornt_transform(ornt_ras, ornt)
    out_header = header if isinstance(header, nib.Nifti1Header) else None
    return nib.Nifti1Image(nib.apply_orientation(array, ornt_inv), affine, out_header)


class BrainExtraction:
    def __init__(self):
        self.model = model
        self.bbox_model = box
        self.bbox = None

    def run(self, input, brain_path=None, mask_path=None, tiv_path=None, threshold=.5, n_dilate=0):
        img = nib.load(input) if isinstance(input, str) else input
        x = nib.as_closest_canonical(img).get_fdata(dtype=np.float32)
        x = x[..., 0] if len(img.shape) == 4 else x
        mask = self.run_model(x.copy())
        mask = mask > threshold
        mask = self.postprocess(mask, n_dilate)
        x = apply_mask(x, mask)
        x = x[..., None] if len(img.shape) == 4 else x
        mask = mask[..., None] if len(img.shape) == 4 else mask
        tiv = 1e-3 * mask.sum() * np.prod(img.header.get_zooms()[:3])
        img = reoriented_nifti(x, img.affine, img.header)
        mask = reoriented_nifti(mask, img.affine, img.header)
        mask.header.set_data_dtype(np.uint8)
        self.save(img, mask, tiv, brain_path, mask_path, tiv_path)
        return img, mask, tiv

    def run_model(self, x, small_shape=(128, 128, 128), shape=(256, 256, 256), bbox_margin=.1):
        x = torch.from_numpy(x).to(next(self.model.parameters()).device)
        x = torch.nan_to_num(x)
        mask = torch.zeros_like(x)
        x_small = F.interpolate(x[None, None], small_shape, mode='nearest-exact')[0, 0]
        low, high = x_small.quantile(.005), x_small.quantile(.995)
        with torch.no_grad():
            mask_small = self.bbox_model(normalize(x_small, low, high)[None, None])[0, 1]
        mask_small = keep_largest_connected_component((mask_small > .5).float().cpu().numpy())
        mask_small = torch.from_numpy(mask_small).to(next(self.model.parameters()).device)
        self.bbox = self.get_bbox_with_margin(mask_small, mask.shape, bbox_margin)
        x = F.interpolate(x[self.bbox][None, None], shape, mode='nearest-exact')[0, 0]
        with torch.no_grad():
            mask_bbox = self.model(normalize(x, low, high)[None, None])[0, 1]
        mask[self.bbox] = F.interpolate(mask_bbox[None, None], mask[self.bbox].shape, mode='nearest-exact')[0, 0]
        return mask.cpu().numpy()

    def postprocess(self, mask, n_dilate=0):
        mask[self.bbox] = keep_largest_connected_component(mask[self.bbox])
        mask[self.bbox] = fill_voids.fill(mask[self.bbox])
        return dilate(mask, n_dilate)

    def get_bbox_with_margin(self, mask_small, shape, margin):
        margin = margin * torch.ones(3)
        scale_factor = torch.tensor(shape) / torch.tensor(mask_small.shape)
        center, size = self.get_bbox(mask_small)
        center, size = scale_factor * center, scale_factor * size
        size = (1 + 2 * margin) * size
        center, size = center.round(), size.round()
        bbox = [[int(c - s / 2), int(c + s / 2)] for c, s in zip(center, size)]
        return tuple([slice(max(0, b[0]), min(s, b[1]), 1) for b, s in zip(bbox, shape)])

    @staticmethod
    def get_bbox(x, threshold=.02):
        rs = [torch.where(x.mean(dims) > threshold)[0] for dims in [(1, 2), (0, 2), (0, 1)]]
        assert all([r.numel() > 0 for r in rs]), 'Not enough foreground to calculate bounding box'
        center = [(r.max() + 1 + r.min()) / 2 for r in rs]
        size = [(r.max() + 1) - r.min() for r in rs]
        return torch.tensor(center), torch.tensor(size)

    @staticmethod
    def save(img, mask, tiv, img_path, mask_path, tiv_path):
        if img_path is not None:
            img.to_filename(img_path)
        if mask_path is not None:
            mask.to_filename(mask_path)
        if tiv_path is not None:
            pd.Series([tiv], name='tiv_cm3').to_csv(tiv_path)


def calculate_dice(pred_mask, true_mask):
    pred_mask = pred_mask.flatten()
    true_mask = true_mask.flatten()
    intersection = np.sum(pred_mask * true_mask)
    return (2. * intersection) / (np.sum(pred_mask) + np.sum(true_mask))

def calculate_iou(pred_mask, true_mask):
    pred_mask = pred_mask.flatten()
    true_mask = true_mask.flatten()
    intersection = np.sum(pred_mask * true_mask)
    union = np.sum(pred_mask) + np.sum(true_mask) - intersection
    return intersection / union



# 实例化 BrainExtraction 类
brain_extractor = BrainExtraction()
import os

# 创建目录
output_dir = r'E:\\dataset\\3D Hippo'
os.makedirs(output_dir, exist_ok=True)

# 调用 run 方法
input_image_path = r"E:\dataset\I102040.nii" # 替换为你的输入图像路径
# ground_truth_mask_path = r"E:\dataset\nfbs\A00028185\sub-A00028185_ses-NFB3_T1w_brainmask.nii\sub-A00028185_ses-NFB3_T1w_brainmask.nii"
# 更新保存路径
brain_path = os.path.join(output_dir, 'brain_image.nii')
mask_path = os.path.join(output_dir, 'mask_image.nii')
tiv_path = os.path.join(output_dir, 'tiv.csv')

img, mask, tiv = brain_extractor.run(input_image_path, brain_path, mask_path, tiv_path)

# # 加载 ground truth mask
# ground_truth_mask = nib.load(ground_truth_mask_path).get_fdata(dtype=np.float32)
# 提取数据数组
mask_data = mask.get_fdata(dtype=np.float32)
# ground_truth_mask_data = ground_truth_mask
print("大脑提取完成。")

import matplotlib.pyplot as plt

def visualize_slices(image, mask):
    # 获取图像的切片数量
    num_slices = image.shape[2]

    for i in range(num_slices):
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))

        # 显示大脑切片
        axes[0].imshow(image[:, :, i], cmap='gray')
        axes[0].set_title(f'Brain Slice {i + 1}')

        # 显示对应的掩码切片
        axes[1].imshow(mask[:, :, i], cmap='gray')
        axes[1].set_title(f'Mask Slice {i + 1}')

        plt.show()

# 加载大脑和掩码图像
brain_img = nib.load(r"E:\dataset\3D Hippo\brain_image.nii")
mask_img = nib.load(r"E:\dataset\3D Hippo\mask_image.nii")

# 获取图像数据
brain_data = brain_img.get_fdata()
mask_data = mask_img.get_fdata()

# 可视化切片
visualize_slices(brain_data, mask_data)
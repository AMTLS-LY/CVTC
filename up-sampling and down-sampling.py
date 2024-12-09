import os
import random
from PIL import Image

def resample_images(input_folder, target_count):
    # 获取文件夹中的所有图片文件
    image_files = [f for f in os.listdir(input_folder) if f.lower().endswith(('png', 'jpg', 'jpeg', 'bmp', 'gif'))]
    current_count = len(image_files)

    if current_count == 0:
        print("文件夹中没有找到图片。")
        return

    # 如果当前图片数量等于目标数量，不做任何操作
    if current_count == target_count:
        print("图片数量已经等于目标数量。")
        return

    # 如果当前图片数量大于目标数量，进行下采样
    if current_count > target_count:
        images_to_remove = random.sample(image_files, current_count - target_count)
        for img in images_to_remove:
            os.remove(os.path.join(input_folder, img))
        print(f"已下采样至 {target_count} 张图片。")

    # 如果当前图片数量小于目标数量，进行上采样
    else:
        images_to_add = random.choices(image_files, k=target_count - current_count)
        for idx, img in enumerate(images_to_add, start=1):
            original_path = os.path.join(input_folder, img)
            base_name, ext = os.path.splitext(img)
            new_name = f"{base_name}({image_files.count(img) + idx}){ext}"
            new_path = os.path.join(input_folder, new_name)
            image = Image.open(original_path)
            image.save(new_path)
        print(f"已上采样至 {target_count} 张图片。")
# 示例用法
input_folder = r"D:\6666\CVT\allin_file\MildDemented"  # 替换为你的图片文件夹路径
target_count = 20000  # 指定目标图片数量
resample_images(input_folder, target_count)
import pandas as pd
import numpy as np
import os
from math import sqrt
from tkinter import Tk, filedialog
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt


# 检查两个坐标是否在指定距离内的函数
def is_within_distance(coord1, coord2, distance=5):
    return sqrt((coord1[0] - coord2[0]) ** 2 + (coord1[1] - coord2[1]) ** 2) < distance


# 过滤在指定范围内的坐标的函数
def filter_coordinates(coords, x_range=(25, 226), y_range=(18, 233)):
    return [coord for coord in coords if x_range[0] <= coord[0] <= x_range[1] and y_range[0] <= coord[1] <= y_range[1]]


# 查找多个Excel文件中相等坐标的函数
def find_equal_coordinates(files):
    all_coords = []

    # 从每个文件中读取坐标
    for file in files:
        df = pd.read_excel(file)
        coords = df.iloc[:, 0].apply(lambda x: tuple(map(int, x.strip('()').split(',')))).tolist()
        filtered_coords = filter_coordinates(coords)
        all_coords.append(filtered_coords)

    # 查找在所有文件中都出现或满足距离条件的坐标
    equal_coords = set(all_coords[0])
    for coords in all_coords[1:]:
        current_set = set()
        for coord1 in equal_coords:
            for coord2 in coords:
                if is_within_distance(coord1, coord2):
                    current_set.add(coord1)
                    break
        equal_coords = current_set

    return list(equal_coords)


# 使用tkinter选择多个Excel文件
def select_files(filetypes):
    root = Tk()
    root.withdraw()  # 隐藏主窗口
    file_paths = filedialog.askopenfilenames(filetypes=filetypes)
    return list(file_paths)


from torchvision.transforms import transforms
# 在图片上标记坐标的函数
def mark_coordinates_on_image(image_path, coordinates):
    # 定义预处理步骤
    preprocess = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])

    # 打开图像并进行预处理
    with Image.open(image_path) as img:
        img = img.convert("RGB")  # 确保图片是RGB模式
        input_tensor = preprocess(img)
        img_resized = transforms.ToPILImage()(input_tensor)

        # 在调整大小后的图像上绘制点
        draw = ImageDraw.Draw(img_resized)
        for coord in coordinates:
            draw.point((coord[0], coord[1]), fill='red')

        # 将图像转换为可以用 plt.show() 显示的格式
        img_array = np.array(img_resized)

        plt.imshow(img_array)
        plt.axis('off')  # 隐藏坐标轴
        plt.show()


# 将相等的坐标保存为新的Excel文件的函数，用一列保存坐标，用元组形式
def save_coordinates_to_excel(coordinates):
    root = Tk()
    root.withdraw()  # 隐藏主窗口
    output_file = filedialog.asksaveasfilename(defaultextension=".xlsx", filetypes=[("Excel files", "*.xlsx")])

    if output_file:
        df = pd.DataFrame({'Coordinates': [str(coord) for coord in coordinates]})
        df.to_excel(output_file, index=False)
        print(f"相等的坐标已保存到 {output_file} 文件中")
    else:
        print("未选择存储路径")


# 主程序
if __name__ == "__main__":
    excel_files = select_files([("Excel files", "*.xlsx")])
    if excel_files:
        equal_coordinates = find_equal_coordinates(excel_files)
        print("在多个Excel文件中找到的相等坐标:", equal_coordinates)

        # 保存相等的坐标到新的Excel文件，用户自定义存储路径，用一列保存坐标，用元组形式
        save_coordinates_to_excel(equal_coordinates)

        image_files = select_files([("Image files", "*.png;*.jpg;*.jpeg")])
        if image_files:
            for image_file in image_files:
                mark_coordinates_on_image(image_file, equal_coordinates)
        else:
            print("未选择任何图片文件")
    else:
        print("未选择任何Excel文件")

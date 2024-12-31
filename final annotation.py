import pandas as pd
from PIL import Image
import torchvision.transforms as transforms
import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox

def mark_image(exc_path, image_path):
    try:
        # 读取Excel文件和图像
        df = pd.read_excel(exc_path)
        image = Image.open(image_path).convert('RGB')

        # 预处理图像
        preprocess = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor()
        ])
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
                heatmap[y, x] = 255  # 将点的值设为最大值

        # 应用高斯模糊使热图平滑
        heatmap = cv2.GaussianBlur(heatmap, (15, 15), 0)

        # 将热图转换为彩色
        heatmap_color = cv2.applyColorMap(heatmap.astype(np.uint8), cv2.COLORMAP_JET)

        # 创建一个透明图层
        transparent_layer = np.zeros_like(image_cv, dtype=np.uint8)
        transparent_layer[heatmap > 0] = heatmap_color[heatmap > 0]

        # 将热图叠加到原始图像上
        overlay = cv2.addWeighted(image_cv, 1.0, transparent_layer, 0.6, 0)

        # 转换为灰度图像
        gray = cv2.cvtColor(image_cv, cv2.COLOR_BGR2GRAY)

        # 应用阈值以创建二值图像
        _, binary = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)

        # 查找轮廓
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # 创建一个全黑的掩码
        mask = np.zeros_like(gray)

        # 绘制大脑区域的轮廓
        cv2.drawContours(mask, contours, -1, 255, thickness=cv2.FILLED)

        # 调整掩码的尺寸以匹配标注后的图像
        mask_resized = cv2.resize(mask, (overlay.shape[1], overlay.shape[0]))

        # 使用掩码将大脑区域复制到标注后的图像上
        brain_region = cv2.bitwise_and(overlay, overlay, mask=mask_resized)

        # 将大脑区域外的部分变为黑色
        final_image = np.where(mask_resized[:, :, np.newaxis] == 0, 0, brain_region)

        # 保存标注后的图像
        save_path = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG Files", "*.png")])
        if save_path:
            cv2.imwrite(save_path, final_image)
            messagebox.showinfo("完成", f"标注后的图片已保存为：{save_path}")

        # 显示结果
        cv2.imshow('Result', final_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    except Exception as e:
        messagebox.showerror("错误", f"发生错误: {e}")

def select_excel():
    exc_path = filedialog.askopenfilename(title="选择Excel文件", filetypes=[("Excel Files", "*.xlsx")])
    if exc_path:
        excel_path_var.set(exc_path)

def select_image():
    image_path = filedialog.askopenfilename(title="选择图像文件", filetypes=[("Image Files", "*.png;*.jpg;*.jpeg")])
    if image_path:
        image_path_var.set(image_path)

def process_files():
    exc_path = excel_path_var.get()
    image_path = image_path_var.get()
    if exc_path and image_path:
        mark_image(exc_path, image_path)
    else:
        messagebox.showwarning("警告", "请先选择Excel文件和图像文件")

# 创建主窗口
root = tk.Tk()
root.title("图像标注和处理")

# 创建变量来存储文件路径
excel_path_var = tk.StringVar()
image_path_var = tk.StringVar()

# 创建按钮和标签
select_excel_button = tk.Button(root, text="选择Excel文件", command=select_excel)
select_excel_button.pack(pady=5)

excel_path_label = tk.Label(root, textvariable=excel_path_var)
excel_path_label.pack(pady=5)

select_image_button = tk.Button(root, text="选择图像文件", command=select_image)
select_image_button.pack(pady=5)

image_path_label = tk.Label(root, textvariable=image_path_var)
image_path_label.pack(pady=5)

process_button = tk.Button(root, text="处理文件", command=process_files)
process_button.pack(pady=20)

# 运行主循环
root.mainloop()
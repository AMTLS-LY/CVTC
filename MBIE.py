import cv2
import numpy as np
from PIL import Image, ImageEnhance
import os

def calculate_contrast(channel):
    return np.std(channel)

def apply_clahe_channel(image, channel_index):
    channels = list(cv2.split(image))
    clahe = cv2.createCLAHE(clipLimit=0.5, tileGridSize=(8, 8))
    channels[channel_index] = clahe.apply(channels[channel_index])
    return cv2.merge(channels)

def apply_unsharp_mask_channel(image, channel_index):
    channels = list(cv2.split(image))
    blurred = cv2.GaussianBlur(channels[channel_index], (9, 9), 5.0)
    channels[channel_index] = cv2.addWeighted(channels[channel_index], 1.1, blurred, -0.1, 0)
    return cv2.merge(channels)

def apply_subtle_sharpening_channel(image, channel_index):
    channels = list(cv2.split(image))
    pil_image = Image.fromarray(channels[channel_index])
    enhancer = ImageEnhance.Sharpness(pil_image)
    enhanced_image = enhancer.enhance(1.05)
    channels[channel_index] = np.array(enhanced_image)
    return cv2.merge(channels)

def apply_high_pass_filter_channel(image, channel_index):
    channels = list(cv2.split(image))
    gray = channels[channel_index]
    dft = cv2.dft(np.float32(gray), flags=cv2.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(dft)
    rows, cols = gray.shape
    crow, ccol = rows // 2, cols // 2
    mask = np.ones((rows, cols, 2), np.uint8)
    r = 30  # 将半径从 50 调整为 30
    center = [crow, ccol]
    x, y = np.ogrid[:rows, :cols]
    mask_area = (x - center[0]) ** 2 + (y - center[1]) ** 2 <= r * r
    mask[mask_area] = 0
    fshift = dft_shift * mask
    f_ishift = np.fft.ifftshift(fshift)
    img_back = cv2.idft(f_ishift)
    img_back = cv2.magnitude(img_back[:, :, 0], img_back[:, :, 1])
    img_back = cv2.normalize(img_back, None, 0, 255, cv2.NORM_MINMAX)
    channels[channel_index] = np.uint8(img_back)
    return cv2.merge(channels)

def apply_non_local_means_denoising_channel(image, channel_index):
    channels = list(cv2.split(image))
    channels[channel_index] = cv2.fastNlMeansDenoising(channels[channel_index], None, 3, 7, 21)
    return cv2.merge(channels)

def apply_histogram_equalization_channel(image, channel_index):
    channels = list(cv2.split(image))
    channels[channel_index] = cv2.equalizeHist(channels[channel_index])
    return cv2.merge(channels)

def apply_gamma_correction_channel(image, channel_index, gamma=1.05):
    channels = list(cv2.split(image))
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    channels[channel_index] = cv2.LUT(channels[channel_index], table)
    return cv2.merge(channels)

def resize_to_match(image, reference_image):
    return cv2.resize(image, (reference_image.shape[1], reference_image.shape[0]))

def create_pseudo_rgb(image_path):
    image = cv2.imread(image_path)
    channels = cv2.split(image)

    # Calculate contrast for each channel
    contrasts = [calculate_contrast(channel) for channel in channels]
    max_contrast_index = np.argmax(contrasts)

    # Apply enhancements based on the channel with the highest contrast
    if max_contrast_index == 0:
        pseudo_rgb_image = cv2.merge((apply_unsharp_mask_channel(image, 0)[:, :, 0],
                                      apply_clahe_channel(image, 1)[:, :, 1],
                                      apply_high_pass_filter_channel(image, 2)[:, :, 2]))
    elif max_contrast_index == 1:
        pseudo_rgb_image = cv2.merge((apply_non_local_means_denoising_channel(image, 0)[:, :, 0],
                                      apply_unsharp_mask_channel(image, 1)[:, :, 1],
                                      apply_histogram_equalization_channel(image, 2)[:, :, 2]))
    else:
        pseudo_rgb_image = cv2.merge((apply_gamma_correction_channel(image, 0)[:, :, 0],
                                      apply_unsharp_mask_channel(image, 1)[:, :, 1],
                                      apply_subtle_sharpening_channel(image, 2)[:, :, 2]))

    # Resize back to original size if needed
    pseudo_rgb_image = resize_to_match(pseudo_rgb_image, image)

    return pseudo_rgb_image
def process_images_in_folder(input_folder_path, output_folder_path):
    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder_path):
        os.makedirs(output_folder_path)

    # Iterate through each subfolder in the input folder
    for subfolder_name in os.listdir(input_folder_path):
        subfolder_path = os.path.join(input_folder_path, subfolder_name)

        # Check if it's a directory
        if os.path.isdir(subfolder_path):
            output_subfolder_path = os.path.join(output_folder_path, subfolder_name)

            # Create the output subfolder if it doesn't exist
            if not os.path.exists(output_subfolder_path):
                os.makedirs(output_subfolder_path)

            # Iterate through each image in the subfolder
            for image_name in os.listdir(subfolder_path):
                image_path = os.path.join(subfolder_path, image_name)

                # Check if it's a file and has an image extension
                if os.path.isfile(image_path) and image_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                    try:
                        # Process the image and save the result to the output subfolder
                        pseudo_rgb_image = create_pseudo_rgb(image_path)
                        output_image_path = os.path.join(output_subfolder_path, image_name)
                        cv2.imwrite(output_image_path, pseudo_rgb_image)
                        print(f"Processed and saved: {output_image_path}")
                    except Exception as e:
                        print(f"Failed to process {image_name}: {e}")
#
# # Example usage
# input_folder_path = r"D:\6666\CVT\OriginalDataset"
# output_folder_path = r"EnhancedDataset"
# process_images_in_folder(input_folder_path, output_folder_path)


# Initializing
import os
import glob
import pickle
import time
import re

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import cv2

# Scikit image
from skimage import color, segmentation, filters, measure, morphology
from skimage.io import imread
from skimage.segmentation import slic
from skimage.measure import regionprops, find_contours
from skimage.filters import threshold_multiotsu


from scipy import ndimage as ndi
from scipy.ndimage import center_of_mass, gaussian_filter, generic_filter,binary_closing
from scipy.stats import mode

# Cellpose
from cellpose import core, utils, io, models, metrics, plot
from cellpose.io import logger_setup

# Image loading modules
from aicsimageio import AICSImage
import stackview

################################# LOADING IMAGES #######################################
# CZI image loading
def load_images(image_path):
    try:
        aics_img = AICSImage(image_path)

        scene_arrays = []
        scene_names = []

        for scene in aics_img.scenes:
            scene_names.append(scene)

            # Get scene
            aics_img.set_scene(scene)

            # Get the data for this scene (as numpy array)
            data = aics_img.get_image_data("XYC", T=0)  # This will return data for the currently active scene

            # Append the numpy array to the list
            scene_arrays.append(data)

        nuc_image_1 = scene_arrays[0].squeeze()
        nuc_image_2 = scene_arrays[1].squeeze()

        return nuc_image_1, nuc_image_2, scene_names

    except IOError as err:
        print ("Image path does not exist", str(err))

################################# DATA PREPROCESSING #######################################
###     GENERAL USE FUNCTIONS

# Downscaling
def downscale_image(image, scale_percent):
    # Calculate new dimensions
    original_shape = image.shape[:2]  # (height, width)
    width = int(original_shape[1] * scale_percent / 100)
    height = int(original_shape[0] * scale_percent / 100)
    dim = (width, height)

    # Resize the image
    resized = cv2.resize(image, dim, interpolation=cv2.INTER_LINEAR)
    return resized

def quantile_normalize(image, alpha=0.001):
    img_min, img_max = np.quantile(image.flatten(), [alpha / 2, 1 - alpha / 2])
    img = (image - img_min) / (img_max - img_min)  # Use 'image' instead of 'img'
    img[img > 1] = 1
    img[img < 0] = 0
    return img


def preprocess_image(image, max_intensity=2500, alpha = 0.01): # single channel
    # Clip values above 2500 (max intensity) to 2500
    clip_image = np.clip(image, None, max_intensity)
    mean_val = np.mean(clip_image)
    mean_image = (clip_image > mean_val).astype(np.float32)
    qn_img = quantile_normalize(mean_image, alpha = alpha)
    return qn_img

#mean_image = np.where(DAPI_channel > np.mean(DAPI_channel), DAPI_channel, 0)

##############################################################################################################
                #DATA OVERVIEW: DOWNSCALING, PLOTTING, HISTOGRAMS ETC - NOT IN REPORT
##############################################################################################################
###     DOWNSCALE ALL IMAGES

def downscale_all_images(base_path, scale_percent = 10, outfile = "downscaled_images"):
    """
    Downscales all images in path and saves in a pickle file. Be aware that it is specialized to certain naming conditions.
    base_path = r"/work/imaging_data_f2805_HEV/2023_07_14_F2805_w.*"
    """

    # Use glob to find all matching files
    image_paths = glob.glob(base_path)

    # Initialize a dictionary to store results
    results_dict = {}

    # Loop through each image path
    for path in image_paths:
        print(f"Processing image: {path}")

        # Load and preprocess image
        scenes = load_images(path)
        filename = os.path.splitext(os.path.basename(path))[0]   #removes .czi
        
        for scene in range(len(scenes)-1):
            image_name = f"{filename}_Scene_{scene}"
            image = scenes[scene]
            original_shape = image.shape[:2]

            downscaled_image = downscale_image(image, scale_percent=scale_percent)   

            # Store the result in the dictionary
            results_dict[outfile] = {
                'image_name': image_name,
                'downscaled_image': downscaled_image,
                'original_shape' : original_shape   # So it can be upscaled again
            }

    #Save results dictionary to a pickle file
    with open('outfile'+'.pkl', 'wb') as f:
       pickle.dump(results_dict, f, protocol=4)


### DATA OVERVIEW
def average_image_size(results_dict):
    image_sizes = []
    for value in results_dict.values():
        height = value['original_shape'][0]
        width = value['original_shape'][1]
        total_size = height * width
        image_sizes.append(total_size)

    # Compute the average size
    average_size = np.mean(image_sizes)

    # Print summary
    print(f"\nAverage size: {average_size:.2f} pixels")

    return average_size

### Show all images and channels

# Sorting the images
def parse_image_name(image_name):
    """
    Extract week, type, and scene information from an image name.
    """
    pattern = r'(\d{4}_\d{2}_\d{2}_F2805)_w\.(\d+)\s([A-Za-z0-9\+\-]+)_iBALT_Scene_(\d)'
    match = re.search(pattern, image_name)
    if match:
        date, week, type_, scene = match.groups()
        return {
            'date': date,
            'week': int(week),
            'type': type_,
            'scene': int(scene),
        }
    return None

def type_sort_order(type_):
    """
    Custom sorting order for types to ensure the desired order:
    A1, A2, A3, A4, B1, B2, B3, B4.
    """
    order = {
        "A1+A2": 0, "A3+A4": 1, "B1+B2": 2, "B3+B4": 3,
    }
    return order.get(type_, float('inf'))

def generate_sorted_image_list(results_dict):
    """
    Generate a sorted list of image names based on week, type, and scene.
    """
    parsed_images = []
    for image_name in results_dict.keys():
        parsed = parse_image_name(image_name)
        if parsed:
            parsed['name'] = image_name
            parsed_images.append(parsed)

    # Sort the parsed image data
    parsed_images.sort(key=lambda x: (x['week'], type_sort_order(x['type']), x['scene']))

    # Return the sorted list of image names
    return [img['name'] for img in parsed_images]

# Thresholding 
def binary_thresholding(image, lower_bound=1000):
    binary_image = np.zeros_like(image, dtype=np.float32)
    binary_image[image >= lower_bound] = 1.0
    return binary_image

# Create 3 channels as RGB and 1 greyscale
def create_combined_rgb_image(image_data):
    downscaled_image = image_data['downscaled_image']
    binary_image = binary_thresholding(
        downscaled_image, 
        lower_bound=1000
    )
    T = binary_image[..., 0]
    B = binary_image[..., 1]
    HEV = binary_image[..., 2]
    DAPI = binary_image[..., 3]

    combined_rgb = np.zeros((downscaled_image.shape[0], downscaled_image.shape[1], 3), dtype=np.float32)
    combined_rgb[..., 0] = B
    combined_rgb[..., 1] = T
    combined_rgb[..., 2] = HEV
    return combined_rgb, DAPI

def plot_all_downscaled(results_dict):
    # Generate the sorted list of image names
    sorted_image_names = generate_sorted_image_list(results_dict)

    #sorted_image_names = sorted_image_names[10:15]

    # Load the pickle file
    with open('downscaled_images.pkl', 'rb') as f:
        results_dict = pickle.load(f)

    # Plotting
    num_images = len(sorted_image_names)  
    fig, axes = plt.subplots(nrows=num_images, ncols=5, figsize=(20, 5 * num_images))


    for row_idx, image_name in enumerate(sorted_image_names):
        if image_name not in results_dict:
            continue

        image_data = results_dict[image_name]
        combined_rgb, dapi_binary = create_combined_rgb_image(image_data)

        # Add subtitle above the row
        axes[row_idx, 0].annotate(
            image_name, 
            xy=(0, 0), xytext=(3.0, 1.2),  # Position above the row
            textcoords='axes fraction',
            ha='center',
            fontsize=12,
            fontweight='bold'
        )
        
        # Plot combined RGB and DAPI image
        ax = axes[row_idx, 0]

        # Create a mask for combined_rgb where any RGB channel is non-zero
        mask_rgb_nonzero = np.any(combined_rgb != 0, axis=-1)

        # Create a combined image where the RGB is shown where it is non-zero, otherwise DAPI is shown
        combined_visible = np.zeros_like(combined_rgb)
        combined_visible[mask_rgb_nonzero] = combined_rgb[mask_rgb_nonzero]
        combined_visible[~mask_rgb_nonzero] = np.repeat(dapi_binary[..., np.newaxis], 3, axis=-1)[~mask_rgb_nonzero]

        # Display the combined image where RGB shows where non-zero and DAPI elsewhere
        ax.imshow(combined_visible)

        # Set the title and remove axis
        ax.set_title('Combined (RGB on DAPI)', fontsize=10)
        ax.axis('off')


        # Plot individual binary channels
        channels = ['T', 'B', 'HEV', 'DAPI']
        
        for k, channel in enumerate(channels):
            ax = axes[row_idx, k + 1]
            ax.imshow(combined_rgb[..., k] if k < 3 else dapi_binary, cmap='gray')
            ax.set_title(channel, fontsize=10)
            ax.axis('off')
    plt.show()

############################# HISTOGRAMs ##########################
def plot_combined_hist_otsu_multiple_images(image_names, downscaled_images, channel_names = ["T cell", "B cell", "HEV", "DAPI"]):
    """
    Plot the combined average histogram for multiple images across all channels (T cell, B cell, HEV, DAPI).
    Also plots the Otsu threshold, mean, and median lines based on the combined histogram.
    
    Args:
        image_names (list of str): List of image names for display.
        downscaled_images (list of np.ndarray): List of downscaled images for each image (each of shape (H, W, 4)).
    """
    num_channels = len(channel_names)
    num_images = len(downscaled_images)

    # Create a figure with histograms for each channel
    fig, axs = plt.subplots(1, num_channels, figsize=(20, 5))  # One row, for each channel

    for i in range(num_channels):
        # List to collect all pixel intensities for the current channel across all images
        all_pixel_values = []

        # Calculate the histogram for each image in the channel
        for img_idx in range(num_images):
            downscaled_image = downscaled_images[img_idx]
            image_name = image_names[img_idx]

            # Get the data for the current channel and flatten it
            channel_data = downscaled_image[:, :, i].ravel()

            # Clip values above 2500 (max intensity) to 2500
            channel_data_clipped = np.clip(channel_data, None, 2500)

            # Collect the clipped pixel values for Otsu calculation
            all_pixel_values.extend(channel_data_clipped)

        # Convert list to a numpy array for histogram calculation
        all_pixel_values = np.array(all_pixel_values)

        # Calculate the combined histogram from the concatenated pixel values
        hist, bins = np.histogram(all_pixel_values, bins=256, range=(0, 2500))

        # Plot the combined (averaged) histogram
        axs[i].bar(bins[:-1], hist, width=(bins[1] - bins[0]), color='gray', alpha=0.7)
        axs[i].set_xlim(0, 2500)
        axs[i].set_xlabel('Intensity')
        axs[i].set_ylabel('Frequency')
        axs[i].set_title(f"Average Histogram for {channel_names[i]}", fontsize=12)

        # **Otsu Threshold Calculation on the Combined Pixel Values**
        otsu_threshold = threshold_otsu(all_pixel_values)  # Otsu on the full pixel values

        # **Mean and Median on Combined Histogram** 
        mean_val = np.mean(all_pixel_values)
        median_val = np.median(all_pixel_values)

        # Plot the Otsu, mean, and median lines
        axs[i].axvline(otsu_threshold, color='red', linestyle='--', label=f'Otsu: {otsu_threshold:.2f}')
        axs[i].axvline(mean_val, color='blue', linestyle='-.', label=f'Mean: {mean_val:.2f}')
        axs[i].axvline(median_val, color='green', linestyle=':', label=f'Median: {median_val:.2f}')

        # Add legend for the lines
        axs[i].legend()

    # Add the main title
    fig.suptitle("Combined Average Histograms for Multiple Images", fontsize=16)

    # Adjust spacing between plots
    plt.subplots_adjust(hspace=0.4, top=0.9)

    # Make layout tight
    plt.tight_layout()
    plt.show()

### Multiple threshold DAPI channel

def plot_dapi_histogram_and_thresholds(image_name, downscaled_image):
    """
    Plot the histogram of the DAPI channel, the original image (vmin=500, vmax=2500),
    the mean thresholded image, and a single Multi-Otsu thresholded image with 
    specific lower and upper bound handling.
    
    Args:
        image_name (str): Name of the image for display.
        downscaled_image (np.ndarray): Input image with DAPI channel as the 4th channel.
    """
    # Select only the DAPI channel (channel 3, index 3)
    dapi_channel_data = downscaled_image[:, :, 3]

    # Calculate the histogram for the DAPI channel
    hist, bins = np.histogram(dapi_channel_data, bins=256, range=(0, 3000))

    # Apply Multi-Otsu threshold
    start_time = time.time()
    multiotsu_thresholds = threshold_multiotsu(dapi_channel_data, classes=3)
    multiotsu_time = time.time() - start_time

    # Apply mean threshold
    start_time = time.time()
    mean_threshold = dapi_channel_data.mean()
    mean_time = time.time() - start_time

    # Generate thresholded images
    mean_thresh_img = np.where(dapi_channel_data >= mean_threshold, dapi_channel_data, 0)
    multiotsu_thresh_img = np.where(
        dapi_channel_data < multiotsu_thresholds[0], 0,
        np.where(dapi_channel_data > multiotsu_thresholds[1], multiotsu_thresholds[1], dapi_channel_data)
    )

    # Plot the results in a single row
    fig, axs = plt.subplots(1, 4, figsize=(24, 6))

    # Histogram
    axs[0].bar(bins[:-1], hist, width=(bins[1] - bins[0]), color='gray', alpha=0.7)
    axs[0].axvline(multiotsu_thresholds[0], color='red', linestyle='--', label=f'Lower Multi-Otsu: {multiotsu_thresholds[0]:.2f}')
    axs[0].axvline(multiotsu_thresholds[1], color='blue', linestyle='--', label=f'Upper Multi-Otsu: {multiotsu_thresholds[1]:.2f}')
    axs[0].axvline(mean_threshold, color='purple', linestyle='--', label=f'Mean: {mean_threshold:.2f}')
    axs[0].set_xlim(0, 3000)
    axs[0].set_xlabel('Intensity')
    axs[0].set_ylabel('Frequency')
    axs[0].legend(fontsize=10)
    axs[0].set_title('Histogram for DAPI Channel')

    # Original image
    axs[1].imshow(dapi_channel_data, cmap='gray', vmin=500, vmax=2500)
    axs[1].axis('off')
    axs[1].set_title('Original DAPI Image (500-2500)')

    # Mean thresholded image
    axs[2].imshow(mean_thresh_img, cmap='gray', vmax=2500)
    axs[2].axis('off')
    axs[2].set_title('Mean Thresholded Image')

    # Multi-Otsu thresholded image
    axs[3].imshow(multiotsu_thresh_img, cmap='gray')
    axs[3].axis('off')
    axs[3].set_title('Multi-Otsu Thresholded Image')

    # Add the main title with computation times
    fig.suptitle(f"{image_name} | Multi-Otsu: {multiotsu_time:.4f}s | Mean: {mean_time:.4f}s", fontsize=16)

    # Adjust layout and display
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()

################## Test
# Create downscaled images and save to pickle
# downscale_all_images(base_path, scale_percent = 10, outfile = "downscaled_images")

# # Load the results dictionary
# with open('downscaled_images.pkl', 'rb') as f:
#     results_dict = pickle.load(f)

# average_size = average_image_size(results_dict)    
# plot_all_downscaled(results_dict)

# # # Extract image names and downscaled images from the results_dict
# image_names = [value['image_name'] for value in results_dict.values()]
# downscaled_images = [value['downscaled_image'] for value in results_dict.values()]
# plot_combined_hist_otsu_multiple_images(image_names, downscaled_images, channel_names = ["T cell", "B cell", "HEV", "DAPI"])

# for key, value in results_dict.items():
#     image_name = value['image_name']
#     downscaled_image = value['downscaled_image']
#     plot_dapi_histogram_and_thresholds(image_name, downscaled_image)




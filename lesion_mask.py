import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import stackview

# Scikit image
from skimage.io import imread
from skimage import color, segmentation, filters, measure
from skimage.segmentation import slic
from skimage.measure import label, regionprops

from skimage.measure import regionprops
from scipy.ndimage import zoom
from scipy.ndimage import gaussian_filter, generic_filter
from scipy.ndimage import binary_closing
from scipy.stats import mode
from scipy.ndimage import gaussian_filter, generic_filter
from scipy import ndimage as ndi

# pyclesperanto
import pyclesperanto_prototype as cle
from pyclesperanto_prototype import imshow

# Cellpose
from cellpose import core, utils, io, models, metrics, plot
from cellpose.io import logger_setup
import cv2


# Image loading modules
import os
from aicsimageio import AICSImage
import glob
import pickle
import matplotlib.cm as cm


from scipy import ndimage as ndi

from skimage import filters, measure, morphology, segmentation
from scipy.ndimage import center_of_mass

import time
import cv2

from PIL import Image

#### Personal ####
import preprocessing as pp


################### Code snippets    ##############################

def blob_mask(image, sigma=5, intensity_threshold=0.1):
    # Step 1: Smooth the image using Gaussian filter
    smooth = filters.gaussian(image, sigma=sigma)
    
    # Step 2: Adjust the threshold to include low-intensity blobs
    # Here, we use a lower intensity threshold based on a fixed percentage of the image's maximum intensity.
    thresh_value = filters.threshold_otsu(smooth)
    low_thresh_value = np.max(smooth) * intensity_threshold  # Use a relative intensity threshold
    
    # Apply the threshold to identify potential blobs
    thresh = smooth > low_thresh_value
    
    # Step 3: Perform morphological closing to fill small gaps and merge nearby blobs
    # This will help merge fragmented blobs and capture lower-intensity areas.
    closed = ndi.binary_closing(thresh, structure=np.ones((3, 3)))  # Apply a 3x3 kernel for closing
    
    # Step 4: Fill holes within blobs
    filled = ndi.binary_fill_holes(closed)
    
    # Step 5: Label the connected blobs
    labeled_array, num_features = ndi.label(filled)

    # Step 6: Find the centroid of each blob
    image_center = np.array(filled.shape) / 2
    centroids = center_of_mass(filled, labeled_array, range(1, num_features + 1))
    
    # Step 7: Calculate distances from image center to the centroids of blobs
    distances = [np.linalg.norm(np.array(c) - image_center) for c in centroids]
    
    # Step 8: Identify the central blob as the one closest to the center
    central_blob_index = np.argmin(distances) + 1
    central_blob_mask = labeled_array == central_blob_index
    
    return central_blob_mask

def std_filter(image,kernel_size=20, std_thresh=0.2):
    std_image = generic_filter(image,np.std,size=kernel_size)
    std_mask =  std_image < std_thresh
    valid_mask = image > 0
    final_mask = np.logical_and(std_mask, valid_mask)
    return final_mask


def get_lesion_mask(image, kernel_size=20, sigma = 20,std_thresh=0.2, scale_percent = 10, re_gauss_sigma = 20, re_gauss_threshold = 0.5): #, gauss_thresh=0.3
        
    # Timer for the entire function
    DAPI_channel = image[:, :, 3]

    # Calculate new dimensions
    original_shape = image.shape[:2]  # (height, width)
    width = int(original_shape[1])
    height = int(original_shape[0])
    dim = (width, height)

    # Step 1.1: Preprocessing and downscale
    ds_img = pp.downscale_image(DAPI_channel, scale_percent=scale_percent) 

    # Step 2: Tissue blob detection
    binary_mean_image = (ds_img > np.mean(ds_img)).astype(np.float32)    
    processed_binary = pp.quantile_normalize(binary_mean_image, alpha = 0.1)
    central_blob_mask = blob_mask(processed_binary, sigma=sigma)

    # Step 1.2: Preprocess 
    processed_img = pp.preprocess_image(ds_img, max_intensity=2500, alpha = 0.01)

    # Step 3: Standard deviation filter
    std_mask = std_filter(processed_img, kernel_size=kernel_size, std_thresh=std_thresh)

    # Step 4: Combine masks
    lesion_mask = np.logical_and(std_mask, central_blob_mask)
    
    # Step 5: Re-gauss
    lesion_mask_gauss = gaussian_filter(lesion_mask.astype(float),sigma=re_gauss_sigma)>re_gauss_threshold
    
    # Step 6: Upscale
    scaled_mask = cv2.resize(lesion_mask_gauss.astype(np.uint8), dim, interpolation=cv2.INTER_NEAREST)
    scaled_blob_mask = cv2.resize(central_blob_mask.astype(np.uint8), dim, interpolation=cv2.INTER_NEAREST)

    return scaled_mask, scaled_blob_mask



from PIL import Image
import numpy as np
import os
import time
import matplotlib
matplotlib.use('TkAgg')  # for interactive display

# import the functions from affine_transformations.py
from affine_transformations import (
    apply_affine_transform_forward,
    apply_affine_transform_backward_no_interpolation,
    apply_affine_transform_backward_bilinear,
    get_affine_transform_matrix,
    clip_and_convert
)

# create the assets directory if it doesn't exist
if not os.path.exists('./assets/outputs'):  # for outputs
    os.makedirs('./assets/outputs')

# load the image
image = Image.open('./assets/inputs/istanbul.jpg')  
image_np = np.array(image)

# get the image dimensions (height and width)
height, width = image_np.shape[:2]

# create the transform and inverse transform matrices
transform_matrix, inverse_transform_matrix = get_affine_transform_matrix(height, width)

# to show loading animation
def loading_animation(step_name):
    print(f"\nprocessing {step_name}...")
    for i in range(10):
        time.sleep(0.1)  # simulate processing time
        print(".", end="", flush=True)
    print()  # new line after loading animation

# apply forward mapping
loading_animation("forward mapping")
transformed_image_forward = apply_affine_transform_forward(image_np, transform_matrix)

# convert the transformed image to a numpy array
transformed_image_forward = np.array(transformed_image_forward, dtype=np.uint8)

# save the transformed image
Image.fromarray(transformed_image_forward).save('./assets/outputs/forward_mapping.png')

# apply backward mapping without interpolation
loading_animation("backward mapping - no interpolation")
transformed_image_backward_no_interp = apply_affine_transform_backward_no_interpolation(image_np, inverse_transform_matrix)

# convert to numpy array before saving
transformed_image_backward_no_interp = np.array(transformed_image_backward_no_interp, dtype=np.uint8)
Image.fromarray(transformed_image_backward_no_interp).save('./assets/outputs/backward_no_interpolation.png')

# apply backward mapping with bilinear interpolation
loading_animation("backward mapping - bilinear interpolation")
transformed_image_backward_bilinear = apply_affine_transform_backward_bilinear(image_np, inverse_transform_matrix)

# clip and convert the transformed image
transformed_image_backward_bilinear = clip_and_convert(transformed_image_backward_bilinear)

# convert to numpy array before saving
transformed_image_backward_bilinear = np.array(transformed_image_backward_bilinear, dtype=np.uint8)
Image.fromarray(transformed_image_backward_bilinear).save('./assets/outputs/backward_bilinear_interpolation.png') 


print("images saved in ./assets/outputs folder.")

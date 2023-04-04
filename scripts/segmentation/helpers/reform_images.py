"""Multiple Image Reforming Script

Writes a folder of images to a text file

Image formats supported can be found here:
https://pillow.readthedocs.io/en/stable/handbook/image-file-formats.html

@Author: Patrick Roe (http://pproe.dev)

"""

from os import listdir
from os.path import isfile, join

import numpy as np
from PIL import Image

# Ask user for inputs of image and output file paths
folder_path = input("Enter path of folder containing images: ")
output_file_path = input("Enter path to save output: ")

# Open image files, return error if unsuccessful
try:
    image_file_paths = [f for f in listdir(folder_path) if isfile(join(folder_path, f))]
    images = [Image.open(join(folder_path, path)) for path in image_file_paths]
except FileNotFoundError:
    print("Image Folder not found. Please check the path and try again.")
except:
    print("Error opening specified image folder.")

# Convert to numpy array
data = np.asarray([np.asarray(image) for image in images]).flatten().astype("float64")

# Open output file to write data to, return error if unsuccessful
try:
    file_buffer = open(output_file_path, "w")
    np.savetxt(file_buffer, data, fmt="%1.4f")
except:
    print("Error writing to specified path.")

# Clean up file buffer
file_buffer.close()

print(f"Successfully wrote {len(data)} lines to {output_file_path}")

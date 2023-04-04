"""Single Image Reforming Script

Writes a single image to a text file.

Image formats supported can be found here:
https://pillow.readthedocs.io/en/stable/handbook/image-file-formats.html

@Author: Patrick Roe (http://pproe.dev)

"""

from PIL import Image
import numpy as np

# Ask user for inputs of image and output file paths
image_file_path = input("Enter image path: ")
output_file_path = input("Enter path to save output: ")

# Open image file, return error if unsuccessful
try:
    image = Image.open(image_file_path)
except FileNotFoundError:
    print("File not found. Please check the path and try again.")
except:
    print("Error opening specified image.")

# Convert to numpy array
data = np.asarray(image).flatten().astype("float64")

# Open output file to write data to, return error if unsuccessful
try:
    file_buffer = open(output_file_path, "w")
    np.savetxt(file_buffer, data, fmt="%1.4f")
except:
    print("Error writing to specified path.")

# Clean up file buffer
file_buffer.close()

print(f"Successfully wrote {len(data)} lines to {output_file_path}.")

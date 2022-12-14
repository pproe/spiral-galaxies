"""
This module provides functions for formatting a folder of image files into a .dat
file of float pixel intensity values for use in machine learning applications.
@author: Patrick Roe (http://pproe.dev)
"""

# Import Packages
import time
import math
from pathlib import Path
from PIL import Image
import numpy as np


# File Locations
INPUT_IMAGES_DIRECTORIES = ["./Tadaki+20_spiral/Tadaki+20_S", "./Tadaki+20_spiral/Tadaki+20_Z"]
INPUT_LABELS_LOCATION = "./Tadaki+20_spiral/Tadaki+20_spiral.cat"
OUTPUT_IMAGES_FILE = "./Tadaki_images.dat"
OUTPUT_LABELS_FILE = "./Tadaki_labels.dat"

# Global Constants
IMAGE_SIZE_X = 64
IMAGE_SIZE_Y = 64


def print_progress_bar(
    iteration,
    total,
    prefix="",
    suffix="",
    decimals=1,
    length=100,
    fill="█",
    print_end="\r",
):
    """
    Prints terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(
        100 * (iteration / float(total))
    )
    filled_length = int(length * iteration // total)
    progress_bar = fill * filled_length + "-" * (length - filled_length)
    print(f"\r{prefix} |{progress_bar}| {percent}% {suffix}", end=print_end)
    # Print New Line on Complete
    if iteration == total:
        print()


def estimate_time_remaining(
    iteration,
    total,
    avg_time,
):
    """
    Returns the remaining time for a process (as formatted string)
    @params:
        iteration   - Required : current iteration (Int)
        total       - Required : total iterations (Int)
        avg_time    - Required : average time for process so far in seconds (Int)
    """

    seconds_remaining = (total - iteration) * avg_time

    minutes_remaining = math.floor(seconds_remaining / 60)
    seconds_remaining = seconds_remaining - (minutes_remaining * 60)

    if minutes_remaining == 0:
        return f"{seconds_remaining:.0f}s"

    return f"{minutes_remaining:.0f}m {seconds_remaining:.0f}s"

def extract_labels(input_label_location):
    """
    Returns dictionary with spiral classification of each image ID
    0 is non-spiral as predicted by the Tadaki CNN and 1 is spiral
    """
    label_dict = {}
    
    input_path = Path(input_label_location)
    if not input_path.exists():
        print("Input labels location does not exist.")
        return -1

    input_file = open(input_path, "r", encoding="UTF-8")
    
    for line in input_file:
        line_data = line.split()
        
        # Skip Comment lines
        if line_data[0] == "#":
            continue
        
        prediction = abs(round(float(line_data[13]))-1)
        
        label_dict[line_data[0]] = prediction
    
    input_file.close()
    
    return label_dict

def extract_images(input_locations, output_image_location, output_label_location):
    """
    Outputs all image data in 64x64 format in output_location
    @params:
        input_location  - Required : location for input images in jpg format (String)
        output_location - Required : location for output text file (String)
    """

    # Clear output files
    output_image_path = Path(output_image_location)
    output_label_path = Path(output_label_location)
    
    open(output_image_path, "w", encoding="UTF-8").close()
    open(output_label_path, "w", encoding="UTF-8").close()
    
    # Get all locations of filenames and store in list
    for idx, input_location in enumerate(input_locations):
        
        print(f'Processing folder {idx+1} of {len(input_locations)}')
        input_path = Path(input_location)
        
        if not input_path.exists():
            print("Input Image location does not exist.")
            return -1
        filenames = list(input_path.glob("*.jpg"))

        # Open Output buffer
        output_image_buffer = open(output_image_path, "ab")
        output_label_buffer = open(output_label_path, "ab")

        # Initiate progress bar
        num_files = len(filenames)
        print_progress_bar(
            0, num_files, prefix="Progress:", suffix="Complete", length=50
        )

        # Variable to track estimated time
        avg_time = 0

        # Open each image location in filenames and append results to output_location
        for i, file in enumerate(filenames):

            start_time = time.time()

            image = Image.open(file)
            # Resize image if necessary
            if not image.size == (IMAGE_SIZE_X, IMAGE_SIZE_Y):
                image = image.resize((IMAGE_SIZE_X, IMAGE_SIZE_Y))

            # Convert to numpy array, flatten, and normalize pixel intensities
            data = np.asarray(image).flatten().astype("float64")
            data_max = np.amax(data)
            data = data / data_max

            # Write data to output file
            # output_buffer.write(b"\n")
            np.savetxt(output_image_buffer, data, fmt="%1.4f")
            
            # Write data to output labels file
            output_label_buffer.write(b"1\n")

            # Estimate time remaining and Update Progress Bar
            end_time = time.time() - start_time
            avg_time = (avg_time * i + end_time) / (i + 1)
            time_remaining = estimate_time_remaining(i + 1, num_files, avg_time)
            print_progress_bar(
                i + 1,
                num_files,
                prefix="Progress:",
                suffix=f"Complete, Estimated {time_remaining} remaining.",
                length=50,
            )

        output_image_buffer.close()
        output_label_buffer.close()


if __name__ == "__main__":
    #print(extract_labels(INPUT_LABELS_LOCATION))
    extract_images(INPUT_IMAGES_DIRECTORIES, OUTPUT_IMAGES_FILE, OUTPUT_LABELS_FILE)

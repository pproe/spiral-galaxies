import numpy as np
import matplotlib.pyplot as plt

ORIG_IMG_FILE= r"E:\Documents\Thesis\spiral-galaxies\segmentation\tadaki_segmentation\set_2_orig.dat" #../tadaki_segmentation/set_2_orig.dat"
SEG_IMG_FILE=r"E:\Documents\Thesis\spiral-galaxies\segmentation\tadaki_segmentation\set_2_segmentation.dat" #"../tadaki_segmentation/set_2_segmentation.dat"

#ORIG_IMG_FILE="../tadaki_segmentation/tadaki_images.dat"
#SEG_IMG_FILE="../tadaki_segmentation/tadaki_images_pred.dat"

IMG_HEIGHT=64
IMG_WIDTH=64
NUM_IMAGES=100

with open(ORIG_IMG_FILE) as f:
  lines=f.readlines()

with open(SEG_IMG_FILE) as f:
  lines0=f.readlines()

#print("len(lines)",len(lines))
#print("len(lines0)",len(lines0))  

orig_images = np.zeros((NUM_IMAGES, IMG_HEIGHT, IMG_WIDTH))
seg_images = np.zeros((NUM_IMAGES, IMG_HEIGHT, IMG_WIDTH))

for num, j in enumerate(lines):
  if num >= NUM_IMAGES*IMG_HEIGHT*IMG_WIDTH:
    break
  orig = j.strip().split()
  seg = lines0[num].strip().split()
  img_num = num // (IMG_HEIGHT*IMG_WIDTH)
  pixel_num = num % (IMG_HEIGHT*IMG_WIDTH)
  orig_images[img_num, pixel_num // IMG_HEIGHT, pixel_num % IMG_WIDTH] = float(orig[0])
  seg_images[img_num, pixel_num // IMG_HEIGHT, pixel_num % IMG_WIDTH] = float(seg[0])

for num, img in enumerate(orig_images):
  if num >= NUM_IMAGES:
    break

  fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
  fig.suptitle(f'U-Net Segmentation Test: Image {num:04}')
  ax1.imshow(img, cmap='gray')
  ax3.imshow(img, cmap='gray')
  seg_binary = np.where(seg_images[num] > 0.5, 1, 0)
  ax2.imshow(seg_binary, cmap='gray')
  ax3.imshow(seg_binary, cmap='Reds', alpha=0.5)

  # Display titles
  ax1.set_title('Original')
  ax2.set_title('Segmented')
  ax3.set_title('Overlayed')

  # Turn off tick labels
  ax1.axis('off')
  ax2.axis('off')
  ax3.axis('off')
  
  plt.savefig(f'../images/tadaki/set_2_ground_truth/{num:04}.png')
  
  # Close figure to prevent memory leak
  plt.close()
  
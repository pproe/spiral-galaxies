import numpy as np
import matplotlib.pyplot as plt

#ORIG_IMG_FILE="../tadaki_segmentation/tadaki_images.dat"
#SEG_IMG_FILE="../tadaki_segmentation/tadaki_masks.dat"

ORIG_IMG_FILE="../tadaki_segmentation/set_2_orig.dat"
TRUTH_SEG_IMG_FILE="../tadaki_segmentation/set_2_segmentation.dat"
SEG_IMG_FILE="../tadaki_segmentation/set_2_pred.dat"

IMG_HEIGHT=64
IMG_WIDTH=64
NUM_IMAGES=100

def calculate_jaccard(im1, im2):

    im1 = np.asarray(im1).astype(np.bool)
    im2 = np.asarray(im2).astype(np.bool)

    if im1.shape != im2.shape:
        raise ValueError("Shape mismatch: im1 and im2 must have the same shape.")

    intersection = np.logical_and(im1, im2)

    union = np.logical_or(im1, im2)

    return intersection.sum() / float(union.sum())

def calculate_dice(im1, im2, empty_score=1.0):

    im1 = np.asarray(im1).astype(np.bool)
    im2 = np.asarray(im2).astype(np.bool)

    if im1.shape != im2.shape:
        raise ValueError("Shape mismatch: im1 and im2 must have the same shape.")

    im_sum = im1.sum() + im2.sum()
    if im_sum == 0:
        return empty_score

    # Compute Dice coefficient
    intersection = np.logical_and(im1, im2)

    return 2. * intersection.sum() / im_sum

with open(ORIG_IMG_FILE) as f:
  lines=f.readlines()
  
with open(TRUTH_SEG_IMG_FILE) as f:
  lines0=f.readlines()

with open(SEG_IMG_FILE) as f:
  lines1=f.readlines()

#print("len(lines)",len(lines))
#print("len(lines0)",len(lines0))  

orig_images = np.zeros((NUM_IMAGES, IMG_HEIGHT, IMG_WIDTH))
truth_seg_images = np.zeros((NUM_IMAGES, IMG_HEIGHT, IMG_WIDTH))
seg_images = np.zeros((NUM_IMAGES, IMG_HEIGHT, IMG_WIDTH))

for num, j in enumerate(lines):
  if num >= NUM_IMAGES*IMG_HEIGHT*IMG_WIDTH:
    break
  orig = j.strip().split()
  truth_seg = lines0[num].strip().split()
  seg = lines1[num].strip().split()
  
  img_num = num // (IMG_HEIGHT*IMG_WIDTH)
  pixel_num = num % (IMG_HEIGHT*IMG_WIDTH)
  
  orig_images[img_num, pixel_num // IMG_HEIGHT, pixel_num % IMG_WIDTH] = float(orig[0])
  truth_seg_images[img_num, pixel_num // IMG_HEIGHT, pixel_num % IMG_WIDTH] = float(truth_seg[0])
  seg_images[img_num, pixel_num // IMG_HEIGHT, pixel_num % IMG_WIDTH] = float(seg[0])

for num, img in enumerate(orig_images):
  if num >= NUM_IMAGES:
    break

  fig, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2, 3)
  fig.suptitle(f'U-Net Segmentation Test: Image {num:04} (Truth vs Prediction)')
  ax1.imshow(img, cmap='gray')
  ax3.imshow(img, cmap='gray')
  ax2.imshow(truth_seg_images[num], cmap='gray')
  ax3.imshow(truth_seg_images[num], cmap='Reds', alpha=0.5)
  seg_binary = np.where(seg_images[num] > 0.5, 1, 0)
  ax5.imshow(seg_binary, cmap='gray')
  ax6.imshow(seg_binary, cmap='Reds', alpha=0.5)
  
  # Calculate Dice
  dice = calculate_dice(truth_seg_images[num], seg_binary)
  jaccard = calculate_jaccard(truth_seg_images[num], seg_binary)
  ax5.set_title(f'Dice: {dice:.4f}')
  ax6.set_title(f'Jaccard: {jaccard:.4f}')

  # Display titles
  ax1.set_title('Original')
  ax2.set_title('Segmented')
  ax3.set_title('Overlayed')

  # Turn off tick labels
  ax1.axis('off')
  ax2.axis('off')
  ax3.axis('off')
  ax4.axis('off')
  ax5.axis('off')
  ax6.axis('off')
  
  plt.savefig(f'../images/tadaki/12_augmentation/{num:04}.png')
  
  # Close figure to prevent memory leak
  plt.close()
  
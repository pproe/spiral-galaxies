from pathlib import Path
from PIL import Image

IMAGE_DIRECTORY=r"..\data\Tadaki_Segmentation\orig_images"
BITMAP_DIRECTORY=r"..\data\Tadaki_Segmentation\segmentation"
OUT_DIRECTORY=r"..\data\Tadaki_Segmentation\overlays"

def main(input_location):
  input_path = Path(input_location, bitmap_location, out_location)
          
  if not input_path.exists():
      print("Input Image location does not exist.")
      return -1
    
  filenames = list(input_path.glob("*.jpg"))
  
  for file in filenames:

    image = Image.open(file)
    
    

if __name__=='__main__':
  main(IMAGE_DIRECTORY)
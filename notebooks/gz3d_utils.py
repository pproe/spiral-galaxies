from astropy.table import Table
from astropy.io import fits
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import pandas as pd
import os
from multiprocessing.pool import Pool
from time import sleep
from contextlib import redirect_stdout

class SpiralMaskDataManager:
    
    def __init__(self, metadata_filename, data_endpoint, out_dir):
        self.metadata_filename = metadata_filename
        self.data_endpoint = data_endpoint
        self.out_dir = out_dir

    def fetch_metadata(self):

        local_path = os.path.join(self.out_dir, self.metadata_filename)
        gz3d_df = -1

        if os.path.isfile(local_path):
            print("Local metadata file found!")
            t = Table.read(local_path)
            gz3d_df = t.to_pandas()
            
            # Remove entries with no spiral mask, according to this page:
            # https://data.sdss.org/datamodel/files/MANGA_MORPHOLOGY/galaxyzoo3d/GZ3DVER/gz3d_metadata.html
            print("Extracting galaxies of interest...")
            gz3d_df = gz3d_df.loc[gz3d_df['GZ_spiral_votes']/gz3d_df['GZ_total_classifications']>0.2] 
        
        else:
            print("No local metadata found.")
            print("Fetching metadata...")
            t = Table.read(f'{self.data_endpoint}/{self.metadata_filename}')
            t.write(local_path, format='fits')
            print(f"Saved local metadata backup at {local_path}")
            gz3d_df = t.to_pandas()
            
            # Remove entries with no spiral mask, according to this page:
            # https://data.sdss.org/datamodel/files/MANGA_MORPHOLOGY/galaxyzoo3d/GZ3DVER/gz3d_metadata.html
            print("Extracting galaxies of interest...")
            gz3d_df = gz3d_df.loc[gz3d_df['GZ_spiral_votes']/gz3d_df['GZ_total_classifications']>0.2] 
            
        return gz3d_df
    
    def save_originals(self, file_name, verbose=False):

        # Format filenames
        file_name = file_name.decode('utf-8').strip()
        new_fn = file_name.split('.')[0]

        out_img = os.path.join(self.out_dir, 'orig_img', f'{new_fn}.png')
        out_msk = os.path.join(self.out_dir, 'orig_msk', f'{new_fn}.png')

        # Skip files that are already processed
        if os.path.isfile(out_img) and os.path.isfile(out_msk):
            if verbose: print(f"{file_name} already processed. Ignoring...") 
            return

        if verbose: print(f"Processing file: {file_name}") 
        # Read Image from SDSS dataset
        with open('fits.log', 'w') as log:
            with redirect_stdout(log):
                with fits.open(f'{self.data_endpoint}/gz3d_{file_name}.gz') as hdulist:
                    img = Image.fromarray(hdulist[0].data).convert('L')
                    msk = Image.fromarray(hdulist[3].data).convert('L')
        

        # Don't include blank masks
        if not (np.sum(msk) > 0):
            if verbose: print(f"{file_name} has no visible mask. Ignoring...")
            return

        img.save(out_img, 'PNG')
        msk.save(out_msk, 'PNG')
        if verbose: print(f"{file_name} saved successfully.") 
    
if __name__ == '__main__':
    # Parallelized version of running extraction
    
    dm = SpiralMaskDataManager('gz3d_metadata.fits', 
                               'https://data.sdss.org/sas/dr17/manga/morphology/galaxyzoo3d/v4_0_0',
                               os.path.join(os.pardir, 'data', 'galaxyzoo3d'))
    df = dm.fetch_metadata()
    
    file_names = np.array(df['file_name'])
    
    def track_job(job, update_interval=3):
        while job._number_left > 0:
            print("Tasks remaining = {0}".format(
            job._number_left * job._chunksize))
            sleep(update_interval)
            
        if job.successful():
            print("Images processed successfully")

    # create and configure the process pool
    with Pool() as pool:
        
        def kill_pool(err_msg):
            print(err_msg)
            pool.terminate()
        
        res = pool.map_async(dm.save_originals, file_names, error_callback=kill_pool)
        
        track_job(res)
    # process pool is closed automatically
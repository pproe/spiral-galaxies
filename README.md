# Classification and Segmentation of Spiral Galaxies using Keras

This repository hosts all relevant python scripts and notebooks used in my final-year application of the U-Net Image Segmentation model to images of spiral galaxies. This work was completed as part of my final-year research project to achieve my Masters Degree in Professional Engineering (Software). 

> **Update (08/03/23)** - At this stage of the project I am reaching the final stretch of the research, transitioning my focus to collating my results and writing the accompanying paper. The model has performed promisingly on prelimnary data-sets at this stage, and I will be tweaking hyperparameters and data augmentation techniques. I intend to add more to this README as the project matures, along with a IPython notebook to make the project more usable.

> **Update (25/03/23)** - Image segmentation of the small dataset of 100 images (with an augmentation factor of ~118x) has produced results exceeeding my expectations for such a small set of data on this relatively complex task. I am focusing on complementing my dataset with the Galaxy Zoo 3D from the MaNGA DR17 to investigate if this will improve my results. This will lead into analysis of the flux ratio of the galaxies imaged by the HSC at the Subaru Telescope.

> > **Update (05/07/23)** - The codebase is updated with the final compilation of python notebooks and accompanying scripts for generating the graphics used for the thesis report. Complementing the Subaru Telescope Dataset with data obtained from the GZ3D data release provided improvements in performance that are discussed further in the final paper. Future work would involve application of more appropriate pre-processing techniques to utilise the GZ3D data and/or generation of spiral segmentation masks for the images obtained by the Subaru Telescope (with the goal of dataset specificity).

## Sample Results

### Classification

![Confusion Matrix of 3-Class Classification of Galaxy Morphology](/images/confusion_200epoch_100batch.png)

> The above confusion matrix demonstrates the performance of the mulit-class classification model when presented with the dataset provided by [Nair & Abraham](https://arxiv.org/abs/1001.2401), calatoging the SDSS DR4. It shows reliable performance, particularly in distinguishing spiral galaxies, which are the focus of this study.

### Segmentation

Using big image data taken from Subaru/Hyper Suprime-Cam (HSC) Survey, which was catalogued to isolate spiral galaxies by [Tadaki et al.](https://academic.oup.com/mnras/article-abstract/496/4/4276/5866497), the intent here is to train an implementation of the [U-Net](https://lmb.informatik.uni-freiburg.de/people/ronneber/u-net/) to segment the spiral arms from images of spiral galaxies allowing various analysis of the galaxies, in particular Flux Ratio of spiral to surrounding regions.

#### Dataset of Size 10

![Demonstration of Performance of very small dataset with U-Net](/images/segmentation_10_images.png)

> The above figure displays an example of the kind of results the model produces with an extremely small dataset of 10 augmented by a factor of 10x.


#### Dataset of Size 100

![Demonstration of Performance of small dataset with U-Net](/images/20230322_test_predictions_reduced.png)

> The above figure displays various masks produced by the model when trained with a dataset of 68 augmented by a factor of ~118x (8024 training images total).

#### Dataset of Size 1599

![Demonstration of Performance of larger dataset with U-Net](/images/20230410_test_predictions_reduced.png)

> The above figure displays various masks produced by the model when trained with a dataset of 1087 augmented by a factor of ~9.2x (10000 training images total). The images from the Hyper-Suprime Cam were supplemented with images and segmentation masks obtained throught the GalaxyZoo3D project.

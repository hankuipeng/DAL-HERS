# Hierarchical Entropy Rate Superpixel Segmentation with Learned Affinities

This is an implementation of the methodology introduced in our work:

**Hierarchical Entropy Rate Superpixel Segmentation with Learned Affinities**

[Hankui Peng](https://hankuipeng.github.io/), [Angelica I. Aviles-Rivero](https://angelicaiaviles.wordpress.com/), and [Carola-Bibiane Sch\"{o}nlieb](https://www.damtp.cam.ac.uk/user/cbs31/Home.html)

Please contact Hankui Peng (hp467@cam.ac.uk) if you have any question.


## Prerequisites
The training code was mainly developed and tested with Python 3.8.5, PyTorch 1.7.0, OpenCV 4.4.0, and Ubuntu 20.04.


## Build HERS module in Python
Type the following in terminal under the main project repository:
```
make module 
```
This would create ```./pybuild/hers_superpixel.*.so```, which can then be imported directly in Python as ```import hers_superpixel```.


## Testing
We provide code to generate superpixel segmentation results (in .csv and .png) for a folder of input images given: 1) a sequence of numbers of desired superpixels and 2) a specific number of superpixels. For testing, we provide 10 sample images from the BSDS500 data set in ```./sample_imgs/input/``` as input.

To produce segmentation results for various numbers of superpixels, run in terminal: 
```
bash analysis_multi_nC.sh
```

To produce segmentation results for a given number of superpixels, run in terminal: 
```
bash analysis_single_nC.sh
```

To test on other data sets,  please first collect all the images into one folder ```<CUSTOM_DIR>```, and then convert them into the same format (e.g. ```.png``` or ```.jpg```) if necessary. Then, simply modify the ```INPUT_DIR``` parameter within ```analysis_multi_nC.sh``` or ```analysis_single_nC.sh``` and follow the instructions above.


## Evaluation
We use the code from [superpixel benchmark](https://github.com/davidstutz/superpixel-benchmark) for superpixel evaluation. A detailed  [instruction](https://github.com/davidstutz/superpixel-benchmark/blob/master/docs/BUILDING.md) is available in the repository.

 
## Acknowledgement
The C++ part of our code is developed based on the code provided by [SH](https://github.com/semiquark1/boruvka-superpixel). 

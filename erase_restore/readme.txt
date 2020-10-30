This folder contains the implementation of our Erase&Restore approach.

Please note:
1) The script provided here is used for ImageNet. The implementation for CIFAR10 (or other small-sized images) is different because no PCA is needed.
2) There may be various random masks generation methods. What we present here is a sample that is specifically designed for Telea’s inpainting algorithm.
   According to Telea’s paper, the inpainting algorithm we adopt performs very well when the portion of corrupted pixels in an image is less than 15%.

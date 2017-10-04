# PyAiminator

PyAiminator is a practice project and a demo for Tampere University of Technology
course ASE-7410. Course is about Image based measurements.

I approached the object detection as a concept by trying to detect character models
in a classic game of Counter-Strike 1.6. Most of the image processing is done with 
opencv.

## Strategy:
* Filtering and thresholding
* Analysing contours
* Matching features
* Matching color space

## Tests
![Test 1](/images/test/test1.jpg)

In a simple situations the detection works fine.

![Test 2](/images/test/test2.jpg)

When models overlap it causes problems

![Test 3](/images/test/test3.jpg)

Different map and similiar colored background caused problems

**Due to uncertain copyright the images files doesn't contain any templates of the models.**
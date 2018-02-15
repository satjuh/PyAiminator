# PyAiminator

PyAiminator is a practice project and a demo for Tampere University of Technology
course ASE-7410.

I approached the object detection as a concept by trying to detect character models
in a classic game of Counter-Strike 1.6. Most of the image processing is done with 
OpenCV.

The detection process is compared and evaluated using Tensorflow object detection API [link](https://github.com/tensorflow/models/tree/master/research/object_detection).


### Strategy:
* Filtering and thresholding
* Analysing contours
* Matching features
* Matching color space


### Requirements

- Python 3.6.4 (64-bit for tensorflow)
- Tensorflow [object_detection](https://github.com/tensorflow/models/tree/master/research/object_detection) for judge.


### Examples

Example usage:

1. List of all available commands
```
$ python aiminator.py -h
```

2. Run a live demo
```
$ python aiminator.py --demo live
```

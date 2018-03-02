# PyAiminator

PyAiminator is a practice project and a demo for Tampere University of Technology
course ASE-7410.

I approached the object detection as a concept by trying to detect character models
in a classic game of Counter-Strike 1.6. Most of the image processing is done with 
OpenCV.


### Detection Strategy:
* Filtering and thresholding
* Analysing contours
* Matching features
* Matching color space


### Evaluation:

The detection process is compared and evaluated using Tensorflow object detection API [link](https://github.com/tensorflow/models/tree/master/research/object_detection). Already trained model can detect a variety of different objects but I only used it to evaluate how many humans or different objects associated with Counter-Strike models can I detect. Another evalution method was using the model RPNplus by Shiyu Huang [link](https://github.com/huangshiyu13/RPNplus) and some of his code to make the model work in the way we needed it to work.

Accepted objects:
- Human
- Backpack

Tensorflow threshold: 0.4


### Requirements

Basic usage for collecting and using Brute Force detection:
- Python 3.6.4
- [RPNplus models](https://github.com/huangshiyu13/RPNplus)

(Optional)
- Python 64-bit
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

3. Display ImageProcess steps
```
$ python aiminator.py --demo steps
```


# Visual Odometry

This repository consists of implementation of the visual odometry using monocular camera images in C++ using OpenCV library on KITTI dataset.

### Algorithm Overview

1) Streaming the images.
2) Using FAST algorithm detect the key points in the first image frame and then track them in the next frame. 
3) Nister's 5-point algorithm with RANSAC is used to compute the essential matrix. 
4) From the essential matrix, R (rotational) and T (translational) is estimated. 
5) Concatinated the translational and rotational vectors from each frame.

### Result video




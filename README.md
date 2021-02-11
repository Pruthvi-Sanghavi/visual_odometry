# Visual Odometry for Autonomous Vehicle
This is a software written as a part of project for the course Perception for Autonomous Vehicles.

<p align="center">
  <img src="https://github.com/Pruthvi-Sanghavi/visual_odometry/blob/main/result.gif" height="250px"/>
  <img src="https://github.com/Pruthvi-Sanghavi/visual_odometry/blob/main/vo-res.png" height="250px"/>

</p>

### Contents
1. [Introduction](https://github.com/Pruthvi-Sanghavi/visual_odometry#introduction)
2. [Dataset and Processing](https://github.com/Pruthvi-Sanghavi/visual_odometry#dataset-and-processing)
3. [Dependencies](https://github.com/Pruthvi-Sanghavi/visual_odometry#dependencies)
4. [Pipeline](https://github.com/Pruthvi-Sanghavi/visual_odometry#pipeline) 

## Introduction
<p align="justify">
In robotics and computer vision, visual odometry is the process of determining the position and orientation of a robot by analyzing the associated camera images.
Visual Odometry is a crucial concept in Robotics Perception for estimating the trajectory of the
robot (the camera on the robot to be precise). The concepts involved in Visual Odometry are quite the same
for SLAM which needless to say is an integral part of Perception. In this project we are given frames of a
driving sequence taken by a camera in a car, and the scripts to extract the intrinsic parameters.
</p>

## Dataset and Processing
The dataset can be obtained at [Link](https://drive.google.com/drive/folders/1hAds4iwjSulc-3T88m9UDRsc6tBFih8a)

1. The first step in the project was the preparation of the given dataset and reading all the images in the
dataset. The input images are in Bayer format on which demosaicing function with GBRG alignment was
used. Thus, the Bayer pattern encoded image img was converted to a color image using the opencv function:
color image = cv2.cvtColor(img, cv2.COLOR BayerGR2BGR)

2. The next step in the data preparation phase was to extract the camera parameters using ReadCamer-
aModel.py as follows: fx , fy , cx , cy , G camera image , LUT = ReadCameraModel ( './model')

3. The images in the given dataset were further Undistorted using the current frame and next frame using
UndistortImage.py: undistorted image = UndistortImage(originalimage,LUT)

## Dependencies
- opencv
- numpy
- matplotlib
- scipy

## Pipeline
To estimate the 3D motion (translation and rotation) between successive frames in the sequence,
the following steps were followed.

1. Find point correspondences between successive frames using a keypoint algorithm of your
choice.
2. Estimate the Fundamental Matrix using the Eight-Point Algorithm within RANSAC
3. Estimation of Essential Matrix from the Fundamental matrix
4. Camera Pose computation from the Essential Matrix
5.Linear Triangulation

- For more information about the pipeline I welcome you to have a look at the [Report](https://github.com/Pruthvi-Sanghavi/visual_odometry/blob/main/ProjectReport.pdf)


# basic_grasping_algorithm
A very basic algorithm (implemented in Python) that generates a grasp based on an input point cloud.
The purpose of this algorithm is NOT to be performant, as it is used to demonstrate how to design an autonomous task with GRIP.

## Principle
The grasping algorithm implemented in this repository is valid under the following conditions:
* A single object is placed on a planar surface, without occlusion
* The height of the object is shorter than the gripper
* The object must be wider or larger than high

As previosuly mentioned, the point of this code is not to provide a state-of-the-art grasping algorithm, but rather to give a concrete example about how to integrate realistic code with several dependencies in the GRIP framework.

## Requirements
The code relies on some libraries that you might need to install beforehand:
```bash
sudo apt install ros-kinetic-ros-numpy
pip install --user scikit-learn
```

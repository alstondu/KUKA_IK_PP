# IK_Path_Planning
Performing Inverse Kinematics and Straight Path Planning on KUKA Youbot

------
## Introduction
### jacobian
#### [youbotKineBase.py](https://github.com/alstondu/IK_Path_Planning/blob/main/jacobian/src/jacobian/youbotKineBase.py) contains common methods needed to compute the jacobian of the Youbot and check the occurrence of singularity.
#### [youbotKine.py](https://github.com/alstondu/IK_Path_Planning/blob/main/jacobian/src/jacobian/youbotKine.py) implements manually coded functions used to compute the jacobian of the Youbot and check the occurrence of singularity.
#### [youbotKineKDL.py](https://github.com/alstondu/IK_Path_Planning/blob/main/jacobian/src/jacobian/youbotKineKDL.py) contains functions used to compute the jacobian of the Youbot and check the occurrence of singularity with the KDL library.
### path_planning
#### [path_planning.py](https://github.com/alstondu/IK_Path_Planning/blob/main/path_planning/src/path_planning.py) controlls the Youbot end-effector to reach each target Cartesian check-point defined in the bagfile [bags/data.bag](https://github.com/alstondu/IK_Path_Planning/blob/main/path_planning/bags/data.bag) , via shortest path in Cartesian space.

------
## Execution Instruction
- git clone all the packages in ROS workspace
- change to the catkin workspace directory and run:
```commandline
catkin_make
```
- Launch the file [Path_Planning.launch](https://github.com/alstondu/IK_Path_Planning/blob/main/path_planning/launch/Path_Planning.launch) to run the simulation.
```commandline
roslaunch path_planning Path_Planning.launch
```

------
## Result
The path planning result is shown in the figure below, with path shown as dot markers:
  <div align="center">
    <img width="100%" src="https://github.com/alstondu/IK_Path_Planning/blob/main/Figure/path_planning.png"></a>
  </div>

#!/usr/bin/env python3
import numpy as np
import rospy
import rosbag
import rospkg
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from jacobian.youbotKineKDL import YoubotKinematicKDL
import itertools
import PyKDL
from visualization_msgs.msg import Marker
from scipy.linalg import expm, logm, inv

class YoubotTrajectoryPlanning(object):
    def __init__(self):
        # Initialize node
        rospy.init_node('youbot_traj', anonymous=True)

        # Save question number for check in main run method
        self.kdl_youbot = YoubotKinematicKDL()
        # Create trajectory publisher and a checkpoint publisher to visualize checkpoints
        self.traj_pub = rospy.Publisher('/EffortJointInterface_trajectory_controller/command', JointTrajectory,
                                        queue_size=5)
        self.checkpoint_pub = rospy.Publisher("checkpoint_positions", Marker, queue_size=100)

    def run(self):
        """This function is the main run function of the class. When called, it runs question 6 by calling the q6()
        function to get the trajectory. Then, the message is filled out and published to the /command topic.
        """
        print("run q6a")
        rospy.loginfo("Waiting 5 seconds for everything to load up.")
        rospy.sleep(2.0)
        traj = self.q6()
        traj.header.stamp = rospy.Time.now()
        traj.joint_names = ["arm_joint_1", "arm_joint_2", "arm_joint_3", "arm_joint_4", "arm_joint_5"]
        self.traj_pub.publish(traj)

    def q6(self):
        """ Methods are called to create the shortest path. Below, a general step-by-step is given.
        Returns:
            traj (JointTrajectory): A list of JointTrajectory points giving the robot joint positions to achieve in a
            given time period.
        """
        # Steps
        # 1. Load in targets from the bagfile (checkpoint data and target joint positions).
        # 2. Compute the shortest path achievable visiting each checkpoint Cartesian position.
        # 3. Determine intermediate checkpoints to achieve a linear path between each checkpoint and have a full list of
        #    checkpoints the robot must achieve. You can publish them to see if they look correct. Look at slides 39 in lecture 7
        # 4. Convert all the checkpoints into joint values using an inverse kinematics solver.
        # 5. Create a JointTrajectory message.

        # Load the checkpoints form the rosbag
        target_cart_tf, target_joint_positions = self.load_targets()

        # Sort the order of the checkpoints for the shortest path
        sorted_order, min_dist = self.get_shortest_path(target_cart_tf)

        # Add intermediate checkpoints
        full_checkpoint_tfs = self.intermediate_tfs(sorted_order, target_cart_tf, 6)

        # Visualise the checkpoints
        self.publish_traj_tfs(full_checkpoint_tfs)

        # Convert the transformation matrix to joint states
        q_checkpoints = self.full_checkpoints_to_joints(full_checkpoint_tfs,target_joint_positions[:,0])

        # Create a trajectory message and publish to get the robot to move to this checkpoints
        traj = JointTrajectory()
        t = 0
        dt = 2
        for i in range(q_checkpoints.shape[1]):
            traj_point = JointTrajectoryPoint()
            traj_point.positions = q_checkpoints[:, i]
            t = t + dt
            traj_point.time_from_start.secs = t
            traj.points.append(traj_point)
        # Your code ends here ------------------------------

        assert isinstance(traj, JointTrajectory)
        return traj

    def load_targets(self):
        """This function loads the checkpoint data from the 'data.bag' file. In the bag file, you will find messages
        relating to the target joint positions. You need to use forward kinematics to get the goal end-effector position.
        Returns:
            target_cart_ tf (4x4x5 np.ndarray): The target 4x4 homogenous transformations of the checkpoints found in the
            bag file. There are a total of 5 transforms (4 checkpoints + 1 initial starting cartesian position).
            target_joint_positions (5x5 np.ndarray): The target joint values for the 4 checkpoints + 1 initial starting
            position.
        """
        # Defining ros package path
        rospack = rospkg.RosPack()
        path = rospack.get_path('cw2q6')

        # Initialize arrays for checkpoint transformations and joint positions
        target_joint_positions = np.zeros((5, 5))
        # Create a 4x4 transformation matrix, then stack 6 of these matrices together for each checkpoint
        target_cart_tf = np.repeat(np.identity(4), 5, axis=1).reshape((4, 4, 5))

        # Load path for selected question
        bag = rosbag.Bag(path + '/bags/data.bag')
        # Get the current starting position of the robot
        target_joint_positions[:, 0] = self.kdl_youbot.kdl_jnt_array_to_list(self.kdl_youbot.current_joint_position)
        # Initialize the first checkpoint as the current end effector position
        target_cart_tf[:, :, 0] = self.kdl_youbot.forward_kinematics(target_joint_positions[:, 0])

        # Extract the joint angles iteratively  
        for msg, i in zip(bag.read_messages(), range(target_joint_positions.shape[1]-1)):
            # Load the angles in target_joint_positions, starting from the second column
            target_joint_positions[:,i+1] = msg[1].position
            # Derive the transformation matrix of the 4 checkpoints
            target_cart_tf[:,:,i+1] = self.kdl_youbot.forward_kinematics(target_joint_positions[:,i+1], 5)

        # Close the bag
        bag.close()

        assert isinstance(target_cart_tf, np.ndarray)
        assert target_cart_tf.shape == (4, 4, 5)
        assert isinstance(target_joint_positions, np.ndarray)
        assert target_joint_positions.shape == (5, 5)

        return target_cart_tf, target_joint_positions

    def get_shortest_path(self, checkpoints_tf):
        """This function takes the checkpoint transformations and computes the order of checkpoints that results
        in the shortest overall path.
        Args:
            checkpoints_tf (np.ndarray): The target checkpoints transformations as a 4x4x5 numpy ndarray.
        Returns:
            sorted_order (np.array): An array of size 5 indicating the order of checkpoint
            min_dist:  (float): The associated distance to the sorted order giving the total estimate for travel
            distance.
        """

        checkpoints = []    # the coordinate of the checkpoints
        # Extract the translation component of each transformation, which are the coordinates
        for i in range(checkpoints_tf.shape[2]):
            checkpoints.append(checkpoints_tf[0:3, 3, i])
        checkpoints= np.array(checkpoints).T
        # The number of total points
        n = checkpoints.shape[1]
        d_matrix = np.zeros((n, n))

        # Store the distance between the jth point and the ith joint at position [i,j] of the n x n matrix
        # [d_00, d_01, d_02, d_03, d_04]
        # [d_10, d_11, d_12, d_13, d_14]
        # [d_20, d_21, d_22, d_23, d_24]
        # [d_30, d_31, d_32, d_33, d_34]
        # [d_40, d_41, d_42, d_43, d_44]
        for i in range(n):
            for j in range(n):
                d_matrix[i,j] = np.sqrt(np.sum((checkpoints[:,j] - checkpoints[:,i])**2))

        # Initialize the total distance as infinit
        min_dist = np.inf
        sorted_order = None

        # Derive the permutations of the checkpoints
        for perm in itertools.permutations(range(1, n)):
            # Add the secured starting point to the permutation
            perm = (0,) + perm
            # Calculate the total distance of each permutation
            dist = sum(d_matrix[i][j] for i, j in zip(perm[:-1], perm[1:]))
            # Compare the distance iterately to find the minimum solution
            if dist < min_dist:
                min_dist = dist
                # Use this permutation as the order
                sorted_order = perm
        sorted_order = np.array(sorted_order)

        assert isinstance(sorted_order, np.ndarray)
        assert sorted_order.shape == (5,)
        assert isinstance(min_dist, float)

        return sorted_order, min_dist

    def publish_traj_tfs(self, tfs):
        """This function gets a np.ndarray of transforms and publishes them in a color coded fashion to show how the
        Cartesian path of the robot end-effector.
        Args:
            tfs (np.ndarray): A array of 4x4xn homogenous transformations specifying the end-effector trajectory.
        """
        id = 0
        for i in range(0, tfs.shape[2]):
            marker = Marker()
            marker.id = id
            id += 1
            marker.header.frame_id = 'base_link'
            marker.header.stamp = rospy.Time.now()
            marker.type = marker.SPHERE
            marker.action = marker.ADD
            marker.scale.x = 0.01
            marker.scale.y = 0.01
            marker.scale.z = 0.01
            marker.color.a = 1.0
            marker.color.r = 0.0
            marker.color.g = 0.0 + id * 0.05
            marker.color.b = 1.0 - id * 0.05
            marker.pose.orientation.w = 1.0
            marker.pose.position.x = tfs[0, -1, i]
            marker.pose.position.y = tfs[1, -1, i]
            marker.pose.position.z = tfs[2, -1, i]
            self.checkpoint_pub.publish(marker)

    def intermediate_tfs(self, sorted_checkpoint_idx, target_checkpoint_tfs, num_points):
        """
        This function takes the target checkpoint transforms and the desired order based on the shortest path sorting, 
        and calls the decoupled_rot_and_trans() function.
        Args:
            sorted_checkpoint_idx (list): List describing order of checkpoints to follow.
            target_checkpoint_tfs (np.ndarray): the state of the robot joints. In a youbot those are revolute
            num_points (int): Number of intermediate points between checkpoints.
        Returns:
            full_checkpoint_tfs: 4x4x(4xnum_points + 5) homogeneous transformations matrices describing the full desired
            poses of the end-effector position.
        """

        # Initialization: the transformation matrix of the starting configuration
        full_checkpoint_tfs = target_checkpoint_tfs[:,:,0]

        # Between each consequtive checkpoints
        for i in range(sorted_checkpoint_idx.shape[0]-1): # 0, 1, 2, 3
            # Obtain the transformation of the intermediate points between point i and point i+1
            tfs = self.decoupled_rot_and_trans(target_checkpoint_tfs[:,:,sorted_checkpoint_idx[i]], target_checkpoint_tfs[:,:,sorted_checkpoint_idx[i+1]], num_points)
            # Stack the intermediate transformations on that of the ith point
            full_checkpoint_tfs = np.dstack((full_checkpoint_tfs, tfs))
            # Stack the transformation of point i+1
            full_checkpoint_tfs = np.dstack((full_checkpoint_tfs, target_checkpoint_tfs[:,:,sorted_checkpoint_idx[i+1]]))
       
        return full_checkpoint_tfs

    def decoupled_rot_and_trans(self, checkpoint_a_tf, checkpoint_b_tf, num_points):
        """This function takes two checkpoint transforms and computes the intermediate transformations
        that follow a straight line path by decoupling rotation and translation.
        Args:
            checkpoint_a_tf (np.ndarray): 4x4 transformation describing pose of checkpoint a.
            checkpoint_b_tf (np.ndarray): 4x4 transformation describing pose of checkpoint b.
            num_points (int): Number of intermediate points between checkpoint a and checkpoint b.
        Returns:
            tfs: 4x4x(num_points) homogeneous tr desired
            poses of the end-effector position from checkpoint a to checkpoint b following a linear path.
        """

        # Define the time parameter
        delta_t = 1/(num_points+1)

        # Extract the position of the checkpoint_a
        P_a = checkpoint_a_tf[0:3,3]
        # Extract the rotation matrix of the checkpoint_a
        R_a = checkpoint_a_tf[0:3,0:3]

        # Extract the position of the checkpoint_b
        P_b = checkpoint_b_tf[0:3,3]
        # Extract the rotation matrix of the checkpoint_b
        R_b = checkpoint_b_tf[0:3,0:3]

        # initialise the tfs matrix
        tfs = np.zeros((4,4,num_points))
        tfs[3, 3, :] = 1

        # Calculate the log term of the decoupled rotation
        log =  logm(np.matmul(inv(R_a), R_b))

        # Iterate through the intermediate points
        for i in range(1,num_points+1):
            # Decoupled translation
            P_t = P_a + i*delta_t*(P_b-P_a)
            # Decoupled rotation
            R_t = np.matmul(R_a,expm(log*i*delta_t))
            # Span to tansformation matrix
            tfs[0:3,0:3,i-1] = R_t
            tfs[0:3,3,i-1] = P_t

        return tfs

    def full_checkpoints_to_joints(self, full_checkpoint_tfs, init_joint_position):
        """This function takes the full set of checkpoint transformations, including intermediate checkpoints, 
        and computes the associated joint positions by calling the ik_position_only() function.
        Args:
            full_checkpoint_tfs (np.ndarray, 4x4xn): 4x4xn transformations describing all the desired poses of the end-effector
            to follow the desired path.
            init_joint_position (np.ndarray):A 5x1 array for the initial joint position of the robot.
        Returns:
            q_checkpoints (np.ndarray, 5xn): For each pose, the solution of the position IK to get the joint position
            for that pose.
        """
        
        n = full_checkpoint_tfs.shape[2]
        q_checkpoints = np.zeros((5, n))

        q = init_joint_position
        # Convert all of the transforamtion to q iteratively
        for i in range(n):
            q, error = self.ik_position_only(full_checkpoint_tfs[:, :, i], q)
            q_checkpoints[:, i] = q.flatten()

        return q_checkpoints

    def ik_position_only(self, pose, q0):
        """This function implements position only inverse kinematics.
        Args:
            pose (np.ndarray, 4x4): 4x4 transformations describing the pose of the end-effector position.
            q0 (np.ndarray, 5x1):A 5x1 array for the initial starting point of the algorithm.
        Returns:
            q (np.ndarray, 5x1): The IK solution for the given pose.
            error (float): The Cartesian error of the solution.
        """
        # Only interested in position control - take only the position part of the pose as well as elements of the
        # Jacobian that will affect the position of the error.

        # Extract the position part of the target pose
        x_e_star = pose[:3, 3]

        # Include a maximum number of iterations in case the algorithm does not converge
        max_iterations = 1000
        # Step size
        alpha = 0.1 

        q = q0.copy()  # Initialize joint angles

        for i in range(max_iterations):
            # Compute the current end-effector position
            x_e = self.kdl_youbot.forward_kinematics(q)[:3, 3]
            # Compute the error
            error = np.linalg.norm(x_e_star - x_e)

            # Check if the error is below the tolerance
            if error < 1e-6:
                break

            # Compute the position elements of the Jacobian matrix
            J = self.kdl_youbot.get_jacobian(q)[0:3,:]
            # Update joint angles
            q = np.array(q).reshape(5,1) + alpha * np.dot(J.T, x_e_star - x_e).reshape(5,1)

        return q, error

    @staticmethod
    def list_to_kdl_jnt_array(joints):
        """This converts a list to a KDL jnt array.
        Args:
            joints (joints): A list of the joint values.
        Returns:
            kdl_array (PyKDL.JntArray): JntArray object describing the joint position of the robot.
        """
        kdl_array = PyKDL.JntArray(5)
        for i in range(0, 5):
            kdl_array[i] = joints[i]
        return kdl_array


if __name__ == '__main__':
    try:
        youbot_planner = YoubotTrajectoryPlanning()
        youbot_planner.run()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
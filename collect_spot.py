import bosdyn.client
import json
import time
from bosdyn.client.robot_state import RobotStateClient
from bosdyn.client.image import ImageClient
from bosdyn.client.manipulation_api_client import ManipulationApiClient
from get_image import capture


IP = '138.16.161.24'
USER = 'user'
PASS = "bigbubbabigbubba"

previous_gripper_open_percentage = None

def is_moving(robot_state, threshold=0.05):
    """Determine if the robot is moving in any direction or rotating."""
    # velocity = robot_state.kinematic_state.velocity.vel
    linear_velocity = robot_state.kinematic_state.velocity_of_body_in_vision.linear
    angular_velocity = robot_state.kinematic_state.velocity_of_body_in_vision.angular

    return (abs(linear_velocity.x) > threshold or
            abs(linear_velocity.y) > threshold or
            abs(linear_velocity.z) > threshold or
            abs(angular_velocity.x) > threshold or
            abs(angular_velocity.y) > threshold or
            abs(angular_velocity.z) > threshold)

def is_arm_moving(manipulator_state, linear_threshold=0.1, angular_threshold=0.1):
    """Determine if the robot's arm is moving."""
    # Choose either 'velocity_of_hand_in_vision' or 'velocity_of_hand_in_odom' based on your requirement
    linear_velocity = manipulator_state.velocity_of_hand_in_vision.linear
    angular_velocity = manipulator_state.velocity_of_hand_in_vision.angular
    # print(angular_velocity)
    # print(linear_velocity)

    # Check if the linear or angular velocity exceeds the thresholds
    linear_moving = abs(linear_velocity.x) > linear_threshold or abs(linear_velocity.y) > linear_threshold or abs(linear_velocity.z) > linear_threshold
    angular_moving = abs(angular_velocity.x) > angular_threshold or abs(angular_velocity.y) > angular_threshold or abs(angular_velocity.z) > angular_threshold

    return linear_moving or angular_moving


def is_gripper_moving(manipulator_state, threshold=0.01):
    global previous_gripper_open_percentage
    current_percentage = manipulator_state.gripper_open_percentage

    if previous_gripper_open_percentage is None:
        previous_gripper_open_percentage = current_percentage
        return False

    if abs(current_percentage - previous_gripper_open_percentage) > threshold:
        previous_gripper_open_percentage = current_percentage
        return True

    previous_gripper_open_percentage = current_percentage
    return False



def collect_data(manipulation_client, robot, robot_state):
    """Collect and return required data."""
    front_image_data = capture(robot, "PIXEL_FORMAT_RGB_U8")
    front_depth_data = capture(robot, "PIXEL_FORMAT_DEPTH_U16")
    gripper_image_data = capture(robot, mode="arm")

    arm_state_data = None # Replace
    gripper_state_data = "gripper_state_data"  # Replace with actual gripper state capture logic

    return {
        "images": {
            "front_rgb_camera": front_image_data,
            "front_depth_camera": front_depth_data,
            "gripper_camera": gripper_image_data
        },
        "arm_state": arm_state_data,
        "gripper_state": gripper_state_data
    }

def is_robot_sitting(robot_state):
    """Determine if the robot is in a sitting position."""
    # Placeholder for demonstration
    # Replace with actual logic to determine if the robot is sitting
    # odom_position = robot_state.kinematic_state.transforms_snapshot.child_to_parent_edge_map['odom'].parent_tform_child.position
    vision_position = robot_state.kinematic_state.transforms_snapshot.child_to_parent_edge_map['vision'].parent_tform_child.position
    if vision_position.z >= 0.15:
        return True
    return False


def main():
    # Create robot object and authenticate
    sdk = bosdyn.client.create_standard_sdk('SpotRobotClient')
    robot = sdk.create_robot(IP)
    robot.authenticate(USER, PASS)


    # Create state, image, and manipulation clients
    state_client = robot.ensure_client(RobotStateClient.default_service_name)

    data_sequence = []

    while True:
        robot_state = state_client.get_robot_state()
        manipulator_state = robot_state.manipulator_state

        # if is_robot_sitting(robot_state):
        #     with open('spot_data.json', 'w') as file:
        #         json.dump(data_sequence, file, indent=4)
        #     break
       
        if is_moving(robot_state) or is_arm_moving(manipulator_state) or is_gripper_moving(manipulator_state):
            print('moving')
            # print(manipulator_state)
            # print(arm_state)
            print(robot_state)
            # collected_data = collect_data(manipulation_client, robot, robot_state)
            # data_sequence.append(collected_data)
        else:
            print('not moving')

        time.sleep(0.1)

if __name__ == '__main__':
    main()
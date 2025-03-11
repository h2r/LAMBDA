import configparser
import argparse
import os
import pickle
from tqdm import tqdm
import numpy as np
from collections import defaultdict
from queue import PriorityQueue
from PIL import Image
import clip
import torch
import time
import json
from google.protobuf import wrappers_pb2

from bosdyn.api import arm_command_pb2, robot_command_pb2, synchronized_command_pb2

from skill_utils import get_best_clip_vild_dirs

from spot_utils.utils import pixel_to_vision_frame, pixel_to_vision_frame_depth_provided, arm_object_grasp, grasp_object, open_gripper
from spot_utils.generate_pointcloud import make_pointcloud
from vild.vild_utils import visualize_boxes_and_labels_on_image_array, plot_mask
from clipseg.clipseg_utils import ClipSeg
import matplotlib.pyplot as plt
from matplotlib import patches
from torchvision import transforms

from bosdyn.api import arm_command_pb2, robot_command_pb2, synchronized_command_pb2, trajectory_pb2
from bosdyn.client import math_helpers
from bosdyn.util import seconds_to_duration

import cv2

# import open3d as o3d

from bosdyn.client.image import ImageClient
from bosdyn.client.manipulation_api_client import ManipulationApiClient
from bosdyn.client.lease import LeaseKeepAlive, LeaseClient
from bosdyn.client.recording import GraphNavRecordingServiceClient
from bosdyn.client import ResponseError, RpcError
from bosdyn.util import duration_to_seconds

import bosdyn.api.gripper_command_pb2
import bosdyn.client
import bosdyn.client.lease
import bosdyn.client.util

from bosdyn.api import geometry_pb2
from bosdyn.api import basic_command_pb2

from bosdyn.client.frame_helpers import VISION_FRAME_NAME, GRAV_ALIGNED_BODY_FRAME_NAME, get_a_tform_b, GRAV_ALIGNED_BODY_FRAME_NAME, ODOM_FRAME_NAME
from bosdyn.client.robot_command import (RobotCommandBuilder, RobotCommandClient,
                                         block_until_arm_arrives, blocking_stand)

from constrained_manipulation_helper import construct_drawer_task
import pandas as pd

from sentence_transformers import SentenceTransformer, util
import spacy

from spot_utils.move_spot_to import simple_move_to

import pandas as pd
from spot_utils.move_spot_to import move_to
from PIL import Image

from recording_command_line import RecordingInterface
from graph_nav_command_line import GraphNavInterface

import copy

def fill_action_template(program_line_args):

    action_templates_to_english = {'pick up': '{} the {}', 'put down': '{} the {} on the {}', 'walk to': '{} the {}', 'open': '{} the {}', 'close': '{} the {}', 'stand up': '{}', 'sit down': '{}', 'look at': '{} the {}' }

        
    template = action_templates_to_english[program_line_args[0]]

   
    if program_line_args[0] == 'put down':
        template = template.format(program_line_args[0], program_line_args[1], program_line_args[2])
    elif program_line_args[0] not in set(['sit down', 'stand up']):
        template = template.format(program_line_args[0], program_line_args[1])
    else:
        template = template.format(program_line_args[0])

    return template

def continue_block():
    
    select = None
    
    while select is None:
        arg = input('continue? [y/n]: ')

        if arg.strip().lower() == 'y':
            select = 'y'
        elif arg.strip().lower() == 'n':
            select = 'n'
        else: 
            print('select a valid option\n')

    return select 


class Action():
    def __init__(self, hostname, scene_num, scene_path='./scene_graphs/scene_1.json', start_location='origin', with_robot = False, dist_threshold=0.7):

        self.with_robot = with_robot

        if self.with_robot:
            self.hostname = hostname
            sdk = bosdyn.client.create_standard_sdk('ActionTester')
            self.robot = sdk.create_robot(hostname)

            try:
                bosdyn.client.util.authenticate(self.robot)
                self.robot.start_time_sync(1)
            except RpcError as err:
                LOGGER.error("Failed to communicate with robot: %s" % err)
                return False

            bosdyn.client.util.authenticate(self.robot)


            self.sources = ['hand_depth_in_hand_color_frame', 'hand_color_image']
            self.default_timeout = 20

            self.robot_id = self.robot.get_id()

        
            # Time sync is necessary so that time-based filter requests can be converted
            self.robot.time_sync.wait_for_sync()

            assert not self.robot.is_estopped(), "Robot is estopped. Please use an external E-Stop client, " \
                "such as the estop SDK example, to configure E-Stop."

            self.robot_state_client = self.robot.ensure_client(
                bosdyn.client.robot_state.RobotStateClient.default_service_name)

            self.lease_client = self.robot.ensure_client(
                LeaseClient.default_service_name)
            self.lease_keepalive = LeaseKeepAlive(
                self.lease_client, must_acquire=True, return_at_exit=True)

            self.image_client = self.robot.ensure_client(
                ImageClient.default_service_name)
            self.manipulation_api_client = self.robot.ensure_client(
                ManipulationApiClient.default_service_name)
            self.robot_command_client = self.robot.ensure_client(
                RobotCommandClient.default_service_name)

            self.robot.power_on(timeout_sec=self.default_timeout)
            assert self.robot.is_powered_on(), "Robot power on failed."


            #fields for recording a scene graph of environment
            self.client_metadata = GraphNavRecordingServiceClient.make_client_metadata(
            session_name='llm_planning', client_username=self.robot._current_user, client_id='RecordingClient',
            client_type='Python SDK')

            self.clip_model, self.clip_preprocess = clip.load('ViT-B/32')
            self.clipseg_model = ClipSeg(threshold_mask=True, threshold_value=0.55, depth_product=False)
            self.use_clipseg_binary = True
        
            # get spot to stand first
            blocking_stand(self.robot_command_client,
                        timeout_sec=self.default_timeout)

            # pointcloud parameters
            self.original_pcd = None
            self.dist_threshold = dist_threshold

            # path for pointcloud generation
            self.data_path = './data/default_data/'
            self.graph_path = './graphs'

            #os.makedirs(self.graph_path, exist_ok=True)

            self.pose_data_fname = "pose_data.pkl"
            self.embedding_data_fname = 'clip_embeddings.pkl'
            self.pointcloud_fname = "pointcloud.pcd"
            #show time taken to process image for grabbing with ViLD vs CLIPSeg
            self.show_time = False

            # Path for images
            self.img_dir_root_path = "./data/"
            self.img_dir_name = "default_data"
            self.img_dir_path = self.img_dir_root_path + self.img_dir_name

        self.similarity_model = SentenceTransformer(
            'sentence-transformers/all-roberta-large-v1')
        self.object_skills = ["pick up", "drop down",
                                 "walk to", "open", "close",]
        self.robot_skills = ["stand up", "sit down"]
        self.nlp = spacy.load("en_core_web_sm")


  

        #NOTE: used for interaction with planner
        self.use_translated = True
        self.dist_epsilon = 2.0

        self.manual_images = True
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

        with open(scene_path) as scene_graph:
            graph_scene = json.load(scene_graph) 


        #NOTE: ROBOT STATE VARIABLES USED FOR CHECKING PRECONDITON FAILURE
        self.robot_standing = graph_scene['spot_state']['standing']
        self.robot_holding = graph_scene['spot_state']['holding']
        self.robot_holding_object = graph_scene['spot_state']['holding_object']
        self.robot_location = graph_scene['spot_state']['location']

        self.robot_misaligned = graph_scene['spot_state']['misaligned']

        self.saved_waypoints = set() #track waypoints that are manually created
        
        self.object_states = self._parse_state_dictionary(copy.deepcopy(graph_scene['object_states'])) 
        self.room_state = self._parse_state_dictionary(graph_scene['room_state'])
        self.id_to_name = self._parse_id_to_name(copy.deepcopy(graph_scene['object_states']))

        #store the initial states too
        self.init_object_states =  copy.deepcopy(self.object_states)
        self.init_room_state = copy.deepcopy(self.room_state)
        self.init_spot_state = copy.deepcopy(graph_scene['spot_state'])

        self.with_pause = True
    
    def _get_curr_spot_state(self):
        return {'standing': self.robot_standing, 'holding': self.robot_holding, 'holding_object': self.robot_holding_object, 'location': self.robot_location}

    def _parse_id_to_name(self, dict_list):
        
        output_dict = {}

        for item in dict_list:
            new_key = item['id']
            new_val = item['name']
        
            output_dict[new_key] = new_val
        
        return output_dict


    def _parse_state_dictionary(self, dict_list):

        output_dict = {}

        for item in dict_list:
            new_key = item['name']
            
            item.pop('name')

            output_dict[new_key] = item
        
        return output_dict

    def _get_location_object_room(self, location_object):

        room = location_object

        all_rooms = set(self.room_state.keys())

        while room not in all_rooms:
            room = self.object_states[room]['location']

        return room
    
    def _get_pose_data_for_closest_image(self, text_prompt):
        
        #load all saved pose data into an array
        with open(self.data_path+self.pose_data_fname, 'rb') as f:
            pose_data = pickle.load(f)

        
        #normalize clip image embeddings
        with open(self.data_path+ self.embedding_data_fname, 'rb') as f:
            clip_embeddings = pickle.load(f)
        
        clip_embeddings /= clip_embeddings.norm(dim=-1, keepdim=True)


        #get clip text embeddings for the prompt
        tokenized_text_prompt = clip.tokenize([text_prompt]).to(self.device)
        with torch.no_grad():
            text_features = self.clip_model.encode_text(text_prompt)
            
        
        clip_scores = (100.0 * clip_embeddings @ text_features.T).softmax(dim=-1)
        best_clip_score, chosen_index = similarity[0].topk(1)

        #fetch pose data for that image and return along with clip score
        chosen_pose_data = pose_data[chosen_index]

        return chosen_pose_data, best_clip_score, chosen_idx

        

    def _check_precondition_error(self, skill, obj, loc, text_prompt):

        precondition_error = False
        error_info = None
        error_params = {}
        error_msg = None
        skippable = True

        #get robot current position
        # frame_tree_snapshot = self.robot.get_frame_tree_snapshot()
        # vision_tform_hand = get_a_tform_b(frame_tree_snapshot, "vision", "hand")

        # graphnav_localization_state = self.graphnav_interface._graph_nav_client.get_localization_state()
        # seed_tform_body = graphnav_localization_state.localization.seed_tform_body
        # seed_tform_body = bosdyn.client.math_helpers.SE3Pose(seed_tform_body.position.x, seed_tform_body.position.y, seed_tform_body.position.z, seed_tform_body.rotation)

        # seed_tform_hand = seed_tform_body * body_tform_hand
        
        # robot_position = [seed_tform_hand.position.x, seed_tform_hand.position.y, seed_tform_hand.position.z]

        #precondition error 4.5: not next to the object when trying to pick up, put down or look at
        # closest_pose_data, __, __ = _get_pose_data_for_closest_image(text_prompt)

        #get the natural language variant of skill, obj, loc
        lang_loc = loc.replace('_',' ')
        lang_skill = skill.replace('_',' ')
        lang_obj = obj.replace('_',' ')

        #find room within which obj is contained
        obj_room = self._get_location_object_room(obj)
        lang_obj_room = obj_room.replace('_',' ')

        #find room in which spot is currently contained
        spot_room = self._get_location_object_room(self.robot_location)

        #robot is already sitting
        if ((skill=="sit_down") and not self.robot_standing):
            precondition_error = True
            error_info = 'I am already sitting down'
            skippable = True
            error_params = {'type':'already_sitting', 'character':'character'}
        
        #robot already standing up
        elif ((skill=="stand_up") and self.robot_standing):
            precondition_error = True
            error_info = 'I am already standing up'
            skippable = True
            error_params = {'type':'already_standing', 'character':'character'}

        #object is not openable, or holdable
        elif ((skill=="close") and self.object_states[obj]['openable'] is False) or ((skill=="open") and  self.object_states[obj]['openable'] is False) or ((skill=='pick_up') and self.object_states[obj]['holdable'] is False):
            precondition_error = True
            error_info = 'cannot {} the {} '.format(skill, obj)
            skippable = False
            error_params = {'type': 'invalid_action', 'action': lang_skill, 'obj': lang_obj}
        
        #object cannot be put down since location is not a container
        elif ((skill=="put_down") and self.object_states[loc]['container'] is False):
            precondition_error = True
            error_info = 'cannot {} the {} on {}'.format(skill, obj, loc)
            skippable = False
            error_params = {'type': 'invalid_action', 'action': lang_skill, 'obj': lang_obj + ' on ' + lang_loc}

        #robot attempts to walk to something it is already holding
        elif ((skill=="walk_to") and self.robot_holding_object==obj):
            precondition_error = True
            error_info = 'I am already holding the {}'.format(obj)
            skippable = True
            error_params = {'type': 'already_holding', 'character': 'character', 'obj': lang_obj}

        #robot walks, picks up, puts down without stading up first
        elif ((skill=="walk_to" or skill=="pick_up" or skill=="put_down" or skill=="open" or skill=="close") and not self.robot_standing):
            precondition_error = True
            error_info = 'I am not standing up'
            skippable = False
            error_params = {'type':'missing_step', 'character':'character', 'subtype':'not_standing_up'}
        
        #robot attempts pick up, put down, open, close after misaligning wrt graphnav
        elif ((skill=="pick_up" or skill=="put_down" or skill=="open" or skill=="close") and self.robot_misaligned):
            precondition_error = True
            error_info = 'I am not near the {}'.format(obj) if skill!="put_down" else 'I am not near the {}'.format(loc)

            skippable = False
            error_params = {'type':'proximity', 'character':'character', 'obj': obj if skill!= "put_down" else loc}

        #robot walks though it is already near location
        elif ((skill=="walk_to") and self.robot_location == obj):
            precondition_error = True
            error_info = 'I am already near the {}'.format(obj)
            skippable = True
            error_params = {'type': 'missing_step', 'character': 'character', 'subtype': 'already near the {}'.format(lang_obj)}
        
        #robot attempts to walk through place obstructed by doors
        elif ((skill=="walk_to") and obj_room in self.room_state[spot_room]['neighbour_doors'] and (self.object_states[self.id_to_name[self.room_state[spot_room]['neighbour_doors'][obj_room]]]['open'] is False)):
            precondition_error = True
            error_info = 'The door between {} and {} is closed'.format(spot_room, obj_room)
            skippable = False
            error_params = {'type': 'door_closed', 'target_room': lang_obj_room}

        #robot attempts to pick but already holding
        elif ((skill=="pick_up") and self.robot_holding):
            precondition_error=  True
            error_info = 'I am already holding something, so cannot {} {}'.format(skill, obj)
            skippable = False
            error_params = {'type':'hands_full', 'obj':lang_obj, 'action': lang_skill}
        
        #robot attempts to pick but object inside a closed container
        elif ((skill=="pick_up") and 'contained_in' in self.object_states[obj] and self.object_states[self.id_to_name[self.object_states[obj]['contained_in']]]['openable'] and (self.object_states[self.id_to_name[self.object_states[obj]['contained_in']]]['open'] is False)):
            precondition_error = True
            error_info = 'the {} is inside something else'.format(obj)
            skippable = False
            error_params = {'type':'internally_contained', 'obj':lang_obj}

        #robot attempts to put down but does not have the object or has the wrong object
        elif ((skill=="put_down") and (self.robot_holding is False or self.robot_holding_object!=obj)):
            precondition_error = True
            error_info = 'I am not holding the {}'.format(obj)
            skippable = False
            error_params = {'type':'not_holding', 'character':'character', 'obj': lang_obj}

        #robot attempts to pick up, open or close something but is not at location
        elif ((skill=="pick_up" or skill=="open" or skill=="close") and self.robot_location!=obj):
            precondition_error = True
            error_info = 'I am not near the {}'.format(obj)
            skippable = False    
            error_params = {'type':'proximity', 'character':'character', 'obj':lang_obj} 
        

        #robot attempts to put down but is not near location
        elif ((skill=="put_down") and self.robot_location!=loc):
            precondition_error = True
            error_info = 'I am not near the {}'.format(loc)
            skippable = False    
            error_params = {'type':'proximity', 'character':'character', 'obj':lang_loc} 
        
        #robot attempts pick up, put down or look at but is too far away
        # elif ((skill=="pick_up" or skill=="put_down" or skill=="look_at") and np.sqrt(np.sum(np.squared(np.array(robot_position) - np.array(closest_pose_data['position']))))>= self.dist_epsilon):
        #     precondition_error = True
        #     error_info = 'I am not near the {}'.format(obj)
        #     skippable = False
        #     error_params = {'type':'proximity', 'character':'character', 'obj':lang_obj}
        
        #robot attempts to open or close something whilst holding an object
        elif ((skill=="open" or skill=="close") and self.robot_holding is True):
            precondition_error = True
            error_info = 'I am already holding something, so cannot {} {}'.format(skill, obj)
            skippable = False
            error_params = {'type': 'hands_full', 'obj': lang_obj, 'action': lang_skill}

        #robot opens something that is already open       
        elif ((skill=="open") and  self.object_states[obj]['openable'] and self.object_states[obj]['open'] is True):
            precondition_error = True
            error_info = 'the {} is already open'.format(obj)
            skippable = True
            error_params = {'type':'unflipped_boolean_state', 'obj':lang_obj, 'error_state': 'open'}
        
        #robot closes something that is already closed
        elif ((skill=="close") and object_states[obj]['openable'] and self.object_states[obj]['open'] is False):
            precondition_error = True
            error_info = 'the {} is already closed'.format(obj)
            skippable = True
            error_params = {'type':'unflipped_boolean_state', 'obj': lang_obj, 'error_state': 'closed'}
        
        if precondition_error:
            task_script = '{} {}'.format(skill, obj) if loc is '' else '{} {} on {}'.format(skill, obj, loc)

            error_msg = '<agent> cannot {}. '.format(task_script) + (error_info[0].upper() + error_info[1:]).replace('I am', '<agent> is')

        return precondition_error, error_params, error_msg, skippable
        
        


    def _parse_step_to_action(self, generated_step, translated_step):

        loc = ''

        if self.use_translated:
            parsed_step = translated_step.split(' ')

            if len(parsed_step) > 2:
                [skill, obj, loc] = parsed_step
                
            else:
                [skill, obj] = parsed_step

        else:
            noun_start = [c for c in self.nlp(generated_step).noun_chunks][0].text
            noun_substring = generated_step[generated_step.index(noun_start):]

            modified_skills_for_noun = [
                s + ' ' + noun_substring for s in self.reference_skills]
            modified_skills_for_noun += self.robot_skills

            # embed the reference and the input prompt
            embedded_ref = self.similarity_model.encode(
                modified_skills_for_noun, convert_to_tensor=True)
            embedded_prompt = self.similarity_model.encode(
                generated_step, convert_to_tensor=True)

            # compute the cosine similarity
            cosine_similarity = util.cos_sim(embedded_prompt, embedded_ref)

            skill_idx = torch.argmax(cosine_similarity).item()

            skill = (self.object_skills + self.robot_skills)[skill_idx]
            obj = noun_substring
        

        precondition_error, error_params, error_msg, skippable = self._check_precondition_error(skill, obj, loc, translated_step if self.use_translated else generated_step)



        return obj, skill, loc, precondition_error, error_params, error_msg, skippable


    def _start_robot_command(self, desc, command_proto, end_time_secs=None):

        def _start_command():
            self.robot_command_client.robot_command(command=command_proto,
                                                    end_time_secs=end_time_secs)

        self._try_grpc(desc, _start_command)

    def _try_grpc(self, desc, thunk):
        try:
            return thunk()
        except (ResponseError, RpcError) as err:
            self.add_message("Failed {}: {}".format(desc, err))
            return None


class WalkToAction():

    def __init__(self, hostname, action):

        self.action_interface = action

        if self.action_interface.with_robot:
            self.recording_interface = RecordingInterface(self.action_interface.robot, self.action_interface.graph_path, self.action_interface.client_metadata)

            self.graphnav_interface = GraphNavInterface(self.action_interface.robot, self.recording_interface._download_filepath)

            
            self.graphnav_interface._upload_graph_and_snapshots()
            self.graphnav_interface._set_initial_localization_fiducial()
            self.graphnav_interface._list_graph_waypoint_and_edge_ids()

            success = self.graphnav_interface._navigate_to(['origin'])

            print(success)

        
       

    def navigate(self, text_prompt):
        
        
        assert  self.action_interface.with_robot is False or self.graphnav_interface is not None, 'Error in WalkToAction: need to define graphnav interface by generate_initial_scene_graph before navigate'

        
        location, __ ,__, __, __, __, __ = self.action_interface._parse_step_to_action(None, text_prompt)

        success = False
        position = None

        if self.action_interface.with_robot:
            with LeaseKeepAlive(self.action_interface.lease_client, must_acquire=True, return_at_exit=True):
                if location in self.action_interface.saved_waypoints:
                    print('walking to waypoint {} ...'.format(location))
                    
                    if self.action_interface.with_pause:
                        out = continue_block()
                        
                        if out == 'n':
                            exit()

                    success = self.graphnav_interface._navigate_to(location)

                else:
                    #directly apply get_pose_data_for_closest_image and then navigate to returned location
                    chosen_pose_data, _, chosen_idx = _get_pose_data_for_closest_image(text_prompt)

                    print('walking to image at index {} ...'.format(chosen_idx))

                    if self.action_interface.with_pause:
                        out = continue_block()

                        if out == 'n':
                            exit()

                    position = chosen_pose_data['position']; rotation_matrix = chosen_pose_data['rotation_matrix']

                    success = self.graphnav_interface._navigate_to_anchor(position + rotation_matrix)
            
        #simple_move_to(self.robot, position, rotation_matrix)
        #move_to(self.robot, self.robot_state_client, pose=position, distance_margin=1.00, hostname=self.hostname, end_time=30)

        #update robot location after navigation
        if success or not self.action_interface.with_robot:
            self.action_interface.object_states[location]['coordinates'] = position
            self.action_interface.robot_location = location
            self.action_interface.robot_misaligned = False

        return success or not self.action_interface.with_robot
        
    
    def navigate_to_origin(self):

        success = False

        if self.action_interface.with_robot:
            with LeaseKeepAlive(self.acction_interface.lease_client, must_acquire=True, return_at_exit=True):
                success = self.graphnav_interface._navigate_to(location)

        if success or not self.action_interface.with_robot:
            self.action_interface.robot_location = 'origin'
            self.action_interface.robot_misaligned = False
        return success or not self.action_interface.with_robot

    def collect_object_poses(self):

        if not os.path.isdir(self.action_interface.data_path):
            os.mkdir(self.action_interface.data_path)

        # We only use the hand color
        camera_sources = ['hand_depth_in_hand_color_frame', 'hand_color_image']

        counter = 0
        img_to_pose_dir = {}  # takes in counter as key, and returns robot pose. saved in img_dir
        done = False

        all_embeddings = [] #stores a tensor of CLIP encoded images for all images taken in scene

        while True:
            
            if self.action_interface.manual_images:
                response = input("Take image [y/n]")
                if response == "n":
                    break
        
            else:
                time.sleep(1/float(pic_hz))

            # capture and save images to disk
            image_responses = self.action_interface.image_client.get_image_from_sources(
                camera_sources)

            # Image responses are in the same order as the requests.
            if len(image_responses) < 2:
                print('Error: failed to get images.')
                return False

            frame_tree_snapshot = self.action_interface.robot.get_frame_tree_snapshot()
            body_tform_hand = get_a_tform_b(frame_tree_snapshot, "body", "hand")

            graphnav_localization_state = self.graphnav_interface._graph_nav_client.get_localization_state()
            seed_tform_body = graphnav_localization_state.localization.seed_tform_body
            seed_tform_body = bosdyn.client.math_helpers.SE3Pose(seed_tform_body.position.x, seed_tform_body.position.y, seed_tform_body.position.z, seed_tform_body.rotation)

            seed_tform_hand = seed_tform_body * body_tform_hand

            img_to_pose_dir[counter] = {"position": [seed_tform_hand.position.x, seed_tform_hand.position.y, seed_tform_hand.position.z],
                                        "quaternion(wxyz)": [seed_tform_hand.rotation.w, seed_tform_hand.rotation.x, seed_tform_hand.rotation.y, seed_tform_hand.rotation.z],
                                        "rotation_matrix": seed_tform_hand.rotation.to_matrix(),
                                        "rpy": [seed_tform_hand.rotation.to_roll(), seed_tform_hand.rotation.to_pitch(), seed_tform_hand.rotation.to_yaw()]}

            pickle.dump(img_to_pose_dir, open(
                self.action_interface.data_path+self.action_interface.pose_data_fname, "wb"))

            robot_position = [seed_tform_hand.position.x,
                              seed_tform_hand.position.y, seed_tform_hand.position.z]

            # robot is moving around
            if np.linalg.norm(np.array(robot_position) - np.array(last_robot_position)) > 0.2:
                print("Robot is moving, no photos!")
                last_robot_position = robot_position
                already_took_photo = False
                continue
            else:
                last_robot_position = [seed_tform_hand.position.x,
                              seed_tform_hand.position.y, seed_tform_hand.position.z]

            # cv_depth is in millimeters, divide by 1000 to get it into meters
            cv_depth_meters = cv_depth / 1000.0

            # Visual is a JPEG
            cv_visual = cv2.imdecode(np.frombuffer(
                image_responses[1].shot.image.data, dtype=np.uint8), -1)

            # Convert the visual image from a single channel to RGB so we can add color
            visual_rgb = cv_visual if len(cv_visual.shape) == 3 else cv2.cvtColor(
                cv_visual, cv2.COLOR_GRAY2RGB)

            # Map depth ranges to color

            # cv2.applyColorMap() only supports 8-bit; convert from 16-bit to 8-bit and do scaling
            min_val = np.min(cv_depth)
            max_val = np.max(cv_depth)
            depth_range = max_val - min_val
            depth8 = (255.0 / depth_range *
                      (cv_depth - min_val)).astype('uint8')
            depth8_rgb = cv2.cvtColor(depth8, cv2.COLOR_GRAY2RGB)
            depth_color = cv2.applyColorMap(depth8_rgb, cv2.COLORMAP_JET)

            # Add the two images together.
            out = cv2.addWeighted(visual_rgb, 0.5, depth_color, 0.5, 0)

            cv2.imwrite(self.action_interface.data_path+"color_" +
                        str(counter)+".jpg", cv_visual)
            pickle.dump(cv_depth_meters, open(
                self.action_interface.data_path+"depth_"+str(counter), "wb"))
            cv2.imwrite(self.action_interface.data_path +
                        "combined_"+str(counter)+".jpg", out)
            

            #perform clip image embedding and save to file
            device = torch.device('cuda:0')
            
            preprocessed_image = self.action_interface.clip_preprocess(cv_visual).unsqueeze(0).to(device)

            with torch.no_grad():
                clip_embedding = self.action_interface.clip_model.encode_image(preprocessed_image).cpu()

            all_embeddings.append(clip_embedding)

            counter += 1

            if counter == 100:
                done = True

            if done:
                break
            
        #save the clip image embeddings from captured scene data
        with open(self.action_interface.data_path+self.action_interface.embedding_data_fname,'wb') as f:
            pickle.dump({'clip_embedding': torch.cat(all_embeddings, dim=0)}, f)

        print('%0d images saved' % counter)

        print('Moving back to origin ...')
        success = self.graphnav_interface._navigate_to('origin')
        self.action_interface.robot_location = 'origin'
        self.action_interface.robot_misaligned = False

        return True and success

    def generate_initial_scene_graph(self):
        
        with LeaseKeepAlive(self.action_interface.lease_client, must_acquire=True, return_at_exit=True):
            #clear out the scene graph
            self.recording_interface._clear_map()

            while True:
                print("""
                Options:
                (0) Clear map.
                (1) Start recording a map.
                (2) Stop recording a map.
                (3) Get the recording service's status.
                (4) Create a default waypoint in the current robot's location.
                (5) Create a named waypoint in the current robot's location
                (6) Download the map after recording.
                (7) List the waypoint ids and edge ids of the map on the robot.
                (8) Create new edge between existing waypoints using odometry.
                (9) Create new edge from last waypoint to first waypoint using odometry.
                (10) Automatically find and close loops.
                (a) Optimize the map's anchoring.
                (q) Exit.
                """)
                try:
                    inputs = input('>')
                except NameError:
                    pass
                req_type = inputs.split(' ')[0]

                args = None
                if len(inputs.split(' ')) > 1:
                    args = inputs.split(' ')[1:]

                if req_type == 'q':
                    print('Quitting data collection for scene graph')
                    break

                if req_type not in self.recording_interface._command_dictionary:
                    print("Request not in the known command dictionary.")
                    continue
                
                if req_type == '5' and args is None:
                    print('Add argument for the waypoint name after the command option')
                    continue

                try:
                    #add to saved waypoints if new waypoint generated
                    if req_type == '5':
                        self.action_interface.saved_waypoints.add(args[0])
                    cmd_func = self.recording_interface._command_dictionary[req_type]
                    cmd_func(args)
                except Exception as e:
                    print('Error: ', e)
                    continue
            
            #after collecting the map, download the entire graph
            self.recording_interface._download_full_graph()

            #create graphnav interface and upload the saved graph after creating map
            self.graphnav_interface = GraphNavInterface(self.robot, self.recording_interface._download_filepath)
            self.graphnav_interface._upload_graph_and_snapshots()

            print('Moving back to origin ...')
            success = self.graphnav_interface._navigate_to('origin')
            self.action_interface.robot_location = 'origin'
            self.action_interface.robot_misaligned = False

        return True and success



    def move_to_position_and_take_photo(self, topk_item, pose_dir):
        # move the position and take the photo
        # return cv file that can be stored as an image using cv2.imwrite
        # reference nlpmal.py go_to_and_pick_top_k
        best_pose = None

        img_name = topk_item[1][0]
        file_num = int(img_name.split("_")[-1].split(".")[0])
        print("processing image {file_num}".format(file_num=file_num))
        depth_img = pickle.load(
            open(self.img_dir_path+"/depth_"+str(file_num), "rb"))
        rotation_matrix = pose_dir[file_num]['rotation_matrix']
        print(rotation_matrix)
        position = pose_dir[file_num]['position']
        print(position)

        # ymin, xmin, ymax, xmax = topk_item[1][3:-1]
        # print(ymin, xmin, ymax, xmax)
        # center_y = int((ymin + ymax)/2.0)
        # center_x = int((xmin + xmax)/2.0)

        # transformed_point, bad_point = pixel_to_vision_frame(
        #     center_y, center_x, depth_img, rotation_matrix, position)
        # print("transformed point is {tp}".format(
        #     tp=transformed_point))
        # print("bad point is {bad_point}".format(bad_point))

        simple_move_to(self.robot, position=position,
                       rotation_matrix=rotation_matrix)

        open_gripper(self.robot_command_client)

        # Capture and save images to disk
        # reference get_depth_color_pose.py
        image_responses = self.image_client.get_image_from_sources(
            self.sources)

        if len(image_responses) < 2:
            print('Error: failed to get images.')
            return False
        frame_tree_snapshot = self.robot.get_frame_tree_snapshot()
        vision_tform_hand = get_a_tform_b(
            frame_tree_snapshot, "vision", "hand")
        loc_data = {"position": [vision_tform_hand.position.x, vision_tform_hand.position.y, vision_tform_hand.position.z],
                    "quaternion(wxyz)": [vision_tform_hand.rotation.w, vision_tform_hand.rotation.x, vision_tform_hand.rotation.y, vision_tform_hand.rotation.z],
                    "rotation_matrix": vision_tform_hand.rotation.to_matrix(),
                    "rpy": [vision_tform_hand.rotation.to_roll(), vision_tform_hand.rotation.to_pitch(), vision_tform_hand.rotation.to_yaw()]}

        
        # Depth is a raw bytestream
        cv_depth = np.frombuffer(
            image_responses[0].shot.image.data, dtype=np.uint16)
        cv_depth = cv_depth.reshape(image_responses[0].shot.image.rows,
                                    image_responses[0].shot.image.cols)

        # cv_depth is in millimeters, divide by 1000 to get it into meters
        cv_depth_meters = cv_depth / 1000.0

        # Visual is a JPEG
        cv_visual = cv2.imdecode(np.frombuffer(
            image_responses[1].shot.image.data, dtype=np.uint8), -1)

        #cv2.imwrite("color.jpg", cv_visual)

        # Convert the visual image from a single channel to RGB so we can add color
        visual_rgb = cv_visual if len(cv_visual.shape) == 3 else cv2.cvtColor(
            cv_visual, cv2.COLOR_GRAY2RGB)

        # Map depth ranges to color

        # cv2.applyColorMap() only supports 8-bit; convert from 16-bit to 8-bit and do scaling
        min_val = np.min(cv_depth)
        max_val = np.max(cv_depth)
        depth_range = max_val - min_val
        depth8 = (255.0 / depth_range *
                  (cv_depth - min_val)).astype('uint8')
        depth8_rgb = cv2.cvtColor(depth8, cv2.COLOR_GRAY2RGB)
        depth_color = cv2.applyColorMap(depth8_rgb, cv2.COLORMAP_JET)

        # Add the two images together.
        out = cv2.addWeighted(visual_rgb, 0.5, depth_color, 0.5, 0)

        return cv_visual, cv_depth_meters, out, loc_data

    def update_images(self, text_prompt, top_k):
        # currently, given the text prompt, use the same procedure of grabbing to
        # pick top k images and update them based on the locations.
        # print("start generating original point cloud")
        # self.original_pcd = make_pointcloud(
        #     data_path=self.data_path, pose_data_fname=self.pose_data_fname, pointcloud_fname=self.pointcloud_fname)
        print("start update images")
        pose_dir = pickle.load(
            open(self.img_dir_path+"/"+self.pose_data_fname, 'rb'))
        # TODO: add all jpg files under the path
        img_names = [f for f in os.listdir(
            "data/default_data") if f[:5] == 'color']
        print(img_names)
        priority_queue_vild_dir, priority_queue_clip_dir, _ = get_best_clip_vild_dirs(model=self.clip_model,
                                                                                      preprocess=self.clip_preprocess,
                                                                                      img_names=img_names,
                                                                                      img_dir_path=self.img_dir_path,
                                                                                      cache_images=False,
                                                                                      cache_text=False,
                                                                                      img_dir_name="",
                                                                                      category_names=[
                                                                                          text_prompt],
                                                                                      headless=True,
                                                                                      use_softmax=False)
        updated_imgs = set()
        print(priority_queue_vild_dir[text_prompt].qsize())
        while top_k > 0 and priority_queue_vild_dir[text_prompt].qsize() > 0:
            vild_item = (
                priority_queue_vild_dir[text_prompt].get())
            # clip_item = (
            #     priority_queue_clip_dir[text_prompt].get())

            vild_img_name = vild_item[1][0]
            # clip_img_name = clip_item[1][0]
            print("should process " + vild_img_name)
            print(vild_img_name not in updated_imgs)

            if vild_img_name not in updated_imgs:
                updated_imgs.add(vild_img_name)
                top_k -= 1
                file_num = int(vild_img_name.split("_")[-1].split(".")[0])
                # print(vild_img_name)
                # rotation_matrix = pose_dir[file_num]['rotation_matrix']
                # position = pose_dir[file_num]['position']
                # print(rotation_matrix)
                # print(position)
                # re-take photos
                cv_visual, cv_depth_meters, out, loc_data = self.move_to_position_and_take_photo(
                    topk_item=vild_item, pose_dir=pose_dir)
                # overwrite the original photos and depth
                # if err is not None:
                print("delete original image " + vild_img_name)
                os.remove(self.img_dir_path + "/" +vild_img_name)
                print("write the updated image " + vild_img_name)
                cv2.imwrite(self.img_dir_path + "/" + vild_img_name, cv_visual)

                os.remove(self.img_dir_path+"/depth_"+str(file_num))
                pickle.dump(cv_depth_meters, open(
                    self.img_dir_path+"/depth_"+str(file_num), "wb"))

                os.remove(self.img_dir_path+"/combined_" +
                          str(file_num)+".jpg")
                cv2.imwrite(self.img_dir_path+"/combined_" +
                            str(file_num)+".jpg", out)

                # update pose_data
                print("update pose data for file num " + str(file_num))
                pose_dir[file_num] = loc_data

            # if clip_img_name not in updated_imgs:
            #     updated_imgs.add(clip_img_name)
            #     # re-take photos
            #     cv_visual, cv_depth_meters, out = self.move_to_position_and_take_photo(
            #         topk_item=clip_item, pose_dir=pose_dir)
            #     # overwrite the original photos and depth
            #     cv2.imwrite(self.img_dir_path + "/" + clip_img_name, cv_visual)
            #     pickle.dump(cv_depth_meters, open(
            #         self.img_dir_path+"/depth_"+str(file_num), "wb"))
            #     cv2.imwrite(self.img_dir_path+"/combined_" +
            #                 str(file_num)+".jpg", out)

        # remove old pose_data
        pose_data_path = self.data_path + self.pose_data_fname
        os.remove(pose_data_path)
        # dump new pose data
        pickle.dump(pose_dir, open(
                pose_data_path, "wb"))
        # regenerate pointcloud
        print("start generating updated point cloud")
        self.original_pcd = make_pointcloud(
            data_path=self.data_path, pose_data_fname=self.pose_data_fname, pointcloud_fname="updated_pointcloud.pcd")

        # print("generate modified point cloud")
        # self.original_pcd = make_pointcloud()

        print("Finished update images")


class GrabAction():

    def __init__(self, hostname, action):
        

        self.use_vild = True
        self.use_bbox_center = False
        self.action_interface = action

    def get_center_of_mask(self, thresholded_mask):
        
        

        if type(thresholded_mask) == np.ndarray:
            thresholded_mask = torch.from_numpy(thresholded_mask)

        
        mask_coord = torch.where(thresholded_mask > 0.0)

        if len(mask_coord[0])==0 or len(mask_coord[1])==0:
            print('Error: could not generate mask!')
            return None, None

        

        mean_y = torch.mean(mask_coord[0].float())
        mean_x = torch.mean(mask_coord[1].float())

        

        distances = torch.sqrt(torch.square(
            mask_coord[0]-mean_y) + torch.square(mask_coord[1]-mean_x))

        
        min_arg = torch.argmin(distances)

        

        center_y = mask_coord[0][min_arg]
        center_x = mask_coord[1][min_arg]
        return center_y, center_x

    
    def put_in_basket(self):

        def make_robot_command(arm_joint_traj):
            """ Helper function to create a RobotCommand from an ArmJointTrajectory.
            The returned command will be a SynchronizedCommand with an ArmJointMoveCommand
            filled out to follow the passed in trajectory. """

            joint_move_command = arm_command_pb2.ArmJointMoveCommand.Request(trajectory=arm_joint_traj)
            arm_command = arm_command_pb2.ArmCommand.Request(arm_joint_move_command=joint_move_command)
            sync_arm = synchronized_command_pb2.SynchronizedCommand.Request(arm_command=arm_command)
            arm_sync_robot_cmd = robot_command_pb2.RobotCommand(synchronized_command=sync_arm)
            return RobotCommandBuilder.build_synchro_command(arm_sync_robot_cmd)
        
        sh0 = 0.0692
        sh1 = -1.5
        el0 = 1.652
        el1 = -0.0691
        wr0 = 1.622
        wr1 = 1.550

        max_vel = wrappers_pb2.DoubleValue(value=1)
        max_acc = wrappers_pb2.DoubleValue(value=5)
        traj_point = RobotCommandBuilder.create_arm_joint_trajectory_point(
            sh0, sh1, el0, el1, wr0, wr1, time_since_reference_secs=10)
        arm_joint_traj = arm_command_pb2.ArmJointTrajectory(points=[traj_point],maximum_velocity=max_vel, maximum_acceleration=max_acc)
        # Make a RobotCommand
        command = make_robot_command(arm_joint_traj)

        # Send the request
        cmd_id = self.action_interface.robot_command_client.robot_command(command)
        # self.robot.logger.info('Requesting a single point trajectory with unsatisfiable constraints.')

        # Query for feedback
        feedback_resp = self.action_interface.robot_command_client.robot_command_feedback(cmd_id)
        #self.robot.logger.info("Feedback for Example 2: planner modifies trajectory")
        time_to_goal = duration_to_seconds(feedback_resp.feedback.synchronized_feedback.arm_command_feedback.arm_joint_move_feedback.time_to_goal)
        time.sleep(time_to_goal)


        sh0 = 3.14
        sh1 = -1.3
        el0 = 1.3
        el1 = -0.0691
        wr0 = 1.622
        wr1 = 1.550

        max_vel = wrappers_pb2.DoubleValue(value=1)
        max_acc = wrappers_pb2.DoubleValue(value=5)
        traj_point = RobotCommandBuilder.create_arm_joint_trajectory_point(
            sh0, sh1, el0, el1, wr0, wr1, time_since_reference_secs=10)
        arm_joint_traj = arm_command_pb2.ArmJointTrajectory(points=[traj_point],maximum_velocity=max_vel, maximum_acceleration=max_acc)
        # Make a RobotCommand
        command = make_robot_command(arm_joint_traj)

        # Send the request
        cmd_id = self.action_interface.robot_command_client.robot_command(command)
        #self.robot.logger.info('Requesting a single point trajectory with unsatisfiable constraints.')

        # Query for feedback
        feedback_resp = self.action_interface.robot_command_client.robot_command_feedback(cmd_id)
        #self.robot.logger.info("Feedback for Example 2: planner modifies trajectory")
        time_to_goal = duration_to_seconds(feedback_resp.feedback.synchronized_feedback.arm_command_feedback.arm_joint_move_feedback.time_to_goal)
        time.sleep(time_to_goal)


        time.sleep(0.75)
        gripper_open = RobotCommandBuilder.claw_gripper_open_fraction_command(1.0)
        cmd_id = self.action_interface.robot_command_client.robot_command(gripper_open)
        time.sleep(10.0)
    
    
    
    def nlmap_grab(self, text_prompt, testing_df=None):
        print('EXECUTING GRAB ACTION')
        time1_start = time.time()

        obj, skill, loc, __, __, __ = self.action_interface._parse_step_to_action(None, text_prompt)
        text_prompt = obj

        executed = False

        if self.action_interface.with_robot:

            with LeaseKeepAlive(self.action_interface.lease_client, must_acquire=True, return_at_exit=True):

                fig, axs = plt.subplots(
                    1, 3, gridspec_kw={'width_ratios': [2, 1, 2]})


                # convert the location from the moving base frame to the world frame.
                robot_state = self.action_interface.robot_state_client.get_robot_state()
                odom_T_flat_body = get_a_tform_b(
                    robot_state.kinematic_state.transforms_snapshot, ODOM_FRAME_NAME, GRAV_ALIGNED_BODY_FRAME_NAME)

                # look at a point 1 meters in front and 0.1 meters below.
                # We are not specifying a hand location, the robot will pick one.
                gaze_target_in_odom = odom_T_flat_body.transform_point(
                    x=+1.0, y=0.0, z=+0.05)

                gaze_command = RobotCommandBuilder.arm_gaze_command(gaze_target_in_odom[0],
                                                                    gaze_target_in_odom[1],
                                                                    gaze_target_in_odom[2], ODOM_FRAME_NAME)

                gaze_command_id = self.action_interface.robot_command_client.robot_command(
                    gaze_command)

                block_until_arm_arrives(
                    self.action_interface.robot_command_client, gaze_command_id, 4.0)

                # open the SPOT gripper once spot has raised its arm + changed its gaze
                open_gripper(self.action_interface.robot_command_client)

                # Capture and save images to disk
                image_responses = self.action_interface.image_client.get_image_from_sources(
                    self.action_interface.sources)

                cv_visual = cv2.imdecode(np.frombuffer(
                    image_responses[1].shot.image.data, dtype=np.uint8), -1)

                # print('DEPTH DATA TEST: ', image_responses[0].shot.image.data)

                if not os.path.isdir('./tmp'):
                    os.mkdir('./tmp')

                # NOTE: maybe replace with direct image feed so that we don't have to save the image
                if not os.path.isdir('./saved_nlmap'):
                    os.mkdir('./saved_nlmap')
                if not os.path.isdir('./saved_nlmap/raw_images'):
                    os.mkdir('./saved_nlmap/raw_images')
                if not os.path.isdir('./saved_nlmap/segmented_images'):
                    os.mkdir('./saved_nlmap/segmented_images')
                if not os.path.isdir('./saved_nlmap/generated_masks'):
                    os.mkdir('./saved_nlmap/generated_masks')

                num = len(os.listdir('./saved_nlmap/raw_images')) + 1
                cv2.imwrite(
                    "./saved_nlmap/raw_images/raw_image{}.jpg".format(num), cv_visual)

                cv2.imwrite("./tmp/color_curview.jpg", cv_visual)

                # NOTE: left cache_path as default argument for now
                _, _, priority_queue_combined = get_best_clip_vild_dirs(self.action_interface.clip_model, self.action_interface.clip_preprocess, [
                    "color_curview.jpg"], "./tmp", cache_images=False, cache_text=False, img_dir_name="", category_names=[text_prompt], headless=True, use_softmax=False)

                top_k_items = priority_queue_combined[text_prompt].get()

                (_, (img_name, _, _, ymin, xmin, ymax, xmax,
                segmentations, max_score)) = top_k_items

                np.save('./saved_nlmap/generated_masks/mask{}'.format(num),
                        segmentations)

                load_to_df['generated_mask_file'] = 'mask{}'.format(num)

                if self.use_bbox_center:
                    ymin, xmin, ymax, xmax = top_k_items[1][3:7]

                    center_y = int((ymin + ymax)/2.0)
                    center_x = int((xmin + xmax)/2.0)

                else:
                    generated_mask = top_k_items[1][-2]

                    center_y, center_x = self.get_center_of_mask(generated_mask)

                best_pixel = (center_x, center_y)

                pixel = plt.Circle((center_x, center_y), 0.9, color='r')

                print('Best Pixel: ', (center_y, center_x))

                axs[0].imshow(cv_visual)
                axs[0].add_patch(pixel)
                axs[0].add_patch(pixel)
                axs[1].imshow(top_k_items[1][2])
                axs[2].imshow(top_k_items[1][-2])

                plt.title(text_prompt)

                plt.savefig(
                    './saved_nlmap/segmented_images/segmented_image{}.png'.format(num))

                load_to_df['color_image_file'] = "raw_image{}.jpg".format(num)

                time1_end = time.time()

                if self.action_interface.show_time:
                    print("Total Time: ", (time1_end-time1_start), " seconds")

                load_to_df['compute_time'] = time1_end - time1_start

                detected = input("Detected Object [0/1]: ")
                detected = int(detected)

                load_to_df['detected'] = detected

                if detected == 0:
                    for f in os.listdir('./tmp'):
                        os.remove(os.path.join('./tmp', f))

                    testing_df = testing_df.append(load_to_df, ignore_index=True)
                    testing_df.to_csv('./grasping_test.csv', index=False)
                    exit()
                else:
                    testing_df = testing_df.append(load_to_df, ignore_index=True)
                    testing_df.to_csv('./grasping_test.csv', index=False)

                execute = input("Execute grasp")

                success = grasp_object(
                    self.action_interface.robot_state_client, self.action_interface.manipulation_api_client, best_pixel, image_responses[1])

                # if object grasped successfully, carry object
                if success:
                    carry_cmd = RobotCommandBuilder.arm_carry_command()
                    self.action_interface.robot_command_client.robot_command(carry_cmd)
                    time.sleep(0.75)
                    # TODO: figure out basket stuff later
                    # self.put_in_basket()
                else:
                    # Recovery after grasping
                    open_gripper(self.action_interface.robot_command_client)

                executed = success

                human_review = input("Executed[0/1]: ")
                executed = int(executed or bool(human_review))

                load_to_df['executed'] = executed

                print("Stowing Arm Away...")
                stow = RobotCommandBuilder.arm_stow_command()
                stow_command_id = self.action_interface.robot_command_client.robot_command(stow)
                block_until_arm_arrives(
                    self.action_interface.robot_command_client, stow_command_id, self.action_interface.default_timeout)
                print("Arm Stowed Away!")

                move_back_proto = RobotCommandBuilder.synchro_velocity_command(
                    v_x=-1.0, v_y=0.0, v_rot=0.0)

                self.action_interface._start_robot_command(
                    'move_backward', move_back_proto, end_time_secs=time.time() + 2.0)

                print(os.listdir("./tmp"))
                # clear out the temporary files stored from ViLD
                for f in os.listdir('./tmp'):
                    os.remove(os.path.join('./tmp', f))

                # replace last entry with the new load_to_df dict
                testing_df.drop(testing_df.tail(1).index, inplace=True)
                testing_df = testing_df.append(load_to_df, ignore_index=True)
                testing_df.to_csv('./grasping_test.csv', index=False)

        if executed or not self.action_interface.with_robot:
            self.action_interface.robot_holding = True
            self.action_interface.robot_holding_object = obj
            # self.action_interface.robot_location = None
            self.action_interface.robot_misaligned = True
            self.action_interface.object_states[obj]['held'] = True
            self.action_interface.object_states[obj]['location'] = 'agent'

            if 'contained_in' in self.action_interface.object_states[obj] and self.action_interface.object_states[obj]['contained_in'] in self.action_interface.id_to_name:
                self.action_interface.object_states[self.action_interface.id_to_name[self.action_interface.object_states[obj]['contained_in']]]['contains'].remove(self.action_interface.object_states[obj]['id'])
            
            self.action_interface.object_states[obj]['contained_in'] = None
        
        return executed or not self.action_interface.with_robot

    def clipseg_grab(self, text_prompt, testing_df=None):

        print('Executing grab action for {}'.format(text_prompt))
        time1_start = time.time()

        if self.action_interface.with_robot:
            # with LeaseKeepAlive(self.lease_client, must_acquire=True, return_at_exit=True):

            fig, axs = plt.subplots(1, 3, gridspec_kw={'width_ratios': [2, 1, 2]})
        
            # convert the location from the moving base frame to the world frame.
            robot_state = self.action_interface.robot_state_client.get_robot_state()
            odom_T_flat_body = get_a_tform_b(
                robot_state.kinematic_state.transforms_snapshot, ODOM_FRAME_NAME, GRAV_ALIGNED_BODY_FRAME_NAME)

            # look at a point 1 meters in front and 0.1 meters below.
            # We are not specifying a hand location, the robot will pick one.
            gaze_target_in_odom = odom_T_flat_body.transform_point(
                x=+1.0, y=0.0, z=+0.05)

            gaze_command = RobotCommandBuilder.arm_gaze_command(gaze_target_in_odom[0],
                                                                gaze_target_in_odom[1],
                                                                gaze_target_in_odom[2], ODOM_FRAME_NAME)

            gaze_command_id = self.action_interface.robot_command_client.robot_command(
                gaze_command)

            block_until_arm_arrives(
                self.action_interface.robot_command_client, gaze_command_id, 4.0)

            # open the SPOT gripper once spot has raised its arm + changed its gaze
            open_gripper(self.action_interface.robot_command_client)

            # Capture and save images to disk
            image_responses = self.action_interface.image_client.get_image_from_sources(
                self.action_interface.sources)

            time.sleep(10.0)


            cv_visual = cv2.imdecode(np.frombuffer(
                image_responses[1].shot.image.data, dtype=np.uint8), -1)

            cv_depth = np.frombuffer(image_responses[0].shot.image.data, dtype=np.uint16)
            cv_depth = cv_depth.reshape(image_responses[0].shot.image.rows, image_responses[0].shot.image.cols)

            time.sleep(5.0)

            print('Running ClipSeg model...')
            # make prediction on image
            generated_mask, binary_mask = self.action_interface.clipseg_model.segment_image(cv_visual, cv_depth, text_prompt)
            
            

            if not os.path.isdir('./saved_clipseg'):
                os.mkdir('./saved_clipseg')
            if not os.path.isdir('./saved_clipseg/raw_images'):
                os.mkdir('./saved_clipseg/raw_images')
            if not os.path.isdir('./saved_clipseg/segmented_images'):
                os.mkdir('./saved_clipseg/segmented_images')
            
            if not os.path.isdir('./saved_clipseg/binary_masks'):
                os.mkdir('./saved_clipseg/binary_masks')
            if not os.path.isdir('./saved_clipseg/clipseg_masks'):
                os.mkdir('./saved_clipseg/clipseg_masks')


            num = len(os.listdir('./saved_clipseg/raw_images')) + 1

            cv2.imwrite(
                "./saved_clipseg/raw_images/raw_image{}.jpg".format(num), cv_visual)

            

            cv2.imwrite("./tmp/color_curview.jpg", cv_visual)

            
            if self.action_interface.use_clipseg_binary:
                center_y, center_x = self.get_center_of_mask(binary_mask)
            else:
                center_index = torch.argmax(generated_mask).item()
                center_y = center_index//generated_mask.shape[0]
                center_x = center_index%(generated_mask.shape[0])

            
            np.save('./saved_clipseg/binary_masks/mask{}'.format(num),
                    binary_mask)
            np.save('./saved_clipseg/clipseg_masks/mask{}'.format(num),
                    generated_mask)
            # load_to_df['clipseg_mask_file'] = 'mask{}'.format(num)
            # load_to_df['binary_mask_file'] = 'mask{}'.format(num)


            best_pixel = (center_x, center_y)

            if center_x is not None  and center_y is not  None:
                pixel = plt.Circle((center_x, center_y), 0.9, color='r')

            print('Best Pixel: ', (center_y, center_x))

            axs[0].imshow(cv_visual)
            if center_x is not None  and center_y is not  None:
                axs[0].add_patch(pixel)
            axs[1].imshow(generated_mask)
            if center_x is not None  or center_y is not  None:
                axs[2].imshow(binary_mask)

            plt.title(text_prompt)

            plt.savefig('./saved_clipseg/segmented_images/segmented_image{}.png'.format(num))

            # load_to_df['color_image_file'] = "raw_image{}.jpg".format(num)

            time1_end = time.time()

            if self.action_interface.show_time:
                print("Total Time: ", (time1_end-time1_start), " seconds")

            # load_to_df['compute_time'] = time1_end - time1_start

            detected = input("Detected Object [0/1]: ")
            detected = int(detected)

            # load_to_df['detected'] = detected

            if detected == 0:
                print('Did not detect correct target object')
                for f in os.listdir('./tmp'):
                    os.remove(os.path.join('./tmp', f))

                # testing_df = testing_df.append(load_to_df, ignore_index=True)
                # testing_df.to_csv('./grasping_test.csv', index=False)
                exit()
            # else:
            
            # testing_df = testing_df.append(load_to_df, ignore_index=True)
            # testing_df.to_csv('./grasping_test.csv', index=False)

            if self.action_interface.with_pause:
                out = continue_block()

                if out == 'n':
                    exit()

            print('Grasping target object ...')
            success = grasp_object(
                self.action_interface.robot_state_client, self.action_interface.manipulation_api_client, best_pixel, image_responses[1])

            # if object grasped successfully, carry object
            if success:
                carry_cmd = RobotCommandBuilder.arm_carry_command()
                self.action_interface.robot_command_client.robot_command(carry_cmd)
                time.sleep(0.75)
                # self.put_in_basket()
            else:
                # Recovery after grasping
                open_gripper(self.action_interface.robot_command_client)


            human_review = input("Executed[0/1]: ")
            executed = int(human_review)

            # load_to_df['executed'] = executed

            print("Stowing Arm Away...")
            stow = RobotCommandBuilder.arm_stow_command()
            stow_command_id = self.action_interface.robot_command_client.robot_command(stow)
            block_until_arm_arrives(
                self.action_interface.robot_command_client, stow_command_id, self.action_interface.default_timeout)
        

            #move_back_proto = RobotCommandBuilder.synchro_velocity_command(
            #   v_x=-1.0, v_y=0.0, v_rot=0.0)

            #self.action_interface._start_robot_command(
            #    'move_backward', move_back_proto, end_time_secs=time.time() + 2.0)

            
            # clear out the temporary files stored from ViLD
            for f in os.listdir('./tmp'):
                os.remove(os.path.join('./tmp', f))

            # replace last entry with the new load_to_df dict
            # testing_df.drop(testing_df.tail(1).index, inplace=True)
            # testing_df = testing_df.append(load_to_df, ignore_index=True)
            # testing_df.to_csv('./grasping_test.csv', index=False)

        executed = False
        if executed or not self.action_interface.with_robot:
            self.action_interface.robot_holding = True
            self.action_interface.robot_holding_object = obj
            # self.action_interface.robot_location = None
            self.action_interface.robot_misaligned = True
            self.action_interface.object_states[obj]['held'] = True
            self.action_interface.object_states[obj]['location'] = 'agent'

            if 'contained_in' in self.action_interface.object_states[obj] and self.action_interface.object_states[obj]['contained_in'] in self.action_interface.id_to_name:
                self.action_interface.object_states[self.action_interface.id_to_name[self.action_interface.object_states[obj]['contained_in']]]['contains'].remove(self.action_interface.object_states[obj]['id'])
            
            self.action_interface.object_states[obj]['contained_in'] = None
        
        return executed or not self.action_interface.with_robot

class StandSitAction():

    def __init__(self, hostname, action):
        self.action_interface = action
    

    def stand_up(self, text_prompt=None):
        
        try:
            
            if self.action_interface.with_robot:

                # with LeaseKeepAlive(self.action_interface.lease_client, must_acquire=False, return_at_exit=True):

                blocking_stand(self.action_interface.robot_command_client, timeout_sec=self.action_interface.default_timeout)
            self.action_interface.robot_standing = True
        except:
            return False

        return True
    
    def sit_down(self, text_prompt=None):

        try:
            if self.action_interface.with_robot:
                # with LeaseKeepAlive(self.action_interface.lease_client, must_acquire=False, return_at_exit=True):

                self.action_interface.robot_command_client.robot_command(command=RobotCommandBuilder.synchro_sit_command(), end_time_secs=self.action_interface.default_timeout)
            self.action_interface.robot_standing = False
        except:
            return False

        return True


class OpenCloseAction():

    def __init__(self, hostname, action):
        self.action_interface = action
    

    def get_center_of_mask(self, thresholded_mask):
        
        

        if type(thresholded_mask) == np.ndarray:
            thresholded_mask = torch.from_numpy(thresholded_mask)

        
        mask_coord = torch.where(thresholded_mask > 0.0)

        if len(mask_coord[0])==0 or len(mask_coord[1])==0:
            print('Error: could not generate mask!')
            return None, None

        

        mean_y = torch.mean(mask_coord[0].float())
        mean_x = torch.mean(mask_coord[1].float())

        

        distances = torch.sqrt(torch.square(
            mask_coord[0]-mean_y) + torch.square(mask_coord[1]-mean_x))

        
        min_arg = torch.argmin(distances)

        

        center_y = mask_coord[0][min_arg]
        center_x = mask_coord[1][min_arg]
        return center_y, center_x

    

    def open(self, text_prompt):

        with LeaseKeepAlive(self.action_interface.lease_client, must_acquire=True, return_at_exit=True):

            obj, skill, _, __, __,__, __ = self.action_interface._parse_step_to_action(None, text_prompt)
            executed = False

            if obj == 'fridge':
                obj = 'blue handle'
            elif obj == 'door':
                obj = 'door handle'

            fig, axs = plt.subplots(1, 3, gridspec_kw={'width_ratios': [2, 1, 2]})

            robot_state = self.action_interface.robot_state_client.get_robot_state()
            odom_T_flat_body = get_a_tform_b(
                robot_state.kinematic_state.transforms_snapshot, ODOM_FRAME_NAME, GRAV_ALIGNED_BODY_FRAME_NAME)

            # look at a point 1 meters in front and 0.1 meters below.
            # We are not specifying a hand location, the robot will pick one.
            gaze_target_in_odom = odom_T_flat_body.transform_point(
                x=+1.0, y=0.0, z=+0.05)

            gaze_command = RobotCommandBuilder.arm_gaze_command(gaze_target_in_odom[0],
                                                                gaze_target_in_odom[1],
                                                                gaze_target_in_odom[2], ODOM_FRAME_NAME)

            gaze_command_id = self.action_interface.robot_command_client.robot_command(
                gaze_command)

            block_until_arm_arrives(
                self.action_interface.robot_command_client, gaze_command_id, 4.0)

            # open the SPOT gripper once spot has raised its arm + changed its gaze
            open_gripper(self.action_interface.robot_command_client)

            # Capture and save images to disk
            image_responses = self.action_interface.image_client.get_image_from_sources(
                self.action_interface.sources)

            time.sleep(10.0)


            cv_visual = cv2.imdecode(np.frombuffer(
                image_responses[1].shot.image.data, dtype=np.uint8), -1)

            cv_depth = np.frombuffer(image_responses[0].shot.image.data, dtype=np.uint16)
            cv_depth = cv_depth.reshape(image_responses[0].shot.image.rows, image_responses[0].shot.image.cols)

            time.sleep(5.0)

            print('Running ClipSeg model...')
            # make prediction on image
            generated_mask, binary_mask = self.action_interface.clipseg_model.segment_image(cv_visual, cv_depth, obj)
            
            if not os.path.isdir('./saved_clipseg/segmented_images_open'):
                os.mkdir('./saved_clipseg/segmented_images_open')
                

            cv2.imwrite("./tmp/color_curview.jpg", cv_visual)

            center_y, center_x = self.get_center_of_mask(binary_mask)

            
            best_pixel = (center_x, center_y)

            if center_x is not None  and center_y is not  None:
                pixel = plt.Circle((center_x, center_y), 0.9, color='r')

            print('Best Pixel: ', (center_y, center_x))

            axs[0].imshow(cv_visual)
            if center_x is not None  and center_y is not  None:
                axs[0].add_patch(pixel)
            axs[1].imshow(generated_mask)
            if center_x is not None  or center_y is not  None:
                axs[2].imshow(binary_mask)

            plt.title(text_prompt)

            plt.savefig('./saved_clipseg/segmented_images/segmented_image{}.png'.format(num))

            detected = input("Detected Object [0/1]: ")
            detected = int(detected)

            if detected == 0:
                print('Did not detect correct target object')
                exit()
            
            if self.action_interface.with_pause:
                out = continue_block()

                if out == 'n':
                    exit()

            print('Grasping target object ...')
            success = grasp_object(self.action_interface.robot_state_client, self.action_interface.manipulation_api_client, best_pixel, image_responses[1])

            if success:
                construct_drawer_task(velocity_normalized = -0.5, force_limit=80)

            if executed or not self.action_interface.with_robot:
                # self.action_interface.robot_location = None
                self.action_interface.object_states[obj]['open'] = True
                self.action_interface.robot_misaligned = True
        
        return executed or not self.action_interface.with_robot


    
    def close(self, text_prompt):
        obj, skill, _, __, __,__, __ = self.action_interface._parse_step_to_action(None, text_prompt)
        executed = False

        # construct_drawer_task()
        
        if executed or not self.action_interface.with_robot:
            # self.action_interface.robot_location = None
            self.action_interface.object_states[obj]['open'] = True
            self.action_interface.robot_misaligned = True
        
        return executed or not self.action_interface.with_robot


class PutDownAction():

    def __init__(self, hostname, action):
        self.action_interface = action

    def put_down(self, text_prompt=None):
        
        # obj, skill, loc, __, __,__, __ = self.action_interface._parse_step_to_action(None, text_prompt)

        executed = True

        if self.action_interface.with_robot:

            # with LeaseKeepAlive(self.action_interface.lease_client, must_acquire=True, return_at_exit=True):
                
            try:
                x1 = 0.75  # a reasonable position in front of the robot
                x2 = 1.0
                x3 = 1.2
                y1 = 0  # centered
                z1 = 0.2  # at the body's height
                z2 = 0.0 

                # Use the same rotation as the robot's body.
                rotation = math_helpers.Quat()
                rotation_roll = math_helpers.Quat.from_roll(3.14/2.0)
                print(f"rotation roll {rotation_roll}")
                print(f"euler version: {math_helpers.quat_to_eulerZYX(rotation_roll)}")
                print(f"original roll {rotation}")
                print(f"euler version: {math_helpers.quat_to_eulerZYX(rotation)}")

                # Define times (in seconds) for each point in the trajectory.
                t_first_point = 0  # first point starts at t = 0 for the trajectory.
                t_second_point = 5.0
                t_third_point = 7.0

                # Build the points in the trajectory.
                hand_pose1 = math_helpers.SE3Pose(x=x1, y=y1, z=z1, rot=rotation)
                hand_pose2 = math_helpers.SE3Pose(x=x2, y=y1, z=z1, rot=rotation)
                hand_pose3 = math_helpers.SE3Pose(x=x3, y=y1, z=z2, rot=rotation_roll)

                # Build the points by combining the pose and times into protos.
                traj_point1 = trajectory_pb2.SE3TrajectoryPoint(
                    pose=hand_pose1.to_proto(), time_since_reference=seconds_to_duration(t_first_point))
                traj_point2 = trajectory_pb2.SE3TrajectoryPoint(
                    pose=hand_pose2.to_proto(), time_since_reference=seconds_to_duration(t_second_point))
                traj_point3 = trajectory_pb2.SE3TrajectoryPoint(
                    pose=hand_pose3.to_proto(), time_since_reference=seconds_to_duration(t_third_point))

                # Build the trajectory proto by combining the points.
                hand_traj = trajectory_pb2.SE3Trajectory(points=[traj_point1, traj_point2, traj_point3])

                # Build the command by taking the trajectory and specifying the frame it is expressed
                # in.
                #
                # In this case, we want to specify the trajectory in the body's frame, so we set the
                # root frame name to the flat body frame.
                arm_cartesian_command = arm_command_pb2.ArmCartesianCommand.Request(
                    pose_trajectory_in_task=hand_traj, root_frame_name=GRAV_ALIGNED_BODY_FRAME_NAME)

                # Pack everything up in protos.
                arm_command = arm_command_pb2.ArmCommand.Request(
                    arm_cartesian_command=arm_cartesian_command)

                synchronized_command = synchronized_command_pb2.SynchronizedCommand.Request(
                    arm_command=arm_command)

                robot_command = robot_command_pb2.RobotCommand(synchronized_command=synchronized_command)

                # Keep the gripper closed the whole time.
                robot_command = RobotCommandBuilder.claw_gripper_open_fraction_command(0.0, build_on_command=robot_command)

                x1 = 0.75  # a reasonable position in front of the robot
                x2 = 1.0
                x3 = 1.2
                y1 = 0  # centered
                z = 0.2  # at the body's height

                # Use the same rotation as the robot's

                # Send the trajectory to the robot.
                cmd_id = self.action_interface.robot_command_client.robot_command(robot_command)

                # Wait until the arm arrives at the goal.
                while True:
                    feedback_resp = self.action_interface.robot_command_client.robot_command_feedback(cmd_id)
                    self.action_interface.robot.logger.info('Distance to final point: ' + '{:.2f} meters'.format(
                        feedback_resp.feedback.synchronized_feedback.arm_command_feedback.
                        arm_cartesian_feedback.measured_pos_distance_to_goal) + ', {:.2f} radians'.format(
                            feedback_resp.feedback.synchronized_feedback.arm_command_feedback.
                            arm_cartesian_feedback.measured_rot_distance_to_goal))

                    if feedback_resp.feedback.synchronized_feedback.arm_command_feedback.arm_cartesian_feedback.status == arm_command_pb2.ArmCartesianCommand.Feedback.STATUS_TRAJECTORY_COMPLETE:
                        self.action_interface.robot.logger.info('Move complete.')
                        break
                    time.sleep(0.1)

                ###### ROBOT OPENS GRIPPER
                traj_point4 = trajectory_pb2.SE3TrajectoryPoint(
                    pose=hand_pose3.to_proto(), time_since_reference=seconds_to_duration(0))
                traj_point5 = trajectory_pb2.SE3TrajectoryPoint(
                    pose=hand_pose3.to_proto(), time_since_reference=seconds_to_duration(1.0))
                traj_point6 = trajectory_pb2.SE3TrajectoryPoint(
                    pose=hand_pose1.to_proto(), time_since_reference=seconds_to_duration(5.0))

                # Build the trajectory proto by combining the points.
                hand_traj = trajectory_pb2.SE3Trajectory(points=[traj_point4, traj_point5, traj_point6])

            
                # In this case, we want to specify the trajectory in the body's frame, so we set the
                # root frame name to the flat body frame.
                arm_cartesian_command = arm_command_pb2.ArmCartesianCommand.Request(
                    pose_trajectory_in_task=hand_traj, root_frame_name=GRAV_ALIGNED_BODY_FRAME_NAME)

                # Pack everything up in protos.
                arm_command = arm_command_pb2.ArmCommand.Request(
                    arm_cartesian_command=arm_cartesian_command)

                synchronized_command = synchronized_command_pb2.SynchronizedCommand.Request(
                    arm_command=arm_command)

                robot_command = robot_command_pb2.RobotCommand(synchronized_command=synchronized_command)

                # Keep the gripper open the whole time.
                robot_command = RobotCommandBuilder.claw_gripper_open_fraction_command(
                    1.0, build_on_command=robot_command)

                # Send the trajectory to the robot.
                cmd_id = self.action_interface.robot_command_client.robot_command(robot_command)

                # Wait until the arm arrives at the goal.
                while True:
                    feedback_resp = self.action_interface.robot_command_client.robot_command_feedback(cmd_id)
                    self.action_interface.robot.logger.info('Distance to final point: ' + '{:.2f} meters'.format(
                        feedback_resp.feedback.synchronized_feedback.arm_command_feedback.
                        arm_cartesian_feedback.measured_pos_distance_to_goal) + ', {:.2f} radians'.format(
                            feedback_resp.feedback.synchronized_feedback.arm_command_feedback.
                            arm_cartesian_feedback.measured_rot_distance_to_goal))

                    if feedback_resp.feedback.synchronized_feedback.arm_command_feedback.arm_cartesian_feedback.status == arm_command_pb2.ArmCartesianCommand.Feedback.STATUS_TRAJECTORY_COMPLETE:
                        self.action_interface.robot.logger.info('Move complete.')
                        break
                    time.sleep(0.1)

                ### STOW ARM
                stow = RobotCommandBuilder.arm_stow_command()
                stow_command_id = self.action_interface.robot_command_client.robot_command(stow)
                block_until_arm_arrives(self.action_interface.robot_command_client, stow_command_id,10.0)
            
            except:
                executed = False

        executed = False
        if executed or not self.action_interface.with_robot:
            self.action_interface.robot_holding = False
            self.action_interface.robot_holding_object = None
            self.action_interface.robot_location = loc
            self.action_interface.robot_misaligned = True
            self.action_interface.object_states[obj]['held'] = False
            self.action_interface.object_states[obj]['location'] = loc

            if 'contained_in' in self.action_interface.object_states[obj]:
                self.action_interface.object_states[obj]['contained_in'] = self.action_interface.object_states[loc]['id']
            
            
            self.action_interface.object_states[loc]['contains'].append(self.action_interface.object_states[obj]['id'])
            
            
        
        return executed or not self.action_interface.with_robot

# if __name__ == '__main__':

#     global testing_df
#     if not os.path.isfile('./grasping_test.csv'):

#         testing_df = pd.DataFrame(columns=['phrase', 'trial', 'ablation', 'grounded_verb', 'cosine_sim', 'parsed_noun',
#                                   'detected', 'mask_iou', 'clipseg_mask_file', 'binary_mask_file', 'color_image_file', 'executed', 'compute_time'])
#     else:
#         testing_df = pd.read_csv('./grasping_test.csv')

#     load_to_df = {c: None for c in ['phrase', 'trial', 'ablation', 'grounded_verb', 'cosine_sim', 'parsed_noun',
#                                     'detected', 'mask_iou', 'clipseg_mask_file', 'binary_mask_file', 'color_image_file', 'executed', 'compute_time']}

#     grab_action_executor = GrabAction("tusker")


#     TEXT_PROMPT = input('Text Prompt: ')

#     skill_idx, cosine_sim, parsed_noun = grab_action_executor.find_closest_action(
#         TEXT_PROMPT)

#     load_to_df['cosine_sim'] = cosine_sim
#     load_to_df['parsed_noun'] = parsed_noun

#     print('Parsed Noun: ', parsed_noun)
#     print('Grounded Verb: ', grab_action_executor.reference_skills[skill_idx])

#     trial = int(input('Trial Number: '))
#     ABLATION = input('Ablation Type: ')

#     load_to_df['phrase'] = TEXT_PROMPT
#     load_to_df['trial'] = trial
#     load_to_df['ablation'] = ABLATION

#     load_to_df['grounded_verb'] = grab_action_executor.reference_skills[skill_idx]
#     load_to_df['parsed_noun'] = parsed_noun

#     grab_action_executor.clipseg_grab(parsed_noun, testing_df)





if __name__ == '__main__':

    # import pdb
    # pdb.set_trace()
    # action = Action('138.16.161.22', None, with_robot=True)


    # walk_executor = WalkToAction('138.16.161.22', action)
    # standsit_executor = StandSitAction('138.16.161.22', action)
    # grab_executor = GrabAction('138.16.161.22', action)
    # putdown_executor = PutDownAction('138.16.161.22', action)

    # standsit_executor.sit_down()
    # standsit_executor.stand_up()
    # walk_executor.graphnav_interface._navigate_to(['shoe_rack'])
    # grab_executor.clipseg_grab('shoes')
    # walk_executor.graphnav_interface._navigate_to(['door'])
    # putdown_executor.put_down()
    # walk_executor.graphnav_interface._navigate_to(['office_table'])
    # grab_executor.clipseg_grab('black cap')
    # walk_executor.graphnav_interface._navigate_to(['office_table'])
    # putdown_executor.put_down()
    # walk_executor.graphnav_interface._navigate_to(['office_table'])
    # grab_executor.clipseg_grab('energy drink')
    # walk_executor.graphnav_interface._navigate_to(['door'])
    # putdown_executor.put_down()
    # standsit_executor.sit_down()
    pass
    
    
    
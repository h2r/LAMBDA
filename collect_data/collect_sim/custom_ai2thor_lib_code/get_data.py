import transformations as tft
from math import degrees
from datetime import datetime
import logging
import numpy as np
import json
import zipfile
from io import BytesIO
from numpy import linalg
import openai

num = 0

class get_data():
    def __init__(self):
        self.data = [] #each point is a step
        self.rgb = []
        self.depth = []
        self.inst_seg = []
        self.scene = ''
        self.target_obj = None
        self.target_obj_init_pos = None
        self.receptacle = None
        self.goal_pos = None
        self.init_pos = None
        self.release_state_ee = None
        self.base_start_to_goal_dist = None
        self.prev_global_state_body = []
        self.all_delta_global_state_body = []
        self.all_delta_global_state_ee = []
        self.all_delta_relative_state_ee = []
        self.prev_global_state_ee = []
        self.prev_relative_state_ee = []
        self.prev_hand_sphere_center = 10000000
        self.prev_held_objs = []
        self.prev_held_objs_state = {}
        self.prev_head_pitch = None
        with open("traj_obj_dic.json", "r") as json_file:
    	    self.traj_obj_dic = json.load(json_file)
        with open("cmd_id_dic.json", "r") as json_file:
    	    self.cmd_id_dic = json.load(json_file)
        with open("obj_pos_dic.json", "r") as json_file:
    	    self.obj_pos_dic = json.load(json_file)
        with open("scene_objs.json", "r") as json_file:
    	    self.scene_objs = json.load(json_file)
        with open("cmd_recep_dic.json", "r") as json_file:
    	    self.cmd_recep_dic = json.load(json_file)

    def _convert_to_euler(self,rot_quat):

        # Extract quaternion from the dictionary
        quaternion = [rot_quat['x'], rot_quat['y'], rot_quat['z'], rot_quat['w']]

        # Convert quaternion to Euler angles
        roll, pitch, yaw = tft.euler_from_quaternion(quaternion)

        roll = degrees(roll)
        pitch = degrees(pitch)
        yaw = degrees(yaw)

        return (roll, pitch, yaw)

    def _get_objs_pos(self, held_objs, all_objs):
        pos_rot_dic = {}
        for held_obj_id in held_objs: #usually 1 held object
            for obj_dic in all_objs:
                if obj_dic['objectId'] == held_obj_id:
                    pos_rot_dic[held_obj_id] = {'position': obj_dic['position'],  'rotation':obj_dic['rotation']}
        return pos_rot_dic

    def _get_target_obj_pos(self, all_objs, obj_id):
        pos = []
        dic = {}
        for obj_dic in all_objs:
            # if obj_dic['assetId'] == '':
            #     continue
            # if obj_dic['assetId'] not in dic.keys():
            #     dic[obj_dic['assetId']] = 1
            #     print("hi")
            # else:
            #     breakpoint()
            #     dic[obj_dic['assetId']] += 1
            if obj_dic['assetId'] == obj_id:
                pos = list(obj_dic['position'].values())
        if not pos:
            breakpoint() #if this triggers during data collection, the assetId doesn't exist
        return pos

    def gather_data(self, event, command):
        self.scene = event.metadata['sceneName']
        self.target_obj = self.traj_obj_dic[self.cmd_id_dic[command]]
        self.target_obj_init_pos = list(self.obj_pos_dic[self.cmd_id_dic[command]].values())
        self.receptacle = self.cmd_recep_dic[command]
        if self.receptacle == "NONE":
            breakpoint()
        global num

        dic = {}
        dic['sim_time'] = event.metadata['currentTime']
        dic['wall-clock_time'] = datetime.utcnow().strftime('%H:%M:%S.%f')[:-3]
        dic['action'] = event.metadata['lastAction']            
        dic['head_pitch'] = event.metadata["agent"]['cameraHorizon']
        global_coord_agent = event.metadata['agent']['position']
        yaw_agent =  event.metadata['agent']['rotation']['y'] #degrees
        dic['global_state_body'] = [global_coord_agent['x'], global_coord_agent['y'], global_coord_agent['z'], yaw_agent]
        global_coord_ee = event.metadata["arm"]["joints"][3]['position']
        relative_coord_ee = event.metadata["arm"]["joints"][3]['rootRelativePosition']
        rot_ee = self._convert_to_euler(event.metadata["arm"]["joints"][3]['rotation'])
        dic['global_state_ee'] = [global_coord_ee['x'], global_coord_ee['y'],global_coord_ee['z'], rot_ee[0], rot_ee[1], rot_ee[2]]
        dic['relative_state_ee'] = [relative_coord_ee['x'], relative_coord_ee['y'], relative_coord_ee['z']]
        dic['hand_sphere_radius'] = event.metadata['arm']['handSphereRadius']
        dic['hand_sphere_center'] = event.metadata['arm']['handSphereCenter']
        dic['held_objs'] = event.metadata['arm']['heldObjects']
        dic['curr_target_obj_pos'] = self._get_target_obj_pos(event.metadata["objects"], self.target_obj)
        dic['curr_ee_to_target_obj_dist'] = linalg.norm(np.array(dic['curr_target_obj_pos']) - np.array(dic['global_state_ee'][:3]))
        dic['held_objs_state'] =  self._get_objs_pos(event.metadata['arm']['heldObjects'], event.metadata["objects"])
        if dic['action'] == "ReleaseObject":
            dic['reward'] = 1 # sparse binary reward when reaching target location
            self.release_state_ee = dic['global_state_ee'][:3]
        else:
            dic['reward'] = 0 # sparse binary reward at all other steps
        dic['inst_det2D'] = {'keys': list(event.instance_detections2D.keys()), 'values': list(event.instance_detections2D.values())}
        dic['rgb'] = f'./rgb_{num}.npy'
        dic['depth'] = f'./depth_{num}.npy'
        dic['inst_seg'] = f'./inst_seg_{num}.npy'
        if dic['action'] != "Initialize" and self.init_pos is None:
            self.init_pos = dic['global_state_body'][:3]
        if dic['action'] == "ReleaseObject":
            self.goal_pos = dic['global_state_body'][:3]
        if self.goal_pos is not None and self.init_pos is not None:
            self.base_start_to_goal_dist = linalg.norm(np.array(self.goal_pos) - np.array(self.init_pos))
        num+=1

        if self.prev_global_state_body == dic['global_state_body'] and self.prev_global_state_ee == dic['global_state_ee'] and self.prev_hand_sphere_center == dic['hand_sphere_center'] and self.prev_held_objs == dic['held_objs'] and self.prev_held_objs_state == dic['held_objs_state'] and dic['head_pitch'] == self.prev_head_pitch:
            return
        
        if self.prev_global_state_body:
            self.all_delta_global_state_body.append(np.array(dic['global_state_body']) - np.array(self.prev_global_state_body))
        if self.prev_global_state_ee:
            self.all_delta_global_state_ee.append(np.array(dic['global_state_ee']) - np.array(self.prev_global_state_ee))
        if self.prev_relative_state_ee:
            self.all_delta_relative_state_ee.append(np.array(dic['relative_state_ee']) - np.array(self.prev_relative_state_ee))

        self.prev_global_state_body = dic['global_state_body']
        self.prev_global_state_ee = dic['global_state_ee']
        self.prev_relative_state_ee = dic['relative_state_ee']
        self.prev_hand_sphere_center = dic['hand_sphere_center']
        self.prev_held_objs = dic['held_objs']
        self.prev_held_objs_state = dic['held_objs_state']
        self.prev_head_pitch = dic['head_pitch']

        print(dic['action'])
        self.data.append(dic)
        self.rgb.append(event.frame)
        self.depth.append(event.depth_frame)
        self.inst_seg.append(event.instance_segmentation_frame)


    def _chunk_dict(self, data, chunk_size):
        """Yield successive chunk_size chunks from the dictionary."""
        
        # Construct the base dictionary without the 'hi' key and its associated list
        list_data = data['steps']
        
        # Getting all keys if not equal to steps
        base_keys = [k for k in data.keys() if k != 'steps']

        # Getting all keys if not equal to steps
        base_keys = [k for k in data.keys() if k != 'steps']

        # Iterate over the list in chunks
        for i in range(0, len(list_data), chunk_size):         
            # Should be more memory efficient since we dont make a separate copy of base_dict
            chunked_dict = {k: data[k] for k in base_keys} 
            chunked_dict['steps'] = list_data[i:i+chunk_size]
            yield chunked_dict

    def _add_feats(self):
        """Adds the distance from the base to the goal, and the state deltas"""
        for i, step in enumerate(self.data):
            step['curr_base_to_goal_dist'] = linalg.norm(np.array(self.goal_pos) - np.array(step['global_state_body'][:3]))
            step['curr_ee_to_release_dist'] = linalg.norm(np.array(self.release_state_ee) - np.array(step['global_state_ee'][:3]))
            if i != len(self.data)-1:
                step['delta_global_state_body'] = self.all_delta_global_state_body[i].tolist()
                step['delta_global_state_ee'] = self.all_delta_global_state_ee[i].tolist()
                step['delta_relative_state_ee'] = self.all_delta_relative_state_ee[i].tolist()
            else:
                step['delta_global_state_body'] = None
                step['delta_global_state_ee'] = None
                step['delta_relative_state_ee'] = None

    def save(self,command):
        self._add_feats()

        final_dic = {"nl_command": command, "scene":self.scene, "target_obj": self.target_obj, "target_obj_start_pos": self.target_obj_init_pos, "target_receptacle": self.receptacle, "base_start_pos": self.init_pos, "goal_pos": self.goal_pos, 'base_start_to_goal_dist':self.base_start_to_goal_dist, "steps":self.data}

        print("\nSaving...\n")
        CHUNK_SIZE=1
        now = datetime.now()
        current_time = now.strftime("%H:%M:%S")
        # Write the JSON data chunks directly into a zip file
        with zipfile.ZipFile(f'data_{current_time}.zip', 'w', zipfile.ZIP_DEFLATED) as zipf:
            for idx, chunk in enumerate(self._chunk_dict(final_dic, CHUNK_SIZE)):
                folder_name = f'folder_{idx}/'
                json_filename = f'{folder_name}data_chunk_{idx}.json'
                rgb_filename = f'{folder_name}rgb_{idx}.npy'
                depth_filename = f'{folder_name}depth_{idx}.npy'
                inst_seg_filename = f'{folder_name}inst_seg_{idx}.npy'


                with zipf.open(json_filename, 'w') as json_file:
                    # Convert the chunk to a JSON string and encode it to bytes
                    json_data = json.dumps(chunk).encode('utf-8')
                    json_file.write(json_data)

                # Define a helper function to save numpy data
                def save_npy_to_zip(data, filename):
                    npy_data = BytesIO()
                    np.save(npy_data, data, allow_pickle=True, fix_imports=True)
                    npy_data.seek(0)  # Rewind the buffer
                    with zipf.open(filename, 'w') as npy_file:
                        npy_file.write(npy_data.getvalue())
    
                save_npy_to_zip(self.rgb[idx], rgb_filename)
                save_npy_to_zip(self.depth[idx], depth_filename)
                save_npy_to_zip(self.inst_seg[idx], inst_seg_filename)

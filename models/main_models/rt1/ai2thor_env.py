
import copy
import numpy as np
from collections import Counter, OrderedDict
import ai2thor
from ai2thor.controller import Controller
from json import load
from os import path
import gen.constants as constants
# import gen.utils.image_util as image_util
# from gen.utils import game_util
# from gen.utils.game_util import get_objects_of_type, get_obj_of_type_closest_to_obj
from random import choice, randint
from time import sleep
import pdb

DEFAULT_RENDER_SETTINGS = {'renderImage': True,
                           'renderDepthImage': True,
                           'renderClassImage': False,
                           'renderObjectImage': False,
                           }

class ThorEnv():
    def __init__(self, task, max_episode_length = 1500):

        self.controller = None
        self.last_event = None

        self.task = task
        self.max_episode_length = max_episode_length
        

    def reset(self, scene_name):
        '''
        reset scene / start scene
        '''
        print('Starting Ai2Thor Env...')
        self.controller = Controller(
            agentMode="arm",
            massThreshold=None,
            scene=scene_name,
            visibilityDistance=1.5,
            gridSize=0.25,
            renderDepthImage=False,
            renderInstanceSegmentation=False,
            snapToGrid=False,
            width=300,
            height=300,
            fieldOfView=60
        )
        self.last_event = self.controller.last_event
        return self.last_event


    def step(self, action, kwargs):

        if action in set(['MoveAgent','RotateAgent']):

            if action == 'MoveAgent':
            
                event_move = self.controller.step(
                    action="Teleport",
                    position=dict(x=kwargs['xyz_body'][0], y=kwargs['xyz_body'][1], z=kwargs['xyz_body'][2])
                )

                #execute a rotation body operation
                event_rotate = self.controller.step(
                    action="RotateAgent",
                    degrees=kwargs['body_yaw_delta'],
                    returnToStart=False,
                    speed=1,
                    fixedDeltaTime=0.02
                )

                success = event_move.metadata['lastActionSuccess']
                error = [event_move.metadata['errorMessage']]
                self.last_event = event_move


            elif action == 'RotateAgent':

                #execute a rotation body operation
                event_rotate = self.controller.step(
                    action="RotateAgent",
                    degrees=kwargs['body_yaw_delta'],
                    returnToStart=False,
                    speed=1,
                    fixedDeltaTime=0.02
                )

                event_move = self.controller.step(
                    action="Teleport",
                    position=dict(x=kwargs['xyz_body'][0], y=kwargs['xyz_body'][1], z=kwargs['xyz_body'][2])
                )

                success = event_rotate.metadata['lastActionSuccess']
                error = [event_rotate.metadata['errorMessage']]
                self.last_event = event_rotate

         

        elif action == 'MoveArm':

            #execute smooth move arm operation
            event = self.controller.step(
                action="MoveArm",
                position=dict(x=kwargs['arm_position'][0], y=kwargs['arm_position'][1], z=kwargs['arm_position'][2]),
                coordinateSpace="world",
                restrictMovement=False,
                speed=1,
                returnToStart=False,
                fixedDeltaTime=0.02
            )
            
            success = event.metadata['lastActionSuccess']
            error = [event.metadata['errorMessage']]
            self.last_event = event

        elif action == 'PickupObject':

            #execute pickup
            event = self.controller.step(
                action="PickupObject",
                objectIdCandidates=[]
            )

            success = event.metadata['lastActionSuccess']
            error = [event.metadata['errorMessage']]
            self.last_event = event
        
        elif action == 'ReleaseObject':

            #execute pickup
            event = self.controller.step(
                action="ReleaseObject",
                objectIdCandidates=[]
            )

            success = event.metadata['lastActionSuccess']
            error = [event.metadata['errorMessage']]
            self.last_event = event
            
        elif action in set(['LookDown','LookUp']):

            #execute smooth change in pitch
            events = self.smooth_look(action)

            success = events[-1].metadata['lastActionSuccess'] if len(events)>0 else False
            error = [events[-1].metadata['errorMessage']] if len(events)>0 else ['Reached boundary of LookUp/LookDown']
            self.last_event = events[-1] if len(events)>0 else self.last_event

            
        elif action == 'stop':
            #stop the execution
            event = self.controller.step(action="Done")

            success = event.metadata['lastActionSuccess']
            error = [event.metadata['errorMessage']]
            self.last_event = event

        elif action == None:
            #no operation to be done
            success = True
            error = ['']

        else:

            raise Exception('Error: the provided action {} is not valid'.format(action))

        return success, error, self.last_event

    def step_old(self, action, smooth_nav=False):
        '''
        overrides ai2thor.controller.Controller.step() for smooth navigation and goal_condition updates
        '''
        if 'action' in action:
            if smooth_nav:
                if "MoveAhead" in action['action']:
                    self.smooth_move_ahead(action)
                elif "Rotate" in action['action']:
                    self.smooth_rotate(action)
                elif "Look" in action['action']:
                    self.smooth_look(action)
                else:
                    super().step(action)
            else:
                if "LookUp" in action['action']:
                    self.look_angle(-constants.AGENT_HORIZON_ADJ)
                elif "LookDown" in action['action']:
                    self.look_angle(constants.AGENT_HORIZON_ADJ)
                else:
                    super().step(action)
        else:
            super().step(action)

        event = self.update_states(action)
        self.check_post_conditions(action)
        return event



    def noop(self):
        '''
        do nothing
        '''
        super().step(dict(action='Pass'))

    def smooth_move_ahead(self, action, render_settings=None):
        '''
        smoother MoveAhead
        '''
        if render_settings is None:
            render_settings = DEFAULT_RENDER_SETTINGS
        smoothing_factor = constants.RECORD_SMOOTHING_FACTOR
        new_action = copy.deepcopy(action)
        new_action['moveMagnitude'] = constants.AGENT_STEP_SIZE / smoothing_factor

        new_action['renderImage'] = render_settings['renderImage']
        new_action['renderClassImage'] = render_settings['renderClassImage']
        new_action['renderObjectImage'] = render_settings['renderObjectImage']
        new_action['renderDepthImage'] = render_settings['renderDepthImage']

        events = []
        for xx in range(smoothing_factor - 1):
            event = super().step(new_action)
            if event.metadata['lastActionSuccess']:
                events.append(event)

        event = super().step(new_action)
        if event.metadata['lastActionSuccess']:
            events.append(event)
        return events

    def smooth_rotate(self, action, render_settings=None):
        '''
        smoother RotateLeft and RotateRight
        '''
        if render_settings is None:
            render_settings = DEFAULT_RENDER_SETTINGS
        event = self.last_event
        horizon = np.round(event.metadata['agent']['cameraHorizon'], 4)
        position = event.metadata['agent']['position']
        rotation = event.metadata['agent']['rotation']
        start_rotation = rotation['y']
        if action['action'] == 'RotateLeft':
            end_rotation = (start_rotation - 90)
        else:
            end_rotation = (start_rotation + 90)

        events = []
        for xx in np.arange(.1, 1.0001, .1):
            if xx < 1:
                teleport_action = {
                    'action': 'TeleportFull',
                    'rotation': np.round(start_rotation * (1 - xx) + end_rotation * xx, 3),
                    'x': position['x'],
                    'z': position['z'],
                    'y': position['y'],
                    'horizon': horizon,
                    'tempRenderChange': True,
                    'renderNormalsImage': False,
                    'renderImage': render_settings['renderImage'],
                    'renderClassImage': render_settings['renderClassImage'],
                    'renderObjectImage': render_settings['renderObjectImage'],
                    'renderDepthImage': render_settings['renderDepthImage'],
                }
                event = super().step(teleport_action)
            else:
                teleport_action = {
                    'action': 'TeleportFull',
                    'rotation': np.round(start_rotation * (1 - xx) + end_rotation * xx, 3),
                    'x': position['x'],
                    'z': position['z'],
                    'y': position['y'],
                    'horizon': horizon,
                }
                event = super().step(teleport_action)

            if event.metadata['lastActionSuccess']:
                events.append(event)
        return events

    def smooth_look(self, action, render_settings=None):
        '''
        smoother LookUp and LookDown
        '''
        if render_settings is None:
            render_settings = DEFAULT_RENDER_SETTINGS
        event = self.last_event
        start_horizon = event.metadata['agent']['cameraHorizon']
        rotation = np.round(event.metadata['agent']['rotation']['y'], 4)
        end_horizon = start_horizon + constants.AGENT_HORIZON_ADJ * (1 - 2 * int(action == 'LookUp'))
        position = event.metadata['agent']['position']

        events = []
        for xx in np.arange(.1, 1.0001, .1):
            if xx < 1:
                teleport_action = {
                    'action': 'TeleportFull',
                    'rotation': rotation,
                    'x': position['x'],
                    'z': position['z'],
                    'y': position['y'],
                    'horizon': np.round(start_horizon * (1 - xx) + end_horizon * xx, 3),
                    'tempRenderChange': True,
                    'renderNormalsImage': False,
                    'renderImage': render_settings['renderImage'],
                    'renderClassImage': render_settings['renderClassImage'],
                    'renderObjectImage': render_settings['renderObjectImage'],
                    'renderDepthImage': render_settings['renderDepthImage'],
                    'standing': True,
                }
                event = self.controller.step(teleport_action)
            else:
                teleport_action = {
                    'action': 'TeleportFull',
                    'rotation': rotation,
                    'x': position['x'],
                    'z': position['z'],
                    'y': position['y'],
                    'horizon': np.round(start_horizon * (1 - xx) + end_horizon * xx, 3),
                    'standing':True,
                }
                event = self.controller.step(teleport_action)

            if event.metadata['lastActionSuccess']:
                events.append(event)
        
        return events

    def rotate_angle(self, angle, render_settings=None):
        '''
        rotate at a specific angle
        '''
        if render_settings is None:
            render_settings = DEFAULT_RENDER_SETTINGS
        event = self.last_event
        horizon = np.round(event.metadata['agent']['cameraHorizon'], 4)
        position = event.metadata['agent']['position']
        rotation = event.metadata['agent']['rotation']
        start_rotation = rotation['y']
        end_rotation = start_rotation + angle

        teleport_action = {
            'action': 'TeleportFull',
            'rotation': np.round(end_rotation, 3),
            'x': position['x'],
            'z': position['z'],
            'y': position['y'],
            'horizon': horizon,
            'tempRenderChange': True,
            'renderNormalsImage': False,
            'renderImage': render_settings['renderImage'],
            'renderClassImage': render_settings['renderClassImage'],
            'renderObjectImage': render_settings['renderObjectImage'],
            'renderDepthImage': render_settings['renderDepthImage'],
        }
        event = super().step(teleport_action)
        return event

    def to_thor_api_exec(self, action, object_id="", smooth_nav=False):
        # TODO: parametrized navigation commands

        if "RotateLeft" in action:
            action = dict(action="RotateLeft",
                          forceAction=True)
            event = self.step(action, smooth_nav=smooth_nav)
        elif "RotateRight" in action:
            action = dict(action="RotateRight",
                          forceAction=True)
            event = self.step(action, smooth_nav=smooth_nav)
        elif "MoveAhead" in action:
            action = dict(action="MoveAhead",
                          forceAction=True)
            event = self.step(action, smooth_nav=smooth_nav)
        elif "LookUp" in action:
            action = dict(action="LookUp",
                          forceAction=True)
            event = self.step(action, smooth_nav=smooth_nav)
        elif "LookDown" in action:
            action = dict(action="LookDown",
                          forceAction=True)
            event = self.step(action, smooth_nav=smooth_nav)
        elif "OpenObject" in action:
            action = dict(action="OpenObject",
                          objectId=object_id,
                          moveMagnitude=1.0)
            event = self.step(action)
        elif "CloseObject" in action:
            action = dict(action="CloseObject",
                          objectId=object_id,
                          forceAction=True)
            event = self.step(action)
        elif "PickupObject" in action:
            action = dict(action="PickupObject",
                          objectId=object_id)
            event = self.step(action)
        elif "PutObject" in action:
            inventory_object_id = self.last_event.metadata['inventoryObjects'][0]['objectId']
            action = dict(action="PutObject",
                          objectId=object_id,
                          forceAction=True,
                          placeStationary=True)
            event = self.step(action)
        elif "ToggleObjectOn" in action:
            action = dict(action="ToggleObjectOn",
                          objectId=object_id)
            event = self.step(action)

        elif "ToggleObjectOff" in action:
            action = dict(action="ToggleObjectOff",
                          objectId=object_id)
            event = self.step(action)
        elif "SliceObject" in action:
            # check if agent is holding knife in hand
            inventory_objects = self.last_event.metadata['inventoryObjects']
            if len(inventory_objects) == 0 or 'Knife' not in inventory_objects[0]['objectType']:
                raise Exception("Agent should be holding a knife before slicing.")

            action = dict(action="SliceObject",
                          objectId=object_id)
            event = self.step(action)
        else:
            raise Exception("Invalid action. Conversion to THOR API failed! (action='" + str(action) + "')")

        return event, action

    def take_action(self, word_action, num_action, rand_agent=False):
        i = 0
        incr = 0.025
        x = 0
        y = 0
        z = 0
        fixedDeltaTime = 0.02
        move = 0.2
        a = None

        if rand_agent:
            all_word_actions = ['PickupObject','ReleaseObject', 'LookUp', 'LookDown', 'MoveArm', 'MoveArmBase', 'RotateAgent', 'MoveAhead', 'MoveBack', 'MoveRight', 'MoveLeft', 'stop']
            rand_word_action = choice(all_word_actions)
            if rand_word_action in ["stop"]:
                return "stop", None
            elif rand_word_action in ['PickupObject','ReleaseObject', 'LookUp', 'LookDown']:
                a = dict(action = rand_word_action)
            elif rand_word_action in ['MoveArm', 'MoveArmBase']:
                global_coord_ee = self.last_event.metadata["arm"]["joints"][3]['position']
                curr_x, curr_y, curr_z = global_coord_ee['x'], global_coord_ee['y'], global_coord_ee['z']
                rand_x_indx, rand_y_indx, rand_z_indx = randint(1, 256), randint(1, 256), randint(1, 256) # starts at 1 to skip NoOp
                x_del, y_del, z_del = self.bins["4"][rand_x_indx], self.bins["5"][rand_y_indx], self.bins["6"][rand_z_indx]
                new_x, new_y, new_z = curr_x + x_del, curr_y + y_del, curr_z + z_del
                a = dict(action='MoveArm', position=dict(x=new_x, y=new_y, z=new_z),coordinateSpace="world",restrictMovement=False,speed=1,returnToStart=False,fixedDeltaTime=fixedDeltaTime)
            elif rand_word_action in ['RotateAgent']:
                rand_yaw_indx = randint(1, 256)
                new_yaw = self.bins["3"][rand_yaw_indx]
                a = dict(action=rand_word_action, degrees=new_yaw, returnToStart=False,speed=1,fixedDeltaTime=fixedDeltaTime)
            else: # move base
                a = dict(action=rand_word_action, moveMagnitude=move, returnToStart=False,speed=1,fixedDeltaTime=fixedDeltaTime)

        else:
            if word_action in ['NoOp']:
                print(f"Word Action: NoOP", end="\r") # for debugging
                return None, None, self.last_event.metadata
            if word_action in ['PickupObject','ReleaseObject', 'LookUp', 'LookDown']:
                a = dict(action = word_action)
            elif word_action in ['MoveArm', 'MoveArmBase']:
                global_coord_ee = self.last_event.metadata["arm"]["joints"][3]['position']
                curr_x, curr_y, curr_z = global_coord_ee['x'], global_coord_ee['y'], global_coord_ee['z']
                x_del, y_del, z_del = self.bins["4"][num_action[0]], self.bins["5"][num_action[1]], self.bins["6"][num_action[2]]
                if x_del == -1000 or y_del == -1000 or z_del == -1000: # if any of them are NoOp then skip all. Can do it another way where only skip the specific axis
                    print(f"Word Action: NoOP", end="\r") # for debugging
                    return None, None, self.last_event.metadata
                new_x, new_y, new_z = curr_x + x_del, curr_y + y_del, curr_z + z_del
                a = dict(action='MoveArm',position=dict(x=new_x, y=new_z, z=new_y),coordinateSpace="world",restrictMovement=False,speed=1,returnToStart=False,fixedDeltaTime=fixedDeltaTime)
            elif word_action in ['RotateAgent']:
                yaw_del = num_action.item()
                new_yaw = self.bins["3"][yaw_del]
                if new_yaw == -1000: #make it variable later
                    print(f"Word Action: NoOP", end="\r") # for debugging
                    return None, None, self.last_event.metadata
                a = dict(action=word_action, degrees=new_yaw,returnToStart=False,speed=1,fixedDeltaTime=fixedDeltaTime)
            else: # move base
                a = dict(action=word_action, moveMagnitude=move, returnToStart=False,speed=1,fixedDeltaTime=fixedDeltaTime)
        
        sleep(0.5) #for debugging/movement analysis
        event = self.controller.step(a)
        success = event.metadata['lastActionSuccess']
        error = event.metadata['errorMessage']
        self.last_event = event
        #for debugging/movement analysis
        sleep(0.5)
        if rand_agent:
            print(f"Random Word Action: {rand_word_action} ", end="\r")
        # else:
            # print(f"Word Action: {word_action} ", end="\r")
            # print(f"Num Action: {num_action} ", end="\r")

        return success, error, self.last_event.metadata




if __name__ == '__main__':

    SCENE_NAME = 'FloorPlan_Train5_1'
    TESTED_STEPS = 200

    test = ThorEnv('Walk to the living room')

    event = test.reset(scene_name=SCENE_NAME)

    curr_body_coordinate = np.array(list(event.metadata['agent']['position'].values()))
    curr_body_yaw = event.metadata['agent']['rotation']['y']
    curr_arm_coordinate = np.array(list(event.metadata['arm']['handSphereCenter'].values()))
    agent_holding = np.array([])


    test.controller.step(
        action="MoveArmBase",
        y=0.0,
        speed=1,
        returnToStart=True,
        fixedDeltaTime=0.02
    )
        
    
    for i in range(TESTED_STEPS):

        print('''
        (1) Move X+
        (2) Move X-
        (3) Move Z+
        (4) Move Z-
        (5) Rotate Left
        (6) Rotate Right
        (7) Rotate Up
        (8) Rotate Down
        (9) Open Gripper
        (0) Close Gripper
        (h) Move Gripper up
        (n) Move Gripper down
        (b) Move Gripper left
        (m) Move Gripper right
        (z) Move Gripper forward
        (x) Move Gripper backwards
        ''')


        action = input('>')
        

        if action == '1':

            temp_body_coordinate = copy.copy(curr_body_coordinate)
            temp_body_coordinate[0] += 0.05

            success, error, event = test.step('MoveAgent', {'xyz_body': temp_body_coordinate, 'body_yaw_delta': 0})
            
        elif action == '2':

            temp_body_coordinate = copy.copy(curr_body_coordinate)
            temp_body_coordinate[0] -= 0.05

            success, error, event = test.step('MoveAgent', {'xyz_body': temp_body_coordinate, 'body_yaw_delta': 0})

        elif action == '3':

            temp_body_coordinate = copy.copy(curr_body_coordinate)
            temp_body_coordinate[2] += 0.05

            success, error, event = test.step('MoveAgent', {'xyz_body': temp_body_coordinate, 'body_yaw_delta': 0})
        
        elif action == '4':

            temp_body_coordinate = copy.copy(curr_body_coordinate)
            temp_body_coordinate[2] -= 0.05

            success, error, event = test.step('MoveAgent', {'xyz_body': temp_body_coordinate, 'body_yaw_delta': 0})
        
        elif action == '5':

            success, error, event = test.step('RotateAgent', {'xyz_body': curr_body_coordinate, 'body_yaw_delta': -90})
        
        elif action == '6':

            success, error, event = test.step('RotateAgent', {'xyz_body': curr_body_coordinate, 'body_yaw_delta':+90})

        elif action == '7':

            success, error, event = test.step('LookUp', {})
        
        elif action == '8':

            success, error, event = test.step('LookDown', {})
        
        elif action == '9':

            success, error, event = test.step('PickupObject', {})

        elif action == '0':

            success, error, event = test.step('ReleaseObject', {})

        elif action == 'h':
            temp_arm_coordinate = copy.copy(curr_arm_coordinate)
            temp_arm_coordinate[1] += 0.20

            success, error, event = test.step('MoveArm', {'arm_position': temp_arm_coordinate})
        
        elif action == 'n':
            temp_arm_coordinate = copy.copy(curr_arm_coordinate)
            temp_arm_coordinate[1] -= 0.05

            success, error, event = test.step('MoveArm', {'arm_position': temp_arm_coordinate})

        elif action == 'b':
            temp_arm_coordinate = copy.copy(curr_arm_coordinate)
            temp_arm_coordinate[0] += 0.05

            success, error, event = test.step('MoveArm', {'arm_position': temp_arm_coordinate})

        elif action == 'm':
            temp_arm_coordinate = copy.copy(curr_arm_coordinate)
            temp_arm_coordinate[0] -= 0.05

            success, error, event = test.step('MoveArm', {'arm_position': temp_arm_coordinate})

        elif action == 'z':
            temp_arm_coordinate = copy.copy(curr_arm_coordinate)
            temp_arm_coordinate[2] += 0.05

            success, error, event = test.step('MoveArm', {'arm_position': temp_arm_coordinate})

        elif action == 'x':

            temp_arm_coordinate = copy.copy(curr_arm_coordinate)
            temp_arm_coordinate[2] -= 0.05

            success, error, event = test.step('MoveArm', {'arm_position': temp_arm_coordinate})

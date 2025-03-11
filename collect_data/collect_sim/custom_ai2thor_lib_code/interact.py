import sys
import numpy as np
import os

from enum import Enum
import json

from ai2thor.get_data import get_data #added by ahmed
get_data_obj = get_data() #added by ahmed


class DefaultActions(Enum):
    MoveRight = (0,)
    MoveLeft = (1,)
    MoveAhead = (2,)
    MoveBack = (3,)
    LookUp = (4,)
    LookDown = (5,)
    RotateRight = (8,)
    RotateLeft = 9


# TODO tie this with actions
# class ObjectActions(Enum):
#     PutObject
#     MoveHandAhead
#     MoveHandBack
#     MoveHandRight
#     MoveHandLeft
#     MoveHandUp
#     MoveHandDown
#     DropHandObject
#     PickupObject,
#     OpenObject,
#     CloseObject,
#     ToggleObjectOff

k=0
def get_term_character():
    # NOTE: Leave these imports here! They are incompatible with Windows.
    import tty
    import termios

    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setraw(sys.stdin.fileno())
        ch = sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
    return ch

i = 0
incr = 0.025
x = 0
y = 0
z = 0
fixedDeltaTime = 0.02
move = 0.2
class InteractiveControllerPrompt(object):
    def __init__(
        self,
        default_actions,
        has_object_actions=True,
        image_dir=".",
        image_per_frame=False,
    ):
        self.default_actions = default_actions
        self.has_object_actions = has_object_actions
        self.image_per_frame = image_per_frame
        self.image_dir = image_dir
        self.counter = 0

        default_interact_commands = {
            "d": dict(action="MoveRight", moveMagnitude=move, returnToStart=False,speed=1,fixedDeltaTime=fixedDeltaTime), #changed by ahmed
            "a": dict(action="MoveLeft", moveMagnitude=move, returnToStart=False,speed=1,fixedDeltaTime=fixedDeltaTime),#changed by ahmed
            "w": dict(action="MoveAhead", moveMagnitude=move, returnToStart=False,speed=1,fixedDeltaTime=fixedDeltaTime),#changed by ahmed
            "s": dict(action="MoveBack", moveMagnitude=move, returnToStart=False,speed=1,fixedDeltaTime=fixedDeltaTime),#changed by ahmed
            # "\x1b[1;2A": dict(action="LookUp"), #commented out by ahmed
            # "\x1b[1;2B": dict(action="LookDown"), #commented out by ahmed
            "i": dict(action="LookUp"),
            "k": dict(action="LookDown"),
            "l": dict(action="RotateAgent", degrees=20,returnToStart=False,speed=1,fixedDeltaTime=fixedDeltaTime), #changed by ahmed
            "j": dict(action="RotateAgent", degrees=-20,returnToStart=False,speed=1,fixedDeltaTime=fixedDeltaTime), #changed by ahmed
            # "\x1b[1;2C": dict(action="RotateRight"), #commented out by ahmed
            # "\x1b[1;2D": dict(action="RotateLeft"), #commented out by ahmed
            "\x1b[A": dict(action="MoveArmBase", y=i, speed=1, returnToStart=False, fixedDeltaTime=fixedDeltaTime), #added by ahmed
            "\x1b[B": dict(action="MoveArmBase", y=i, speed=1, returnToStart=False, fixedDeltaTime=fixedDeltaTime), #added by ahmed
            "7": dict(action="MoveArm",position=dict(x=x, y=y, z=z),coordinateSpace="wrist",restrictMovement=False,speed=1,returnToStart=False,fixedDeltaTime=fixedDeltaTime), #added by ahmed
            "4": dict(action="MoveArm",position=dict(x=x, y=y, z=z),coordinateSpace="wrist",restrictMovement=False,speed=1,returnToStart=False,fixedDeltaTime=fixedDeltaTime), #added by ahmed
            "8": dict(action="MoveArm",position=dict(x=x, y=y, z=z),coordinateSpace="wrist",restrictMovement=False,speed=1,returnToStart=False,fixedDeltaTime=fixedDeltaTime), #added by ahmed
            "5": dict(action="MoveArm",position=dict(x=x, y=y, z=z),coordinateSpace="wrist",restrictMovement=False,speed=1,returnToStart=False,fixedDeltaTime=fixedDeltaTime), #added by ahmed
            "9": dict(action="MoveArm",position=dict(x=x, y=y, z=z),coordinateSpace="wrist",restrictMovement=False,speed=1,returnToStart=False,fixedDeltaTime=fixedDeltaTime), #added by ahmed
            "6": dict(action="MoveArm",position=dict(x=x, y=y, z=z),coordinateSpace="wrist",restrictMovement=False,speed=1,returnToStart=False,fixedDeltaTime=fixedDeltaTime), #added by ahmed
            "g": dict(action="PickupObject"), #added by ahmed
            "r": dict(action="ReleaseObject"), #added by ahmed
        }
        action_set = {a.name for a in default_actions}
        action_set.add("MoveArmBase") #added by ahmed
        action_set.add("PickupObject") #added by ahmed
        action_set.add("ReleaseObject") #added by ahmed
        action_set.add("RotateAgent") #added by ahmed
        action_set.add("MoveArm") #added by ahmed


        self.default_interact_commands = {
            k: v
            for (k, v) in default_interact_commands.items()
            if v["action"] in action_set
        }

    def interact(
        self,
        controller,
        command,
        semantic_segmentation_frame=False,
        instance_segmentation_frame=False,
        depth_frame=False,
        color_frame=False,
        metadata=False,
    ): 
        
        if not sys.stdout.isatty():
            raise RuntimeError("controller.interact() must be run from a terminal")

        default_interact_commands = self.default_interact_commands

        self._interact_commands = default_interact_commands.copy()

        command_message = u"Enter a Command: Move \u2190\u2191\u2192\u2193, Rotate/Look Shift + \u2190\u2191\u2192\u2193, Quit 'q' or Ctrl-C"
        # print(command_message) #commented out by ahmed
        
        get_data_obj.gather_data(controller.last_event, command) #added by ahmed, adds the starting spawn data
        controller.step(action="SetHandSphereRadius", radius=0.1) #added by ahmed
        global i,x,y,z
        event = controller.step(action="MoveArmBase", y=i,speed=1,returnToStart=False,fixedDeltaTime=fixedDeltaTime)
        i+=incr
        
        for a, ch in self.next_interact_command(command):
            # new_commands = {}
            # command_counter = dict(counter=1)

            # def add_command(cc, action, **args):
            #     if cc["counter"] < 15:
            #         com = dict(action=action)
            #         com.update(args)
            #         new_commands[str(cc["counter"])] = com
            #         cc["counter"] += 1

            # print("a", a)
            if a['action'] == "MoveArmBase": #block added by ahmed
                if ch == "A":
                    i+=incr
                    a['y'] = i
                elif ch == "B":
                    i-=incr
                    a['y'] = i
            elif a['action'] == "MoveArm": #block added by ahmed
                if ch == "7":
                    x+=incr
                    a['position']['x'] = x
                elif ch == "4":
                    x-=incr
                    a['position']['x'] = x
                elif ch == "8":
                    y+=incr
                    a['position']['y'] = y
                elif ch == "5":
                    y-=incr
                    a['position']['y'] = y
                elif ch == "9":
                    z+=incr
                    a['position']['z'] = z
                elif ch == "6":
                    z-=incr
                    a['position']['z'] = z

            global k
            if k ==0:
                k+=1
                event = controller.step(a)
            event = controller.step(a)
            get_data_obj.gather_data(event, command) #added by ahmed


            visible_objects = []
            # InteractiveControllerPrompt.write_image(
            #     event,
            #     self.image_dir,
            #     "_{}".format(self.counter),
            #     image_per_frame=self.image_per_frame,
            #     semantic_segmentation_frame=semantic_segmentation_frame,
            #     instance_segmentation_frame=instance_segmentation_frame,
            #     color_frame=color_frame,
            #     depth_frame=depth_frame,
            #     metadata=metadata,
            # )

            # self.counter += 1
            # if self.has_object_actions:
            #     for o in event.metadata["objects"]:
            #         if o["visible"]:
            #             visible_objects.append(o["objectId"])
            #             if o["openable"]:
            #                 if o["isOpen"]:
            #                     add_command(
            #                         command_counter,
            #                         "CloseObject",
            #                         objectId=o["objectId"],
            #                     )
            #                 else:
            #                     add_command(
            #                         command_counter,
            #                         "OpenObject",
            #                         objectId=o["objectId"],
            #                     )

            #             if o["toggleable"]:
            #                 add_command(
            #                     command_counter,
            #                     "ToggleObjectOff",
            #                     objectId=o["objectId"],
            #                 )

            #             if len(event.metadata["inventoryObjects"]) > 0:
            #                 inventoryObjectId = event.metadata["inventoryObjects"][0][
            #                     "objectId"
            #                 ]
            #                 if (
            #                     o["receptacle"]
            #                     and (not o["openable"] or o["isOpen"])
            #                     and inventoryObjectId != o["objectId"]
            #                 ):
            #                     add_command(
            #                         command_counter,
            #                         "PutObject",
            #                         objectId=inventoryObjectId,
            #                         receptacleObjectId=o["objectId"],
            #                     )
            #                     add_command(
            #                         command_counter, "MoveHandAhead", moveMagnitude=0.1
            #                     )
            #                     add_command(
            #                         command_counter, "MoveHandBack", moveMagnitude=0.1
            #                     )
            #                     add_command(
            #                         command_counter, "MoveHandRight", moveMagnitude=0.1
            #                     )
            #                     add_command(
            #                         command_counter, "MoveHandLeft", moveMagnitude=0.1
            #                     )
            #                     add_command(
            #                         command_counter, "MoveHandUp", moveMagnitude=0.1
            #                     )
            #                     add_command(
            #                         command_counter, "MoveHandDown", moveMagnitude=0.1
            #                     )
            #                     add_command(command_counter, "DropHandObject")

            #             elif o["pickupable"]:
            #                 add_command(
            #                     command_counter, "PickupObject", objectId=o["objectId"]
            #                 )

            # self._interact_commands = default_interact_commands.copy()
            # self._interact_commands.update(new_commands)

            # print("Position: {}".format(event.metadata["agent"]["position"])) #commented out by ahmed
            # print(command_message) #commented out by ahmed
            # print("Visible Objects:\n" + "\n".join(sorted(visible_objects))) #commented out by ahmed

            # skip_keys = ["action", "objectId"]
            # for k in sorted(new_commands.keys()):
            #     v = new_commands[k]
            #     command_info = [k + ")", v["action"]]
            #     if "objectId" in v:
            #         command_info.append(v["objectId"])

            #     for ak, av in v.items():
            #         if ak in skip_keys:
            #             continue
            #         command_info.append("%s: %s" % (ak, av))

            #     print(" ".join(command_info))

    def next_interact_command(self, command):
        current_buffer = ""
        while True:
            commands = self._interact_commands
            ch = get_term_character() #added by ahmed
            current_buffer += ch
            if current_buffer == "q":
                get_data_obj.save(command)   #added by ahmed
                exit() #added by ahmed
                break
            elif current_buffer == "\x03":
                exit() #added by ahmed

            if current_buffer in commands:
                yield commands[current_buffer], ch #ch added by ahmed
                current_buffer = ""
            else:
                match = False
                for k, v in commands.items():
                    if k.startswith(current_buffer):
                        match = True
                        break

                if not match:
                    current_buffer = ""

    # @classmethod
    # def write_image(
    #     cls,
    #     event,
    #     image_dir,
    #     suffix,
    #     image_per_frame=False,
    #     semantic_segmentation_frame=False,
    #     instance_segmentation_frame=False,
    #     depth_frame=False,
    #     color_frame=False,
    #     metadata=False,
    # ):
    #     def save_image(name, image, flip_br=False):
    #         # TODO try to use PIL which did not work with RGBA
    #         # image.save(
    #         #     name
    #         # )
    #         import cv2

    #         img = image
    #         if flip_br:
    #             img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    #         cv2.imwrite("{}.png".format(name), img)

    #     def array_to_image(arr, mode=None):
    #         return arr

    #     def json_write(name, obj):
    #         with open("{}.json".format(name), "w") as outfile:
    #             json.dump(obj, outfile, indent=4, sort_keys=True)

    #     frame_writes = [
    #         (
    #             "color",
    #             color_frame,
    #             lambda event: event.frame,
    #             array_to_image,
    #             lambda x, y: save_image(x, y, flip_br=True),
    #         ),
    #         (
    #             "instance_segmentation",
    #             instance_segmentation_frame,
    #             lambda event: event.instance_segmentation_frame,
    #             array_to_image,
    #             save_image,
    #         ),
    #         (
    #             "class_segmentation",
    #             semantic_segmentation_frame,
    #             lambda event: event.semantic_segmentation_frame,
    #             array_to_image,
    #             save_image,
    #         ),
    #         (
    #             "depth",
    #             depth_frame,
    #             lambda event: event.depth_frame,
    #             lambda data: array_to_image(
    #                 (255.0 / data.max() * (data - data.min())).astype(np.uint8)
    #             ),
    #             save_image,
    #         ),
    #         (
    #             "depth_raw",
    #             depth_frame,
    #             lambda event: event.depth_frame,
    #             lambda x: x,
    #             lambda name, x: np.save(
    #                 name.strip(".png").strip("./")
    #                 if image_dir == "."
    #                 else name.strip(".png"),
    #                 x.astype(np.float32),
    #             ),
    #         ),
    #         (
    #             "metadata",
    #             metadata,
    #             lambda event: event.metadata,
    #             lambda x: x,
    #             json_write,
    #         ),
    #     ]

    #     for frame_filename, condition, frame_func, transform, save in frame_writes:
    #         frame = frame_func(event)
    #         if frame is not None:
    #             frame = transform(frame)
    #             image_name = os.path.join(
    #                 image_dir,
    #                 "{}{}".format(
    #                     frame_filename, "{}".format(suffix) if image_per_frame else ""
    #                 ),
    #             )
    #             # print("Image {}, {}".format(image_name, image_dir))
    #             save(image_name, frame)

    #         else:
    #             pass #added by ahmed
    #             # print("No frame present, call initialize with the right parameters")

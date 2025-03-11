# Copyright (c) 2023 Boston Dynamics, Inc.  All rights reserved.
#
# Downloading, reproducing, distributing or otherwise using the SDK Software
# is subject to the terms and conditions of the Boston Dynamics Software
# Development Kit License (20191101-BDSDK-SL).

#This BD code is heavily modified by Ahmed Jaafar

import cv2
import numpy as np
from bosdyn.api import image_pb2
from bosdyn.client.image import ImageClient, build_image_request

ROTATION_ANGLE = {
    'back_fisheye_image': 0,
    'frontleft_fisheye_image': -78,
    'frontright_fisheye_image': -102,
    'left_fisheye_image': 0,
    'right_fisheye_image': 180
}


def pixel_format_type_strings():
    names = image_pb2.Image.PixelFormat.keys()
    return names[1:]


def pixel_format_string_to_enum(enum_string):
    return dict(image_pb2.Image.PixelFormat.items()).get(enum_string)


def transform(image, dtype, num_bytes):
    img = np.frombuffer(image.shot.image.data, dtype=dtype)
    if image.shot.image.format == image_pb2.Image.FORMAT_RAW:
        try:
            # Attempt to reshape array into a RGB rows X cols shape.
            img = img.reshape((image.shot.image.rows, image.shot.image.cols, num_bytes))
        except ValueError:
            # Unable to reshape the image data, trying a regular decode.
            img = cv2.imdecode(img, -1)
    else:
        img = cv2.imdecode(img, -1)

    return img


def capture(robot, format=None, mode="front"):
    robot.sync_with_directory()
    robot.time_sync.wait_for_sync()

    image_client = robot.ensure_client(ImageClient.default_service_name)
    img_sources = ['frontleft_fisheye_image','frontright_fisheye_image']
    depth_sources = ['frontright_depth', 'frontleft_depth']
    imgs = []
    if mode == "front":
        #rgb
        if format == "PIXEL_FORMAT_RGB_U8":

            pixel_format = pixel_format_string_to_enum(format)
            
            image_request = [
                build_image_request(source, pixel_format=pixel_format)
                for source in img_sources
            ]
            image_responses = image_client.get_image(image_request)

            for image in image_responses:
                num_bytes = 3
                dtype = np.uint8
                # extension = '.jpg'
                img = transform(image, dtype, num_bytes)
                img = np.rot90(img, k=3)
                imgs.append(img)
        #black and white depth
        elif format == "PIXEL_FORMAT_DEPTH_U16":
            pixel_format = pixel_format_string_to_enum(format)
            
            image_request = [
                build_image_request(source, pixel_format=pixel_format)
                for source in depth_sources
            ]
            image_responses = image_client.get_image(image_request)

            for image in image_responses:
                num_bytes = 1 
                dtype = np.uint16
                # extension = '.png'
                img = transform(image, dtype, num_bytes)
                img = np.rot90(img, k=3)
                imgs.append(img)

        #colored depth
        else:
            left_sources = ['frontleft_depth_in_visual_frame', 'frontleft_fisheye_image']
            right_sources = ['frontright_depth_in_visual_frame', 'frontright_fisheye_image']

            image_client = robot.ensure_client(ImageClient.default_service_name)

            # Capture and save images to disk
            image_responses_lst = [image_client.get_image_from_sources(left_sources), image_client.get_image_from_sources(right_sources)]

            for image_responses in image_responses_lst:
                # Depth is a raw bytestream
                cv_depth = np.frombuffer(image_responses[0].shot.image.data, dtype=np.uint16)
                cv_depth = cv_depth.reshape(image_responses[0].shot.image.rows, image_responses[0].shot.image.cols)

                # Visual is a JPEG
                cv_visual = cv2.imdecode(np.frombuffer(image_responses[1].shot.image.data, dtype=np.uint8), -1)

                # Convert the visual image from a single channel to RGB so we can add color
                visual_rgb = cv_visual if len(cv_visual.shape) == 3 else cv2.cvtColor(cv_visual, cv2.COLOR_GRAY2RGB)

                # Map depth ranges to color

                # cv2.applyColorMap() only supports 8-bit; convert from 16-bit to 8-bit and do scaling
                min_val = np.min(cv_depth)
                max_val = np.max(cv_depth)
                depth_range = max_val - min_val
                depth8 = (255.0 / depth_range * (cv_depth - min_val)).astype('uint8')
                depth8_rgb = cv2.cvtColor(depth8, cv2.COLOR_GRAY2RGB)
                depth_color = cv2.applyColorMap(depth8_rgb, cv2.COLORMAP_JET)

                # Add the two images together.
                out = cv2.addWeighted(visual_rgb, 0.5, depth_color, 0.5, 0)
                img = np.rot90(out, k=3)
                imgs.append(img)

        return imgs
    #gripper rgb
    else:
        image_responses = image_client.get_image_from_sources(['hand_color_image'])

        image = image_responses[0]
        if image.shot.image.pixel_format == image_pb2.Image.PIXEL_FORMAT_DEPTH_U16:
            dtype = np.uint16
            
        else:
            dtype = np.uint8
        img = np.fromstring(image.shot.image.data, dtype=dtype)
        if image.shot.image.format == image_pb2.Image.FORMAT_RAW:
            img = img.reshape(image.shot.image.rows, image.shot.image.cols)
        else:
            img = cv2.imdecode(img, -1)

        return img
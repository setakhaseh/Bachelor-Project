#!/usr/bin/env python

# Copyright (c) 2019 Computer Vision Center (CVC) at the Universitat Autonoma de
# Barcelona (UAB).
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.


import glob
import os
import sys
import random
import time
import numpy as np
import carla
import cv2

IM_WIDTH= 640
IM_HEIGHT= 480

def process_img(image):
    i=np.array(image.raw_data)
    #print(i.shape)
    i2=i.reshape((IM_HEIGHT,IM_WIDTH,4))
    i3=i2[:, :, :3]
    cv2.imshow("",i3)
    cv2.waitKey(1)
    return i2/255.0

actor_list = []
try: 
    client = carla.Client('localhost', 2000)
    client.set_timeout(10)
    world=client.get_world()

    blueprint_library=world.get_blueprint_library()
    #vehicle_bp -> bp
    bp=blueprint_library.filter("model3")[0]
    #for locating in specific area-> transform=world.get_map().get_spawn_points()[75]

    spawn_point=random.choice(world.get_map().get_spawn_points())
    vehicle= world.spawn_actor(bp, spawn_point)
    #...........................................


    vehicle.apply_control(carla.VehicleControl(throttle=1.0, steer=0.0))
    actor_list.append(vehicle)

    #adding other sesnors is in the same way
    cam_bp=blueprint_library.find("sensor.camera.rgb")
    cam_bp.set_attribute("image_size_x",f"{IM_WIDTH}")
    cam_bp.set_attribute("image_size_y",f"{IM_HEIGHT}")
    cam_bp.set_attribute("fov","110")#normally 90

    spawn_point= carla.Transform(carla.Location(x=2.5,z=0.7))

    sensor= world.spawn_actor(cam_bp, spawn_point, attach_to= vehicle)
    if not os.path.exists("output\\"):
        os.mkdir("output\\")
    actor_list.append(sensor)
    #sensor.listen(lambda data: data.save_to_disk('output/%.6d.jpg' % data.frame))
    sensor.listen(lambda data: process_img(data))
    vehicle.set_autopilot(True)
    time.sleep(5)

finally:
    for actor in actor_list:
        actor.destroy()


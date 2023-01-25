import os
import time

import cv2
import matplotlib.pyplot as plt
import mujoco_py
import numpy as np
from matplotlib import animation


def save_movie(video_data, dt=0.04):
    fig = plt.figure()
    flame = []
    time_count = 0
    for img in video_data:
        time_count += 1
        flame.append([
            plt.imshow(img, animated=True),
            plt.text(0.5, 0.5, round(time_count * dt, 1))
        ])
    ani = animation.ArtistAnimation(fig, flame, interval=dt * 1000)
    plt.rcParams['animation.ffmpeg_path'] = '/usr/bin/ffmpeg'
    writer = animation.FFMpegWriter(fps=1.0 / dt, codec='vp9')
    movie_path = os.getcwd()
    print(movie_path)
    ani.save(movie_path + '/movie.mp4', writer=writer)
    plt.close()


cwd_path = os.getcwd()
xml_path = os.path.join(cwd_path, '8_link_arm.xml')
print("model file is located at {}".format(xml_path))
model = mujoco_py.load_model_from_path(xml_path)
sim = mujoco_py.MjSim(model)
data = sim.data

width, height = 1000, 1000
video_frame = []
for i in range(100):
    #print(sim.data.qpos)
    viewer = mujoco_py.MjRenderContextOffscreen(sim, device_id=-1)
    viewer.render(width, height)
    frame = viewer.read_pixels(width, height, depth=False)
    frame = frame[::-1, :, :]
    cv2.imwrite("arm.png", frame)
    video_frame.append(frame)
    time.sleep(0.05)
    ctrl = np.array([0.1] * 8)
    sim.data.ctrl[:] = ctrl
    for _ in range(5):
        sim.step()
    #print(sim.data.qpos, sim.data.qvel)

save_movie(video_frame)

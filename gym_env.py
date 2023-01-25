import os

import cv2
import gym
import matplotlib.pyplot as plt
import mujoco_py
import numpy as np
from gym import spaces
from gym.utils import seeding
from matplotlib import animation

DEFAULT_RENDER_SIZE = 1000


class NLinkArm(gym.Env):

    def __init__(self, number_of_links=8, max_step_num=200):
        self.number_of_links = number_of_links

        cwd_path = os.getcwd()
        xml_path = os.path.join(
            cwd_path, '{}_link_arm.xml'.format(self.number_of_links))
        print("model_file is at {}".format(xml_path))
        model = mujoco_py.load_model_from_path(xml_path)
        self.sim = mujoco_py.MjSim(model)
        self.model = self.sim.model
        self.data = self.sim.data

        self.max_step_num = max_step_num
        self.frame_skip = 5
        self.dt = self.model.opt.timestep * self.frame_skip
        self._viewer = None

        observation_limit = np.array(
            [np.pi] * self.number_of_links +
            [np.finfo(np.float32).max] * self.number_of_links,
            dtype=np.float32)
        self.observation_space = spaces.Box(low=-observation_limit,
                                            high=observation_limit,
                                            dtype=np.float32)

        action_limit = np.array([1.0] * self.number_of_links)
        self.action_space = spaces.Box(low=-action_limit,
                                       high=action_limit,
                                       dtype=np.float32)

        self.seed()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self):
        qpos = np.zeros(self.number_of_links)
        qvel = np.zeros(self.number_of_links)
        self._set_state(0, qpos, qvel)
        state = self._get_state()
        return state

    def step(self, action):
        ctrl = action
        self.data.ctrl[:] = ctrl
        for _ in range(self.frame_skip):
            self.sim.step()

        state = self._get_state()
        reward = self._get_reward(state)
        done = self._get_done()

        return state, reward, done, _

    def _set_state(self, time, qpos, qvel):
        state = self.sim.get_state()
        state = mujoco_py.MjSimState(time, qpos, qvel, state.act,
                                     state.udd_state)
        self.sim.set_state(state)
        self.sim.forward()

    def _get_qpos(self):
        qpos = self.data.qpos % (2 * np.pi)
        return qpos

    def _get_state(self):
        qpos = self._get_qpos()
        state = np.array([*qpos, *self.data.qvel])
        return state

    def _get_reward(self, state):
        qpos = state[:self.number_of_links]
        end_pos = self.get_endpos(qpos)[-1]

        return -end_pos[1] + (-end_pos[1] > 0) + 5.0 * (
            -end_pos[1] > 0.9 * 0.5 * self.number_of_links)

    def get_endpos(self, qpos):
        assert qpos.shape == (self.number_of_links,)
        pos_x = 0
        pos_y = 0
        angle_sum = 0
        pos_list = []
        for theta in qpos:
            angle_sum += theta
            pos_x += 0.5 * np.sin(angle_sum)
            pos_y += 0.5 * np.cos(angle_sum)
            pos_list.append([pos_x, pos_y, theta])
        return pos_list

    def _get_done(self):
        now_time = self.sim.get_state()[0]
        if now_time + 0.01 > self.max_step_num * self.dt:
            done = True
        else:
            done = False
        return done

    def render(
        self,
        mode='rgb_array',
        width: int = DEFAULT_RENDER_SIZE,
        height: int = DEFAULT_RENDER_SIZE,
        override_endpos=False,
    ):
        if not self._viewer:
            _viewer = mujoco_py.MjRenderContextOffscreen(self.sim, device_id=-1)
        _viewer.render(width, height)
        frame = _viewer.read_pixels(width, height, depth=False)
        frame = frame[::-1, :, :].copy()

        if override_endpos:
            qpos = self._get_qpos()
            end_positions = self.get_endpos(qpos)
            text_height = 600
            for end_pos in end_positions:
                text = "x : {:.2f}, y : {:.2f}, theta : {:.2f}".format(
                    end_pos[0], end_pos[1], end_pos[2])
                cv2.putText(frame,
                            str(text), (400, text_height),
                            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                            fontScale=1,
                            color=(255, 255, 255),
                            thickness=1)
                text_height += 50

        return frame


def save_movie(video_data, dt=0.05):
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
    print("movie is saved to", movie_path + '/movie.mp4')
    ani.save(movie_path + '/movie.mp4', writer=writer)
    plt.close()


def test_env():
    env = NLinkArm()
    env.reset()
    video_data = []
    action = np.ones(8) * 1.0
    for i in range(120):
        _, _, done, _ = env.step(action)
        frame = env.render(override_endpos=False)
        video_data.append(frame)
        if done:
            break

    save_movie(video_data)

    print(env.dt)


if __name__ == "__main__":
    test_env()

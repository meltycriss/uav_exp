import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
# import matplotlib.patches as patches

class Env(object):
    def __init__(self, obs_r, goal, bound, uav_r, uav_n):
        self.hp_bound = bound # range of map
        self.hp_n_nearest_obs = 3
        self.hp_n = uav_n
        self.hp_dim = 2
        self.hp_thres = .2
        self.hp_uav_r = uav_r

        self.s_obs_pos = None
        self.s_obs_r = obs_r
        self.s_goal = goal
        self.s_uav_pos = None

        self.get_o_env, self.vis_o_env = self._get_o_env_pose, self._vis_o_env_pose
        self.get_o_partner, self.vis_o_partner = self._get_o_partner_pose, self._vis_o_partner_pose
        self.get_o_goal, self.vis_o_goal = self._get_o_goal_pose, self._vis_o_goal_pose

        self.vis_init = True

    def step(self, bot_pos, obs_pos):
        self.s_uav_pos = bot_pos.reshape((self.hp_n, self.hp_dim))
        self.s_obs_pos = obs_pos.reshape((self.s_obs_r.shape[0], self.hp_dim))
        self.update_observation()
        o = self.get_observation()
        centroid = np.average(self.s_uav_pos, axis=0)
        dis = np.linalg.norm(centroid-self.s_goal)
        done = dis <= self.hp_thres
        return o, done

    def render(self):
        if self.vis_init:
            plt.ion()
            self.vis_fig = plt.figure(figsize=self.hp_bound, dpi=160)
            self.vis_ax = self.vis_fig.gca()
            self.vis_ax.set_xlim(-self.hp_bound[0], self.hp_bound[0])
            self.vis_ax.set_ylim(-self.hp_bound[1], self.hp_bound[1])
            # self.vis_ax.axis('off')

        self.vis_state()
        self.vis_o_env()
        self.vis_o_partner()
        self.vis_o_goal()

        self.vis_fig.canvas.draw()
        self.vis_fig.canvas.flush_events()

        if self.vis_init:
            self.vis_init = False

    def vis_state(self):
        if self.vis_init:
            goal_color = (1, 0, 0)
            centroid_pos_color = (0, 1, 0)
            uav_pos_color = (0, 0, 1)
            obs_pos_color = (.5, .5, .5)

            # goal
            self.vis_s_goal_pos = plt.Circle(np.zeros(2), self.hp_thres, color=goal_color)
            self.vis_ax.add_patch(self.vis_s_goal_pos)

            # centroid
            self.vis_s_centroid_pos = plt.Circle(np.zeros(2), .1, color=centroid_pos_color)
            self.vis_ax.add_patch(self.vis_s_centroid_pos)

            # uav
            self.vis_s_uav_pos = [plt.Circle(np.zeros(2), self.hp_uav_r, color=uav_pos_color) for _ in range(self.hp_n)]
            for i in range(self.hp_n):
                self.vis_ax.add_patch(self.vis_s_uav_pos[i])

            # obs
            self.vis_s_obs_pos = [plt.Circle(np.zeros(2), self.s_obs_r[i], color=obs_pos_color) for i in range(self.s_obs_pos.shape[0])]
            for i in range(self.s_obs_pos.shape[0]):
                self.vis_ax.add_patch(self.vis_s_obs_pos[i])

        # goal
        self.vis_s_goal_pos.center = self.s_goal

        # centroid
        centroid_pos = np.average(self.s_uav_pos, axis=0)
        self.vis_s_centroid_pos.center = centroid_pos

        # uav
        for i in range(self.hp_n):
            self.vis_s_uav_pos[i].center = self.s_uav_pos[i]

        # obs
        for i in range(self.s_obs_pos.shape[0]):
            self.vis_s_obs_pos[i].center = self.s_obs_pos[i]

    ############################################
    # observation stuff
    ############################################

    def update_observation(self):
        self.o_env = [self.get_o_env(i) for i in range(self.hp_n)]
        self.o_partner = [self.get_o_partner(i) for i in range(self.hp_n)]
        self.o_goal = [self.get_o_goal(i) for i in range(self.hp_n)]

    def get_observation(self):
        o = []
        for i in range(self.hp_n):
            o_i = np.hstack([
                self.o_env[i].flatten(),
                self.o_partner[i].flatten(),
                self.o_goal[i].flatten(),
                ])
            o.append(o_i)
        o = np.hstack(o)
        return o

    def _get_o_env_pose(self, i):
        p = self.s_uav_pos[i]
        # normalize bounds and normal obstacles
        bound_pos = np.array([p for _ in range(2*self.hp_dim)]) # surface # is 2*dim
        for i in range(self.hp_dim):
            bound_pos[2*i][i] = -self.hp_bound[i]
            bound_pos[2*i+1][i] = self.hp_bound[i]
        bound_r = np.zeros(bound_pos.shape[0])

        rela_pos = np.vstack([self.s_obs_pos-p, bound_pos-p])
        r = np.hstack([self.s_obs_r, bound_r])

        # n nearest obstacles
        dis = np.linalg.norm(rela_pos, axis=1) - r
        order_mask = np.argsort(dis)[:self.hp_n_nearest_obs]

        res_pos = np.zeros((self.hp_n_nearest_obs, self.hp_dim)) # set default to 0, because it's impossible in reality
        res_r = np.zeros(self.hp_n_nearest_obs) # although 0 radius is possible (bound), invalidity can be inferred via res_pos
        res_pos[:order_mask.shape[0]] = rela_pos[order_mask]
        res_r[:order_mask.shape[0]] = r[order_mask]

        return np.hstack([res_pos, np.expand_dims(res_r, axis=1)])

    def _vis_o_env_pose(self):
        if self.vis_init:
            self.vis_env = []
            for _ in range(self.hp_n):
                lines = []
                for _ in range(self.hp_n_nearest_obs):
                    line = Line2D([], [])
                    self.vis_ax.add_line(line)
                    lines.append(line)
                self.vis_env.append(lines)

        for i in range(self.hp_n):
            obs_rela_pos = self.o_env[i][:,:-1]
            obs_r = self.o_env[i][:,-1]
            norm = np.linalg.norm(obs_rela_pos, axis=1)
            dir_vec = obs_rela_pos / np.expand_dims(norm, axis=1)
            intersec_point_rela_pos = dir_vec * np.expand_dims(norm-obs_r, axis=1)
            intersec_point_pos = self.s_uav_pos[i] + intersec_point_rela_pos
            for j in range(self.hp_n_nearest_obs):
                tmp = np.vstack((self.s_uav_pos[i], intersec_point_pos[j]))
                self.vis_env[i][j].set_data(tmp[:,0], tmp[:,1])

    def _get_o_partner_pose(self, i):
        p = self.s_uav_pos[i]
        res = self.s_uav_pos - p
        res = np.delete(res, i, 0)
        return res

    def _vis_o_partner_pose(self):
        if self.vis_init:
            partner_color = (0, 1, 1)
            self.vis_partner = []
            for _ in range(self.hp_n):
                lines = []
                for _ in range(self.hp_n-1):
                    line = Line2D([], [], color=partner_color)
                    self.vis_ax.add_line(line)
                    lines.append(line)
                self.vis_partner.append(lines)

        for i in range(self.hp_n):
            uav_rela_pos = self.o_partner[i]
            for j in range(self.hp_n-1):
                tmp = np.vstack((self.s_uav_pos[i], self.s_uav_pos[i]+uav_rela_pos[j]))
                self.vis_partner[i][j].set_data(tmp[:,0], tmp[:,1])

    def _get_o_goal_pose(self, i):
        p = self.s_uav_pos[i]
        res = self.s_goal - p
        return res

    def _vis_o_goal_pose(self):
        if self.vis_init:
            goal_color = (238./255., 130./255., 238./255.)
            self.vis_goal = []
            for _ in range(self.hp_n):
                line = Line2D([], [], color=goal_color)
                self.vis_ax.add_line(line)
                self.vis_goal.append(line)

        for i in range(self.hp_n):
            tmp = np.vstack((self.s_uav_pos[i], self.s_uav_pos[i] + self.o_goal[i]))
            self.vis_goal[i].set_data(tmp[:,0], tmp[:,1])

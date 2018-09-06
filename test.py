from Env import Env
import rpyc
import numpy as np

hp_dim = 2
uav_r = .3
obs_r = np.array([.35, .35]) # ORDER IS IMPORTANT
goal = np.array([0., 2.5])
bound = np.array([2., 4.]) # range of the map
hp_n_bot = 3 # number of robots

rpyc.core.protocol.DEFAULT_CONFIG['allow_pickle'] = True

def clip_to_max(vs, maximum):
    vs = vs.reshape((-1, hp_dim)).copy()
    mag = np.linalg.norm(vs, axis=1)
    mask = mag > maximum
    vs[mask] = (vs / np.expand_dims(mag, axis=1) * maximum)[mask]
    return vs

if __name__=='__main__':
    conn = rpyc.connect("172.18.197.179", 18861)
    s = conn.root.get_init_state()

    env = Env(obs_r, goal, bound, uav_r, hp_n_bot)

    test_bot_pos = np.array([[0., 1.4], [-.8, 0.], [.8, 0]]) + np.array([0., -3.])
    # test_bot_pos = np.array([[0., 1.4], [-.8, 0.], [.8, 0], [0., -1.4]]) + np.array([0., -1.])
    test_obs_pos = np.array([[-.8, .3], [.8, .3]])
    v = np.zeros((hp_n_bot, hp_dim))

    counter = 0
    while True:
        # test observation
        o, done = env.step(test_bot_pos, test_obs_pos, v)
        env.render()
        v, s = conn.root.get_velocity(o, s)
        v = rpyc.utils.classic.obtain(v)
        # v = np.arange(6).astype(np.float32)/10.
        v = v.reshape((hp_n_bot, hp_dim))
        v = env.local2global(v)
        v = clip_to_max(v, .5)

        if counter < 2:
            print (o)
            print (v)
            print (np.linalg.norm(v, axis=1))

        test_bot_pos += .5 * v

        counter += 1

        if done:
            break

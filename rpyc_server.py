import rpyc
import numpy as np
import time
import argparse
import torch

rpyc.core.protocol.DEFAULT_CONFIG['allow_pickle'] = True

class MyService(rpyc.Service):
    def __init__(self, model, cuda):
        self.cuda = cuda
        self.actor_critic, self.ob_rms = torch.load(model)
        if self.cuda:
            self.actor_critic.cuda()

    def exposed_get_velocity(self, o, s):
        o = rpyc.utils.classic.obtain(o)
        o = np.clip((o - self.ob_rms.mean) / np.sqrt(self.ob_rms.var + 1e-8), -10., 10.)

        # ensure float32 rather than float64
        current_obs = torch.from_numpy(o).unsqueeze(0).float()
        states = torch.from_numpy(s).float()
        masks = torch.ones((1,1)).float() # should not be torch.zeros((1,1))
        if self.cuda:
            current_obs = current_obs.cuda()
            states = states.cuda()
            masks = masks.cuda()

        with torch.no_grad():
            value, action, _, states = self.actor_critic.act(current_obs,
                                                        states,
                                                        masks,
                                                        deterministic=True)

        cpu_actions = action.squeeze(0).cpu().numpy()
        cpu_states = states.cpu().numpy()
        return cpu_actions, cpu_states

    def exposed_get_init_state(self):
        state = np.zeros((1, self.actor_critic.state_size))
        return state

    # def exposed_get_velocity(self, o, s):
    #     o = rpyc.utils.classic.obtain(o)
    #     s = rpyc.utils.classic.obtain(s)
    #     o = o.astype(np.float32)
    #     v = np.zeros((3, 2))
    #     v[:,1] = .7
    #     v *= s
    #     return v

    # def exposed_get_velocity_diag(self, o, s):
    #     o = rpyc.utils.classic.obtain(o)
    #     s = rpyc.utils.classic.obtain(s)
    #     o = o.astype(np.float32)
    #     v = np.zeros((3, 2))
    #     v[:,0] = .7
    #     v[:,1] = .7
    #     v *= s
    #     return v

    # def on_connect(self, conn):
    #     # code that runs when a connection is created
    #     # (to init the service, if needed)

    # def on_disconnect(self, conn):
    #     # code that runs after the connection has already closed
    #     # (to finalize the service, if needed)

    # def exposed_get_answer(self, x): # this is an exposed method
    #     return np.ones(6)
    #     # return 42

    # exposed_the_real_answer_though = 43     # an exposed attribute

    # def get_question(self):  # while this method is not exposed
    #     return "what is the airspeed velocity of an unladen swallow?"

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='server')
    parser.add_argument('--load-dir', help='directory to load model (e.g.: ./exp_models/multi_miniright2_withvprev_withmindis_highrewvpref_training-v0/model4500.pt)')
    parser.add_argument('--cuda', default=True, help='enable GPU (default: True)')
    args = parser.parse_args()

    from rpyc.utils.server import ThreadedServer
    # t = ThreadedServer(MyService(), port=18861)
    t = ThreadedServer(MyService(args.load_dir, args.cuda), port=18861)
    t.start()


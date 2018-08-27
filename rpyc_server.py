import rpyc
import numpy as np
import time
import argparse
import torch

rpyc.core.protocol.DEFAULT_CONFIG['allow_pickle'] = True

class MyService(rpyc.Service):
    # def __init__(self, model, cuda):
    #     self.cuda = cuda
    #     self.actor_critic, _ = torch.load(model)
    #     if self.cuda:
    #         self.actor_critic.cuda()

    # def exposed_get_velocity(self, o, s):
    #     o = rpyc.utils.classic.obtain(o)
    #     o = o.astype(np.float32)
    #     s = s.astype(np.float32)
    #     current_obs = torch.from_numpy(o).unsqueeze(0)
    #     states = torch.from_numpy(s)
    #     masks = torch.zeros((1,1))

    #     if self.cuda:
    #         current_obs = current_obs.cuda()
    #         states = states.cuda()
    #         masks = masks.cuda()
    #     with torch.no_grad():
    #         value, action, _, states = self.actor_critic.act(current_obs,
    #                                                     states,
    #                                                     masks,
    #                                                     deterministic=True)

    #     cpu_actions = action.squeeze(0).cpu().numpy()
    #     cpu_states = states.cpu().numpy()
    #     print (cpu_actions)
    #     return cpu_actions, cpu_states

    # def exposed_get_init_state(self):
    #     state = np.zeros((1, self.actor_critic.state_size))
    #     return state

    def exposed_get_velocity(self, o):
        o = rpyc.utils.classic.obtain(o)
        o = o.astype(np.float32)
        v = np.zeros((3, 2))
        v[:,1] = -0.2
        return v

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
    # parser = argparse.ArgumentParser(description='server')
    # parser.add_argument('--load-dir', help='directory to load model (e.g.: ./trained_models/)')
    # parser.add_argument('--cuda', default=True, help='enable GPU (default: True)')
    # args = parser.parse_args()

    from rpyc.utils.server import ThreadedServer
    t = ThreadedServer(MyService(), port=18861)
    # t = ThreadedServer(MyService(args.load_dir, args.cuda), port=18861)
    t.start()


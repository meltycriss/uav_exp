import queue
from NatNetClient import NatNetClient
import time
import serial
import struct
import numpy as np
from Env import Env
import argparse
import rpyc
from pyquaternion import Quaternion
import math
import common

###########################################
# ATTENTION
# order of rigid bodies in motive matters!
#   - robots should precede obstacles
#   - obs_r[i] corresponds to i+1-th obstacles
###########################################

###########################################
# task specification
###########################################
uav_r = .3
obs_r = np.array([.4, .4, .4]) # ORDER IS IMPORTANT
goal = np.array([0., 3.])
bound = np.array([2., 4.]) # range of the map

###########################################
# hyper-parameter
###########################################

hp_n_bot = 3 # number of robots
hp_n_obs = obs_r.shape[0] # number of obstacles
hp_dim = 2
hp_queue_size = 1 # queue buffer size
hp_local_fps = 50
hp_global_fps = 15

###########################################
# optitrack stuff
###########################################

qs_bot_pos = [queue.Queue(hp_queue_size) for _ in range(hp_n_bot)]
qs_bot_rot = [queue.Queue(hp_queue_size) for _ in range(hp_n_bot)]
qs_obs_pos = [queue.Queue(hp_queue_size) for _ in range(hp_n_obs)]

def quaternion_to_euler_angle(w, x, y, z):
    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + y * y)
    X = math.degrees(math.atan2(t0, t1))
    t2 = +2.0 * (w * y - z * x)
    t2 = +1.0 if t2 > +1.0 else t2
    t2 = -1.0 if t2 < -1.0 else t2
    Y = math.degrees(math.asin(t2))
    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    Z = math.degrees(math.atan2(t3, t4))
    return X, Y, Z

def receiveRigidBodyFrame(id, position, rotation):
    global qs_bot_pos
    global qs_bot_rot
    global qs_obs_pos

    # robots should precede obstacles
    # position
    if id<=hp_n_bot:
        q = qs_bot_pos[id-1]
    elif id-hp_n_bot<=hp_n_obs:
        q = qs_obs_pos[id-hp_n_bot-1]
    else: # meaningless rigid bodies
        return

    if q.full():
        q.get()
    q.put(position)

    # rotation
    if id<=hp_n_bot:
        q = qs_bot_rot[id-1]
        # rotation: x, y, z, w
        X, Y, Z = quaternion_to_euler_angle(rotation[3], rotation[0], rotation[1], rotation[2])
        if q.full():
            q.get()
        # positive for clockwise
        q.put(Z) # only care about yaw

###########################################
# serial port stuff
###########################################

# command reference
CMD_TAKEOFF = 1
CMD_LAND = 2
CMD_RESET = 3
CMD_CTRL = 4

# serial port parameter
ser = serial.Serial()
ser.port = "/dev/ttyUSB0"
ser.baudrate = 115200
ser.bytesize = serial.EIGHTBITS
ser.parity = serial.PARITY_NONE
ser.timeout = 1.

# auxiliary function encoding float to unsigned int
def float_to_uint(f):
    # >I refers to big endian unsigned integer
    # >f refers to big endian float32
    return struct.unpack('>I', struct.pack('>f', f))[0]

#####################################################################
#   info        #   size    #   remark
#####################################################################
# header        #   1B      #   0xfe
# robot index   #   1B      #
# command       #   1B      #
# v_x           #   4B      #   big endian(significant first) float32
# v_y           #   4B      #
# v_z           #   4B      #
# w             #   4B      #
# checksum      #   1B      #   byte-wise sum of v_x, v_y, v_z and w
#####################################################################
def sendCommand(id, cmd, x, y, z, w):
    assert isinstance(id, int)
    assert isinstance(cmd, int)
    assert isinstance(x, float)
    assert isinstance(y, float)
    assert isinstance(z, float)
    assert isinstance(w, float) # rotation
    # restriction of the receiver
    if x>=3 or y>=3 or z>=3 or w>=3:
        print ("[WARNING] variables >= 3: {}".format((x, y, z, w)))

    header = bytearray.fromhex('fe')
    index  = bytearray.fromhex(format(id, '02x')) # robot are 1-idx
    command  = bytearray.fromhex(format(cmd, '02x'))

    ctrl_vars = [x, y, z, w]
    ctrl_vars_uint = list(map(float_to_uint, ctrl_vars))
    ctrl_vars_ba = bytearray()
    for ctrl_var in ctrl_vars_uint:
        ctrl_vars_ba += bytearray.fromhex(format(ctrl_var, '08x'))

    bytewise_sum = sum([b for b in ctrl_vars_ba])
    checksum = bytearray.fromhex(format(bytewise_sum % 100, '02x'))

    # for b in ctrl_vars_ba:
    #     print (hex(b))
    # print (bytewise_sum)
    # print (int.from_bytes(index, byteorder='big'), )

    frame = header + index + command + ctrl_vars_ba + checksum
    num_of_bytes = ser.write(frame)
    # print (num_of_bytes)

# adapt to bot's local coordinate system
#
#   from   y       to          x
#          |                   |
#          |                   |
#          ----x           y----
def adapt_to_bot_frame(v):
    mat = np.array([[0., 1.], [-1., 0.]])
    v = v.reshape((-1, hp_dim))
    return np.dot(mat, v.transpose()).transpose()

# rectify yaw
# rots: list of degrees, [0, 180] for clockwise, [-180, 0] for counter-clockwise
def adapt_to_bot_yaw(vs, rots):
    rots = np.array(rots) / 180. * np.pi
    rot_mat = [np.array([[np.cos(rot), np.sin(rot)], [-np.sin(rot), np.cos(rot)]]) for rot in rots]
    vs = vs.reshape((-1, hp_dim))
    res = []
    for i in range(vs.shape[0]):
        rot = rot_mat[i]
        v = vs[i]
        v = np.dot(rot, v)
        res.append(v)
    return np.array(res)

def clip_to_max(vs, maximum):
    vs = vs.reshape((-1, hp_dim)).copy()
    mag = np.linalg.norm(vs, axis=1)
    mask = mag > maximum
    vs[mask] = (vs / np.expand_dims(mag, axis=1) * maximum)[mask]
    return vs

def policy(o, sign):
    v = np.zeros((hp_n_bot, hp_dim))
    # v[:,0] = 0.2
    v[:,1] = 0.7
    v *= sign
    return v

###########################################
# RPyC stuff
###########################################

rpyc.core.protocol.DEFAULT_CONFIG['allow_pickle'] = True

###########################################
# main
###########################################
if __name__=='__main__':
    # parameter
    parser = argparse.ArgumentParser(description='ctrl')
    def str2bool(v):
        if v.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        elif v.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        else:
            raise argparse.ArgumentTypeError('Boolean value expected.')
    # parser.add_argument('--id', type=int , nargs='+', default=[i+1 for i in range(hp_n_bot)], help='relevant robot ids (default: {})'.format([i+1 for i in range(hp_n_bot)]))
    parser.add_argument('--id', type=int , nargs='+', default=[], help='relevant robot ids (default: [])')
    parser.add_argument('--render', type=str2bool , default=False, help='real-time rendering (default: False)')
    args = parser.parse_args()

    for id in args.id:
        assert id <= hp_n_bot

    # run motive client
    streamingClient = NatNetClient("172.18.196.170")
    streamingClient.rigidBodyListener = receiveRigidBodyFrame
    streamingClient.run()

    # open serial port communication
    ser.open()
    time.sleep(.1) # ensure serial port is ready

    # connect remote python server
    conn = rpyc.connect("172.18.197.179", 18861)
    s = conn.root.get_init_state()

    env = Env(obs_r, goal, bound, uav_r, hp_n_bot)

    counter = 0
    # main loop
    while True:
        # fetch data from optitrack
        bot_pos = np.zeros((hp_n_bot, hp_dim))
        for i, q in enumerate(qs_bot_pos):
            p = q.get()
            bot_pos[i] = p[0:2] # only care about x and y

        bot_rot = []
        for q in qs_bot_rot:
            rot = q.get()
            bot_rot.append(rot)

        obs_pos = np.zeros((hp_n_obs, hp_dim))
        for i, q in enumerate(qs_obs_pos):
            p = q.get()
            obs_pos[i] = p[0:2] # only care about x and y

        # compute current velocity
        bot_v = np.zeros((hp_n_bot, hp_dim)) if counter==0 else (bot_pos - bot_pos_prev) / (time.time() - bot_timer)
        bot_pos_prev = bot_pos.copy()
        bot_timer = time.time()
        # print (np.linalg.norm(bot_v, axis=1))

        # for simulation
        if counter==0:
            sim_bot_pos = bot_pos
            sim_v = np.zeros((hp_n_bot, hp_dim))
        else:
            # sim_bot_pos += (time.time() - sim_bot_timer) * sim_v
            sim_bot_pos += .5 / .7 * sim_v # neglect real latency

        # mix real and sim
        mix_bot_pos = sim_bot_pos.copy()
        # mix_bot_v = sim_v.copy()
        mix_bot_v = sim_v.copy() / .7 # neglect real latency
        for id in args.id:
            mix_bot_pos[id-1] = bot_pos[id-1]
            mix_bot_v[id-1] = bot_v[id-1]

        # # get observation
        o, done = env.step(mix_bot_pos, obs_pos, mix_bot_v)
        if args.render:
            env.render()

        # compute action
        rpyc_timer = time.time()
        v, s = conn.root.get_velocity(o, s)
        # v = conn.root.get_velocity(o, -1.)
        # v = conn.root.get_velocity_diag(o, .3/.7)
        v = rpyc.utils.classic.obtain(v)
        # print ("rpyc delay: {0:.2f}ms".format(1000*(time.time()-rpyc_timer)))
        # v = policy(o, 1.)
        v = v.reshape((hp_n_bot, hp_dim))
        v = env.local2global(v) # action from policy bases on local frame
        v = clip_to_max(v, .5)
        # print (v)
        # for simulation
        sim_v = v.copy()
        delay_ratio = .7
        sim_v *= delay_ratio
        # if counter % 2 == 0:
        #     v[:,0] *= -1.
        v = adapt_to_bot_frame(v)
        v = adapt_to_bot_yaw(v, bot_rot)

        if counter > 0:
            print ("real fps: {0:.0f}Hz".format(1./(time.time()-fps_timer)))
        # send command to robots
        for i in range(hp_n_bot):
            if i+1 in args.id:
                sendCommand(common.logic2real[i+1], CMD_CTRL, v[i][0], v[i][1], 0., 0.) # robot is 1-idx
            time.sleep(1./hp_local_fps)
        fps_timer = time.time()

        # for simulation
        sim_bot_timer = time.time()

        # # send command to robots
        # for id in args.id:
        #     sendCommand(common.logic2real[id], CMD_CTRL, v[id-1][0], v[id-1][1], 0., 0.) # robot is 1-idx
        #     time.sleep(1./hp_local_fps)


        if done:
            # pass
            break
        else:
            # control fps
            time.sleep(1./hp_global_fps)
            counter += 1

    # close serial port communication
    ser.flush() # ensure no remaining data in buffer before closing serial port
    ser.close() # what if the process is killed?

    # stop motive client
    streamingClient.stop()

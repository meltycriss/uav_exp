import queue
from NatNetClient import NatNetClient
import time
import serial
import struct
import numpy as np
from Env import Env
import argparse
import rpyc

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
obs_r = np.array([.4, .3]) # ORDER IS IMPORTANT
goal = np.array([0., 2.])
bound = np.array([2., 4.]) # range of the map

###########################################
# hyper-parameter
###########################################

hp_n_bot = 3 # number of robots
hp_n_obs = obs_r.shape[0] # number of obstacles
hp_dim = 2
hp_queue_size = 10 # queue buffer size
hp_local_fps = 100
hp_global_fps = 10

###########################################
# optitrack stuff
###########################################

qs_bot = [queue.Queue(hp_queue_size) for _ in range(hp_n_bot)]
qs_obs = [queue.Queue(hp_queue_size) for _ in range(hp_n_obs)]

def receiveRigidBodyFrame(id, position, rotation):
    global qs_bot
    global qs_obs
    # robots should precede obstacles
    if id<=hp_n_bot:
        q = qs_bot[id-1]
    elif id-hp_n_bot<=hp_n_obs:
        q = qs_obs[id-hp_n_bot-1]
    else: # meaningless rigid bodies
        return

    if q.full():
        q.get()
    q.put(position)

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
    m = np.array([[0., 1.], [-1., 0.]])
    if len(v.shape)==1:
        return np.dot(m, v)
    else:
        return np.dot(m, v.transpose()).transpose()

def policy(o):
    v = np.zeros((hp_n_bot, hp_dim))
    # v[:,0] = -0.7
    v[:,1] = -0.7
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
    # parser.add_argument('--id', type=int , nargs='+', default=[1, 2, 3], help='relevant robot ids (default: [1, 2, 3])')
    parser.add_argument('--id', type=int , nargs='+', default=[i+1 for i in range(hp_n_bot)], help='relevant robot ids (default: {})'.format([i+1 for i in range(hp_n_bot)]))
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
    conn = rpyc.connect("172.18.196.173", 18861)

    env = Env(obs_r, goal, bound, uav_r, hp_n_bot)

    # main loop
    # for n in range(10):
    while True:
        # fetch data from optitrack
        bot_pos = np.zeros((hp_n_bot, hp_dim))
        for i, q in enumerate(qs_bot):
            p = q.get()
            bot_pos[i] = p[0:2] # only care about x and y


        obs_pos = np.zeros((hp_n_obs, hp_dim))
        for i, q in enumerate(qs_obs):
            p = q.get()
            obs_pos[i] = p[0:2] # only care about x and y

        # get observation
        o, done = env.step(bot_pos, obs_pos)
        # env.render()

        # compute action
        v = conn.root.get_velocity(o)
        v = rpyc.utils.classic.obtain(v)
        # v = policy(o)
        # v.reshape((hp_n_bot, hp_dim))
        v = adapt_to_bot_frame(v)

        # send command to robots
        # for i in range(hp_n_bot):
        #     sendCommand(i+1, CMD_CTRL, v[i][0], v[i][1], 0., 0.) # robot is 1-idx
        for id in args.id:
            sendCommand(id, CMD_CTRL, v[id-1][0], v[id-1][1], 0., 0.) # robot is 1-idx
            time.sleep(1./hp_local_fps)

        # control fps
        time.sleep(1./hp_global_fps)

        # if done:
        #     break

    # close serial port communication
    ser.flush() # ensure no remaining data in buffer before closing serial port
    ser.close() # what if the process is killed?

    # stop motive client
    streamingClient.stop()

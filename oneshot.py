import time
import serial
import struct
import argparse

###########################################
# hyper-parameter
###########################################

hp_n_bot = 3 # number of robots
hp_local_fps = 100
hp_global_fps = 20

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
ser.port = "/dev/ttyUSB1"
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

###########################################
# main
###########################################
if __name__=='__main__':
    # parameters
    parser = argparse.ArgumentParser(description='oneshot')
    parser.add_argument('--cmd', type=str, choices=['takeoff', 'land'], required=True, help='control command, \'takeoff\' or \'land\'')
    parser.add_argument('--id', type=int , nargs='+', default=[i+1 for i in range(hp_n_bot)], help='relevant robot ids (default: {})'.format([i+1 for i in range(hp_n_bot)]))
    args = parser.parse_args()

    if args.cmd=='takeoff':
        cmd = CMD_TAKEOFF
    else:
        cmd = CMD_LAND

    for id in args.id:
        assert id <= hp_n_bot

    # open serial port communication
    ser.open()
    # time.sleep(.1) # ensure serial port is ready

    # main loop
    while True:
        for id in args.id:
            sendCommand(id, cmd, 0., 0., 0., 0.) # robot is 1-idx
            time.sleep(1./hp_local_fps)

        # control fps
        time.sleep(1./hp_global_fps)


    # close serial port communication
    ser.flush() # ensure no remaining data in buffer before closing serial port
    ser.close() # what if the process is killed?


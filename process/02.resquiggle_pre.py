# -*- coding: utf-8 -*-

import argparse
import sys
import os

from tookit import Tookits

# args define
parser = argparse.ArgumentParser(description='Basecalling')
parser.add_argument('-f', '--fast5', required=True, help="directory of fast5 files")
parser.add_argument('-o', '--output', required=True, help="output directory")
args = parser.parse_args(sys.argv[1:])
global FLAGS
FLAGS = args


# from support import Tookits
tools = Tookits()

# # Convert merged single big fast5 into small size fast5 file
r = os.popen('find %s -name "*.fast5" ' % (FLAGS.fast5))  # 执行该命令
fls = r.readlines()
fls = [line.strip('\r\n') for line in fls]

fl = fls[0]
fsize = os.path.getsize(fl)
fsize = fsize/float(1024*1024)

if fsize > 10:

    print("multi_to_single_fast5...")
    single_out = FLAGS.output + "/single"

    cmd = "%s -i %s -s %s -t 40 --recursive" % (tools.multi_to_single_fast5, FLAGS.fast5, single_out)
    os.system(cmd)

    # create output directory
    dirs = single_out.split("/")
    dirs_list = []
    for i in range(len(dirs)):
        dirs_list.append("/".join(dirs[0:i + 1]))

    for item in dirs_list:
        if not os.path.exists(item):
            os.mkdir(item)

    FLAGS.fast5 = single_out


else:
    single_out = FLAGS.output + "/single"
    # create output directory
    dirs = single_out.split("/")
    dirs_list = []
    for i in range(len(dirs)):
        dirs_list.append("/".join(dirs[0:i + 1]))

    for item in dirs_list:
        if not os.path.exists(item):
            os.mkdir(item)
    # cmd = "cp %s/*.fast5 %s" % (FLAGS.fast5, single_out)
    cmd = "find %s -name  '*.fast5' | xargs -i cp {} %s" % (FLAGS.fast5, single_out)
    os.system(cmd)

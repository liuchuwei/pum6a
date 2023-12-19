# -*- coding: utf-8 -*-

import argparse
import sys
import os

from utils.tookit import Tookits

# args define
parser = argparse.ArgumentParser(description='Basecalling')
parser.add_argument('-i', '--input', required=True, help="directory of fast5 files")
parser.add_argument('-o', '--output', required=True, help="output directory")
args = parser.parse_args(sys.argv[1:])
global FLAGS
FLAGS = args

# basecalling
tools = Tookits()
cmd = tools.basecall + " -i " + FLAGS.input + " -c " + tools.model + " -s " + FLAGS.output + " -r " + \
      "--device cuda:0" + " --num_callers 16"

os.system(cmd)
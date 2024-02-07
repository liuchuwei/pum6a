# -*- coding: utf-8 -*-

import argparse
import sys
import os

from tookit import Tookits

# args define
parser = argparse.ArgumentParser(description='Basecalling')
parser.add_argument('-f', '--fast5', required=True, help="directory of fast5 files")
parser.add_argument('-fq', '--fastq', required=True, help="directory of fastq files")
parser.add_argument('-r', '--reference', required=True, help="directory of reference files")
parser.add_argument('-bam', '--bam', required=True, help="directory of bam files")
parser.add_argument('-o', '--output', required=True, help="output files")
args = parser.parse_args(sys.argv[1:])
global FLAGS
FLAGS = args

# merge fastq
print("merge fastq...")
pass_fastq = FLAGS.fastq
merge_fastq = "/".join(pass_fastq.split("/")[0:-1]) + "/merge.fastq"
pass_fastq = pass_fastq + "/*.fastq"
cmd = "cat %s > %s" % (pass_fastq, merge_fastq)
if not os.path.exists(merge_fastq):
    os.system(cmd)

#index samples
print("nanopolish index samples...")
from tookit import Tookits
tools = Tookits()
cmd = "%s index -d %s %s" % (tools.nanopolish, FLAGS.fast5, merge_fastq)

if not os.path.exists(merge_fastq+".index"):
    os.system(cmd)

#eventalign with samples
print("nanopolish eventalign samples...")

summary_fl = "/".join(FLAGS.fast5.split("/")[0:-1]) + "/sequencing_summary.txt"

# create output directory
dirs = FLAGS.output.split("/")
dirs_list = []
for i in range(len(dirs)):
    dirs_list.append("/".join(dirs[0:i]))

dirs_list = dirs_list[1:]
for item in dirs_list:
    if not os.path.exists(item):
        os.mkdir(item)


cmd = "%s eventalign --reads %s \
--bam %s \
--genome %s \
--scale-events  \
--signal-index  \
--summary %s \
--threads 40 > %s" % (tools.nanopolish, merge_fastq, FLAGS.bam, FLAGS.reference, summary_fl, FLAGS.output)

os.system(cmd)


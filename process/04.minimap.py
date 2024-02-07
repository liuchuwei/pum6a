# -*- coding: utf-8 -*-

import argparse
import sys
import os

from tookit import Tookits

# args define
parser = argparse.ArgumentParser(description='Basecalling')
parser.add_argument('-i', '--input', required=True, help="directory of fastq files")
parser.add_argument('-o', '--output', required=True, help="output directory")
parser.add_argument('-r', '--ref', required=True, help="path of reference")
args = parser.parse_args(sys.argv[1:])
global FLAGS
FLAGS = args

# create output directory
if not os.path.exists(FLAGS.output):
    os.mkdir(FLAGS.output)

# create transcript file & gene2transcript & fasta file
# if not 'cc_ref' in FLAGS.ref.split(".")[0]:
#     out_file = FLAGS.ref.split(".")[0] + "_sm.fa"
#     out_bed = FLAGS.ref.split(".")[0] + "_sm.bed"
#     if not os.path.exists(out_file):
#     # if os.path.exists(out_file):
#
#         f_open = open(FLAGS.ref, "rU")
#         FLAGS.ref = out_file
#
#         length_dictionary = {}
#         gene_info = []
#         count = 0
#         for rec in SeqIO.parse(f_open, "fasta"):
#             gene_info.append(rec.id.split("|"))
#             id = rec.id.split("|")[0]
#             seq = rec.seq
#             # length_dictionary[id] = len(seq)
#
#             with open(out_file, 'a') as f:
#                 f.write(">" + str(id) + "\n" + str(seq) + "\n")
#                 f.close()
#                 # id_file = open(out_file, "w")
#             with open(out_bed, 'a') as f:
#                 f.write(str(id) + "\t" + str(1) + "\t" + str(len(seq)) + "\n")
#                 f.close()
#                 # id_file = open(out_file, "w")
#
#             count += 1
#             # if count == 1000:
#             #     break
#
#         gene_info = pd.DataFrame(gene_info)
#         gene_info.to_csv("/".join(FLAGS.ref.split("/")[0:-1]) + "/gene_info.csv", sep="\t", index=False, header=0)
#         transcript = pd.DataFrame(gene_info.iloc[:,0])
#         transcript.to_csv("/".join(FLAGS.ref.split("/")[0:-1]) + "/transcript.txt", sep="\t", index=False, header=0)
#         gene2transcripts = gene_info.iloc[:,[5,0]]
#         gene2transcripts.to_csv("/".join(FLAGS.ref.split("/")[0:-1]) + "/gene2transcripts.txt", sep="\t", index=False, header=0)
#     else:
#         FLAGS.ref = out_file

# minimap
tools = Tookits()
cmd = tools.minimap2 + " -ax" + " map-ont" + " --MD" + " -t 16 " + FLAGS.ref + " " + FLAGS.input + "/*.fastq"+ \
      " >" + FLAGS.output +"/map.sam"
os.system(cmd)

# sam to bam & index
cmd = tools.samtools + " view -@ 16 -bh -F 2324 " + FLAGS.output +"/map.sam" + " | " + tools.samtools + " sort -@ 16 -o " + \
      FLAGS.output +"/map.bam"
os.system(cmd)

cmd =tools.samtools + " index " +  FLAGS.output +"/map.bam"
os.system(cmd)

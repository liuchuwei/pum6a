#!/usr/bin/env Python
# coding=utf-8
import argparse
import gzip
import multiprocessing
import os
import pickle
from argparse import ArgumentDefaultsHelpFormatter

from utils.ExtractSeqCurrent import extract_feature
from utils.Mapping import mapping
from utils.Merge import Merge_seq_current_dict, obtain_idsTiso, obtain_siteInfo, obtain_genoInfo, obtain_chromeInfo, obtain_chrome_site
from utils.AlignVariant import align_variant
from utils.SlideVariant import slide_per_site_var, slide_RRACH
from tqdm import tqdm
from collections import defaultdict

def argparser():
    parser = argparse.ArgumentParser(
        formatter_class=ArgumentDefaultsHelpFormatter,
        add_help=False
    )
    # parser.add_argument('--mod', required=True, help='Possible value:Extract(extract seq and current information); Merge(merge seq and current information)')
    parser.add_argument('--single', required=True, help='Single fast5 path')
    parser.add_argument('--kmer', default='5', help='Length of kmer')
    parser.add_argument('--kmer_filter', default='[AG][AG]AC[ACT]', help='Define kmer filter')
    parser.add_argument('--basecall_group', default="RawGenomeCorrected_000",
                        help='The attribute group to extract the training data from. e.g. RawGenomeCorrected_000')
    parser.add_argument('--basecall_subgroup', default='BaseCalled_template',
                        help='Basecall subgroup Nanoraw resquiggle into. Default is BaseCalled_template')
    parser.add_argument('--clip', default=10, help='Reads first and last N base signal drop out')
    parser.add_argument('-o', '--output', required=True, help="Output directory")
    parser.add_argument('-g', '--genome', required=True, help="Genome file for mapping")
    parser.add_argument('-r', '--reference', required=True, help="Referance transcripts sequence file")
    parser.add_argument('-i', '--isoform', required=True, help="Gene to referance transcripts information")
    parser.add_argument('-b', '--bam', required=True, help="Path of bam file")
    parser.add_argument('--cpu', default=8, help='cpu number usage,default=8')
    parser.add_argument('--support', default=10,
                        help='The minimum number of DRS reads supporting a modified m6A site in genomic coordinates from one million DRS reads.  The default is 10.  Due to the low sequencing depth for DRS reads, quantification of m6A modification in low abundance gene is difficult.  With this option, the pipeline will attempt to normalize library using this formula: Total number of DRS reads/1,000, 000 to generate \'per million scaling factor\'.   Then the  \'per million scaling factor\'  multiply reads from -r option to generate the cuttoff for the number of modified transcripts  for each modified m6A site.   For example, the option (-r = 10, total DRS reads=2, 000, 000) will generate (2000000/1000000)*10=20 as cuttoff. Than means that modified A base supported by at least 20 modified transcripts will be identified as modified m6A sites in genomic coordinates.')

    return parser


def main(args):

    # if args.mod == "Extract":
    'main funtion for preprocess'
    '1.Get path of single fast5 files'
    print("Get path of single fast5 files...")
    fls =  [args.single + "/" + item for item in os.listdir(args.single)]
    r = os.popen('find %s -name "*.fast5" ' % (args.single))  # 执行该命令
    fls = r.readlines()
    fls = [line.strip('\r\n') for line in fls]

    '2.Extract seq & current information'
    print("Extract seq & current information...")
    pool1 = multiprocessing.Pool(processes = int(args.cpu))

    results=[]
    for fl in fls:
        result=pool1.apply_async(extract_feature,(fl,args))
        results.append(result)
    pool1.close()

    pbar = tqdm(total=len(fls), position=0, leave=True)
    nums = []
    for result in results:
        num, seq = result.get()
        if num and seq:
            nums.append([num, seq])
        pbar.update(1)
    pool1.join()

    dirs = args.output.split("/")
    dirs_list = []
    for i in range(len(dirs)):
        dirs_list.append("/".join(dirs[0:i + 1]))

    for item in dirs_list[:-1]:
        if not os.path.exists(item):
            os.mkdir(item)

    output = open(args.output + ".feature.fa", "w")
    output.write("".join([str(x[1]) for x in nums]))
    output.close()

    # output = open(args.output + ".tmp.tsv", "w")
    # output.write("".join([str(x[0]) for x in nums]))
    # output.close()

    '3.Mapping with genome'
    print("Mapping...")
    basefl = '/'.join(args.output.split("/")[:-1])
    site_path = "%s/extract.reference.isoform.bed12" % (basefl)
    if not os.path.exists(site_path):
        mapping(args, nums)

    '4.Extract align variant information'
    print("Extract align variant information...")
    site_path = args.bam.split("/")[:-1]
    site_path.append("map.plus_strand.per.site.csv")
    site_path = "/".join(site_path)
    if not os.path.exists(site_path):
        align_variant(args)


    # elif args.mod == "Merge":
    '5.Merge RRACH seq & current information'
    print("Merge RRACH seq & current information...")

    # ---------------------------------------------
    ## chrome
    if not os.path.exists('%s.chrome.pkl' % (args.output)):
        chrome = obtain_chrome_site(nums, args)
        f_save = open('%s.chrome.pkl' % (args.output), 'wb')
        pickle.dump(chrome, f_save)
        f_save.close()


    ## isoform
    if not os.path.exists('%s.iso.pkl' % (args.output)):
        iso = obtain_idsTiso(args)
        f_save = open('%s.iso.pkl' % (args.output), 'wb')
        pickle.dump(iso, f_save)
        f_save.close()

    ## gene
    if not os.path.exists('%s.gene.pkl' % (args.output)):
        gene = obtain_genoInfo(args)
        f_save = open('%s.gene.pkl' % (args.output), 'wb')
        pickle.dump(gene, f_save)
        f_save.close()

    # slide for RRACH
    if not os.path.exists('%s.slide.pkl' % (args.output)):
        siteInfo = slide_RRACH(site_path, int(args.kmer))
        f_save = open('%s.slide.pkl' % (args.output), 'wb')
        pickle.dump(siteInfo, f_save)
        f_save.close()


    # ---------------------------------------------
    ## chrome
    f_read = open('%s.chrome.pkl' % (args.output), 'rb')
    chrome_info = multiprocessing.Manager().dict(pickle.load(f_read))
    # chrome_info = pickle.load(f_read)

    ## isoform
    f_read = open('%s.iso.pkl' % (args.output), 'rb')
    iso =  multiprocessing.Manager().dict(pickle.load(f_read))
    # iso =  pickle.load(f_read)

    ## gene
    f_read = open('%s.gene.pkl' % (args.output), 'rb')
    gene =  multiprocessing.Manager().dict(pickle.load(f_read))
    # gene = pickle.load(f_read)

    ## site
    f_read = open('%s.slide.pkl' % (args.output), 'rb')
    siteInfo =  multiprocessing.Manager().dict(pickle.load(f_read))
    # siteInfo =  pickle.load(f_read)
    # ---------------------------------------------
    def init_lock(l):
        global lock
        lock = l
    #
    l = multiprocessing.Lock()
    pool2 = multiprocessing.Pool(processes = int(args.cpu), initializer=init_lock, initargs=(l, ))
    # pool2 = multiprocessing.Pool(processes = int(args.cpu))

    results = []
    for fl in nums:
        result = pool2.apply_async(Merge_seq_current_dict, (fl, args, chrome_info, iso, gene, siteInfo))
        # result = pool.apply_async(Merge_seq_current_grep_dict, (fl, args))
        results.append(result)
    pool2.close()

    pbar = tqdm(total=len(nums), position=0, leave=True)
    meta = []
    for result in results:
        lines = result.get()
        if lines:
            meta.append(lines)
        pbar.update(1)
    pool2.join()

    output = open(args.output + ".feature.tsv", "w")
    output.write("".join([x for x in meta]))
    output.close()
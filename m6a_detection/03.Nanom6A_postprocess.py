import argparse
import pickle
import sys
from collections import defaultdict
from tqdm import tqdm
import numpy as np
import pandas as pd

def load_bed(args):
    fl = args.bed
    bed = defaultdict(dict)
    for i in open(fl, "r"):
        ele = i.rstrip().split()
        reads, pos = ele[2].split("|")[:2]
        pos = int(pos)
        bed[reads][pos]=ele[1]

    return bed

def load_reference(args):

    ref = args.reference
    ## chrome
    f_read = open('%s.chrome.pkl' % (ref), 'rb')
    # chrome_info = multiprocessing.Manager().dict(pickle.load(f_read))
    chrome_info = pickle.load(f_read)

    ## isoform
    f_read = open('%s.iso.pkl' % (ref), 'rb')
    # iso = multiprocessing.Manager().dict(pickle.load(f_read))
    iso = pickle.load(f_read)

    ## gene
    f_read = open('%s.gene.pkl' % (ref), 'rb')
    # gene = multiprocessing.Manager().dict(pickle.load(f_read))
    gene = pickle.load(f_read)

    ## site
    f_read = open('%s.slide.pkl' % (ref), 'rb')
    # siteInfo = multiprocessing.Manager().dict(pickle.load(f_read))
    siteInfo = pickle.load(f_read)

    return chrome_info, iso, gene, siteInfo

def Merge(bed, chrome_info, iso, gene, siteInfo):

    prob_site = defaultdict(dict)
    pbar = tqdm(total=len(bed.keys()), position=0, leave=True)
    for rds in bed.keys():
        pos = list(bed[rds].keys())
        for sub_pos in pos:
            if sub_pos in list(chrome_info[rds].keys()):
                try:
                    chrom = chrome_info[rds][sub_pos]
                    ts = iso[rds]
                    gs = gene[rds]
                    site_pos = int(sub_pos) + int(gs[1]) + 1
                    motif = str(int(site_pos) - 2) + "-" + str(int(site_pos) + 2)
                    motif = siteInfo[ts][motif].split("|")[0]
                    site= chrom + "|" + motif
                    if len(prob_site[site])==0:
                        prob_site[site] = [float(bed[rds][sub_pos])]
                    else:
                        prob_site[site].append(float(bed[rds][sub_pos]))
                except:
                    continue
        pbar.update(1)

    return prob_site

def GetOutput(args, prob_site):
    output = args.output
    min_read = int(args.min_read)
    probs = []
    sites = []
    for item in prob_site:
        reads = prob_site[item]
        if len(reads) >= min_read:
            sites.append(item)
            probs.append(np.array(reads, dtype=np.float).max())
    site_info = pd.DataFrame({'site': np.array(sites), 'pro': np.array(probs)})
    site_path = output
    site_info.to_csv(site_path, index=False)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Extract fast5 files.')
    parser.add_argument('-o', '--output', required=True, help="Output file")
    parser.add_argument('-bed', '--bed', required=True, help="Tombo bed file")
    parser.add_argument('-ref', '--reference', required=True, help="Reference directory: directory of pum6a result")
    parser.add_argument('-min_read', '--min_read', required=True, help="Reference directory: directory of pum6a result")

    FLAGS = parser.parse_args(sys.argv[1:])

    global args
    args = FLAGS
    print("loading bed...")
    bed = load_bed(args)
    print("loading reference...")
    chrome_info, iso, gene, siteInfo = load_reference(args)
    print("merging...")
    prob_site = Merge(bed, chrome_info, iso, gene, siteInfo)
    GetOutput(args, prob_site)


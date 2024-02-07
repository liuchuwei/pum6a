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
        ele = i.rstrip().split() # 'chrom', 'start_loc', 'end_loc', 'strand', 'name', 'ref', 'homo_seq', 'kmer5', 'majorAllel', 'majorAllelFreq', 'kmer7', 'test_err_1', 'model_err_1', 'test_cor_1', 'model_cor_1', 'oddR', 'pval', 'adjPval', 'baseExt', 'total_reads', 'ESB_test', 'ESB_ctrl'
        if not ele[0] == 'chrom':
            if ele[7] in ["AAACA", "AAACT", "AGACC", "GAACA", "GAACT", "GGACC", "AAACC", "AGACA", "AGACT", "GAACC", "GGACA", "GGACT"]:
                if int(ele[-3])>=int(args.min_read):
                    site = ele[2]
                    motif = str(int(site) - 2) + "-" + str(int(site) + 2)
                    bed[ele[0]][motif]=ele[-5]
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

    rev_iso = defaultdict(dict)
    for key in iso:
        if len(rev_iso[iso[key]])==0:
            rev_iso[iso[key]] = [key]
        else:
            rev_iso[iso[key]].append(key)

    ## gene
    f_read = open('%s.gene.pkl' % (ref), 'rb')
    # gene = multiprocessing.Manager().dict(pickle.load(f_read))
    gene = pickle.load(f_read)

    ## site
    f_read = open('%s.slide.pkl' % (ref), 'rb')
    # siteInfo = multiprocessing.Manager().dict(pickle.load(f_read))
    siteInfo = pickle.load(f_read)

    return chrome_info, rev_iso, gene, siteInfo

def Merge(bed, chrome_info, rev_iso, gene, siteInfo):

    ## Merge
    MergeSite = list(set(list(siteInfo.keys())).intersection(set(list(bed.keys()))))
    prob_site = defaultdict(dict)
    pbar = tqdm(total=len(MergeSite), position=0, leave=True)
    for ens in MergeSite:
        bed_motif = list(bed[ens].keys())
        site_motif = list(siteInfo[ens].keys())
        inter_motif = list(set(site_motif).intersection(set(bed_motif)))
        pbar.update(1)

        for motif in inter_motif:
            prob = bed[ens][motif]
            site = siteInfo[ens][motif]
            read = rev_iso[ens]
            for re in read:
                pos = int(motif.split("-")[0])+2-1-int(gene[re][1])
                try:
                    chrome = chrome_info[re][pos] +"|"+site.split("|")[0]
                    if len(prob_site[chrome]) == 0:
                        prob_site[chrome] = [prob]
                    else:
                        prob_site[chrome].append(prob)

                except:
                    continue

    return prob_site

def GetOutput(args, prob_site):
    output = args.output
    probs = []
    sites = []
    for item in prob_site:
        reads = prob_site[item]
        sites.append(item)
        probs.append(np.float32(-np.log10(np.array(reads, dtype=np.float32).min()+10e-320)))
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

    bed = load_bed(args)
    chrome_info, rev_iso, gene, siteInfo = load_reference(args)
    prob_site = Merge(bed, chrome_info, rev_iso, gene, siteInfo)
    GetOutput(args, prob_site)


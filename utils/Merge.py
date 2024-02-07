#!/usr/bin/env Python
# coding=utf-8
import gzip
import os
import pickle

from collections import defaultdict
from tqdm import tqdm


def obtain_idsTiso(args):
    basefl = '/'.join(args.output.split("/")[:-1])
    fl = "%s/extract.reference.isoform.bed12" % (basefl)
    # idsTiso = defaultdict(dict)
    idsTiso = {}
    for i in open(fl, "r"):
        ele = i.rstrip().split()
        idsTiso[ele[3]] = ele[0]
        # if not idsTiso[ele[0]]:
        #     idsTiso[ele[0]] = [ele[3]]
        # else:
        #     idsTiso[ele[0]].append(ele[3])
    return idsTiso


def obtain_siteInfo(args):
    # basefl = '/'.join(args.output.split("/")[:-1])
    basefl = '/'.join(args.bam.split("/")[:-1])
    fl = "%s/map.plus_strand.per.site.csv" % (basefl)
    # fl = "%s/map.plus_strand.per.site.%smer.csv" % (basefl, args.kmer)
    siteInfo = defaultdict(dict)
    for i in open(fl, "r"):
        if i.startswith("#"):
            pre1 = "#"
            continue
        ele = i.rstrip().split(',')
        # Kmer,Window,Ref,Strand,Coverage,q1,q2,q3,q4,q5,mis1,mis2,mis3,mis4,mis5,ins1,ins2,ins3,ins4,ins5,del1,del2,del3,del4,del5
        siteInfo[ele[2]][ele[1]] = ele

    return siteInfo


def obtain_genoInfo(args):
    basefl = '/'.join(args.output.split("/")[:-1])
    fl = "{0}/extract.reference.bed12".format(basefl)
    readgene = {}
    for i in open(fl, "r"):
        ele = i.rstrip().split()
        readgene[ele[3]] = [ele[0], ele[1], ele[2]]
    return readgene

def obtain_chromeInfo(args):
    basefl = '/'.join(args.output.split("/")[:-1])
    fl = "{0}/extract.bed12".format(basefl)
    readgene = {}
    for i in open(fl, "r"):

        ele = i.rstrip().split()
        readgene[ele[3]] = [ele[0], ele[1], ele[2], ele[5]]
    return readgene

def obtain_chrome_site(nums, args):

    print("obtain_chrome_site")
    readsid = defaultdict(dict)
    seq = defaultdict(dict)
    for item in nums:

        cur = item[0].split("\n")
        for it in cur[:-1]:
            pos = it.split("\t")[0]
            id = pos.split("|")[0]
            pos = pos.split("|")[1]
            motif = it.split("\t")[2]
            readsid[id][int(pos)] = motif

        id = item[1].split("\n")[0].strip(">")
        seq[id] = item[1].split("\n")[1]

    basefl = '/'.join(args.output.split("/")[:-1])
    fl = "%s/extract.sort.bam.tsv.gz" % (basefl)
    chrome_info = defaultdict(dict)
    pbar = tqdm(total=len(readsid.keys()), position=0, leave=True)
    pre1 = ""
    for i in gzip.open(fl, "r"):
        i = i.decode("utf-8").rstrip()

        if i.startswith("#"):
            pre1 = "#"
            continue

        ele = i.rstrip().split()

        if ele[3] == "." or ele[6] == ".":
            continue

        ids, chro, idspos, gpos = ele[0], ele[2], int(ele[3]), ele[6]

        if ids != pre1:
            pbar.update(1)
            pre1 = ids

        if ele[1] == "0":
            strand = "+"
        elif ele[1] == "16":
            strand = "-"
            lens = len(seq[ids])
            idspos = lens - idspos - 1

        if ids in readsid and idspos in readsid[ids]:
            chrome_info[ids][idspos] = chro + "|" + gpos + "|" + strand

    return chrome_info

def Merge_seq_current_dict(fl, args, chrome_info, iso, gene, siteInfo):
    # def Merge_seq_current_grep_dict(fl, args):


    try:
        eles = fl[0].rstrip().split("\n")
        lines = []
        for item in eles:

            item = item.split("\t")
            ids = item[0].split("|")[0]
            pos = item[0].split("|")[1]

            ## chorme
            item.append(chrome_info[ids][int(pos)])

            ## gene
            item.append(iso[ids] + "|" + "|".join(gene[ids]))

            site = int(pos) + int(gene[ids][1]) + 1
            ## site
            if args.kmer == '5':
                motif = str(int(site) - 2) + "-" + str(int(site) + 2)
            if args.kmer == '6':
                motif = str(int(site) - 3) + "-" + str(int(site) + 2)
            if args.kmer == '7':
                motif = str(int(site) - 3) + "-" + str(int(site) + 3)
            if args.kmer == '9':
                motif = str(int(site) - 4) + "-" + str(int(site) + 4)

            if iso[ids] in siteInfo and motif in siteInfo[iso[ids]]:

                item.append(siteInfo[iso[ids]][motif])
                line = "\t".join(item) + "\n"
                lines.append(line)

        return "".join(lines)

    except:
        return None



def Merge_seq_current_grep_dict(fl, args, chrome_info, iso, gene):
# def Merge_seq_current_grep_dict(fl, args):

    try:
        ele = fl.rstrip().split()
        ids = ele[0].split("|")[0]
        pos = ele[0].split("|")[1]

        ## chorme
        ele.append("|".join(chrome_info[ids]))

        ## gene
        ele.append(iso[ids] + "|" + "|".join(gene[ids]))

        ## site
        iso = gene[ids][1]
        basefl = '/'.join(args.bam.split("/")[:-1])
        fl = "%s/map.plus_strand.per.site.csv" % (basefl)
        site_var = []
        for i in range(int(args.kmer)):
            transPos = iso[0] + "," + str(int(pos) + int(iso) + i - int(args.kmer) // 3)
            cmd = "grep %s %s -m 1" % (transPos, fl)
            p = os.popen(cmd)
            site_var.append(p.read().strip().split(","))


        site_motif = "".join([item[2] for item in site_var])
        site_pos = site_var[0][1] + "-" + site_var[-1][1]
        site_qmean = ",".join([item[5] for item in site_var])
        mis = ",".join([item[8] for item in site_var])
        ins = ",".join([item[9] for item in site_var])
        _del = ",".join([item[10] for item in site_var])

        ele.append("%s|%s|%s|%s|%s|%s" % (site_motif, site_pos, site_qmean, mis, ins, _del))

        lines = "\t".join(ele) + "\n"
        return lines

    except:
        return None



def Merge_seq_current_grep(fl, args):

    try:
        ele = fl.rstrip().split()
        ids = ele[0].split("|")[0]
        pos = ele[0].split("|")[1]
        basefl = '/'.join(args.output.split("/")[:-1])

        ## chrome
        fl = "{0}/extract.bed12".format(basefl)
        cmd = "grep %s %s" % (ids, fl)
        p = os.popen(cmd)
        chorme = p.read().strip().split("\t")
        ele.append("|".join(chorme[0:3]))

        ## isoform
        fl = "%s/extract.reference.isoform.bed12" % (basefl)

        cmd = "grep %s %s" % (ids, fl)
        p = os.popen(cmd)
        iso = p.read().strip().split("\t")


        ## gene
        fl = "{0}/extract.reference.bed12".format(basefl)
        cmd = "grep %s %s" % (ids, fl)
        p = os.popen(cmd)
        gene = p.read().strip().split("\t")

        ele.append(iso[0] + "|" + "|".join(gene[0:3]))

        ## site
        basefl = '/'.join(args.bam.split("/")[:-1])
        fl = "%s/map.plus_strand.per.site.csv" % (basefl)
        site_var = []
        for i in range(int(args.kmer)):
            transPos = iso[0] + "," + str(int(pos) + int(iso[1]) + i - int(args.kmer) // 3)
            cmd = "grep %s %s -m 1" % (transPos, fl)
            p = os.popen(cmd)
            site_var.append(p.read().strip().split(","))


        site_motif = "".join([item[2] for item in site_var])
        site_pos = site_var[0][1] + "-" + site_var[-1][1]
        site_qmean = ",".join([item[5] for item in site_var])
        mis = ",".join([item[8] for item in site_var])
        ins = ",".join([item[9] for item in site_var])
        _del = ",".join([item[10] for item in site_var])

        ele.append("%s|%s|%s|%s|%s|%s" % (site_motif, site_pos, site_qmean, mis, ins, _del))

        lines = "\t".join(ele) + "\n"
        return lines

    except:
        return None



# def Merge_seq_current(fl, idsTiso, readgene, siteInfo, args):
def Merge_seq_current(fl, args):

    ## 1.ids to isoform
    idsTiso = obtain_idsTiso(args)

    ## 2.site information
    siteInfo = obtain_siteInfo(args)

    ## 3.geno information
    readgene = obtain_genoInfo(args)

    ## 4.chrome information
    readchrom = obtain_chromeInfo(args)

    ## 4.merge information
    # fl = "".join([str(x[0]) for x in nums])
    # fl = fl.split("\n")
    # for i in fl[:-1]:
    # ele = i.rstrip().split()
    try:
        ele = fl.rstrip().split()
        ids = ele[0].split("|")[0]
        pos = ele[0].split("|")[1]
        isoform = idsTiso[ids]
        genemap = readgene[ids]
        site = str(int(pos) + int(genemap[1]) + 1)
        if args.kmer == '5':
            motif = str(int(site) - 2) + "-" + str(int(site) + 2)
        if args.kmer == '6':
            motif = str(int(site) - 3) + "-" + str(int(site) + 2)
        if args.kmer == '7':
            motif = str(int(site) - 3) + "-" + str(int(site) + 3)
        if args.kmer == '9':
            motif = str(int(site) - 4) + "-" + str(int(site) + 4)
    except:
        return None

    if isoform in siteInfo and motif in siteInfo[isoform]:
        align_event = siteInfo[isoform][motif]
        ele.append("|".join(align_event))  # base, strand, cov, q_mean, q_median, q_std, mis, ins, del
        ele.append("|".join(genemap))
        lines = "\t".join(ele) + "\n"
        return lines
    else:
        return None

def MergeSeqCurrent_single_motif(args, nums):
    'Merege sequence and current information'

    basefl = '/'.join(args.output.split("/")[:-1])
    storepos = defaultdict(dict)
    fl = "".join([str(x[0]) for x in nums])
    fl = fl.split("\n")
    for i in fl[:-1]:
        # ~ c9a0d84d-42f6-456c-895f-e957d1172623|6|AAGGGAAAGACTCCAGAGGAAATTAGGAAGACCTTTAACATCAAGAATGACTTTACACCTGAGGAGGAGGAGGAAGTTCGCCGTGAGAACCAGTGGGCATTTGAATGAAGTGCGTCTGATGGTTTCATGGAAGGAATGTTGTTCTAATGCCAAATGAATGCTGTGGGTTATCTTAGCGTAGACAAGACTATGTTTCTATGACTTTATTGTGAACCTGTGAGCACATTGACTGTAAATAATACTTGTATTCTGGGGAGGGGATTGGTAGTAGTTTCCTGCAATCAATCCTCTGCTTGTGGGCAAATGTTATTTGTTGCAGACTTGCAGTGATCCTTATCTGTTGTATCTGTTTTCCCTCTGTGTTCCTGCCAAGTTTGTTTCTTGGACATAATCATCAAGTCTTGGTGTCTCTT	1.0	0.06883740425109862	0.9311625957489014
        # ~ 0.09221864	0.90778136	GXB01149_20180715_FAH87828_GA10000_sequencing_run_20180715_NPL0183_I1_33361_read_17200_ch_182_strand.fast5|177|2,1,3,2,0,1,3,2,0,3,1,1,0,0,0,3,1,1,3,2,0,3,2,0,1,1,1,3,1,3,3,2,3,0,1,1,0,2,0,2,0,3,3,2,1,1,1,0,3,0,3,2,3,0,1,0,0,2,0,1,3,2,0,1,1,2,0,2,1,1,0,0,0,3,0,3,2,0,0,2,1,1,0,1,3,2,1,3,1,2,1,0,2,3,3,2,2,0,1,0,1,0,2,0,0,2,3,0,3,2,1,1,0,3,2,2,2,1,3,2,0,2,2,1,3,2,0,2,2,3,3,0,3,2,2,0,0,0,1,0,0,3,2,3,0,3,0,0,0,3,2,2,1,3,3,2,1,2,3,3,3,0,3,3,2,3,3,0,3,2,3,2,3,3,2,0,0,0,1,0,3,2,2,3,1,3,2,3,3,3,0,1,3,1,3,3,3,3,2,2,2,2,3,3,2,2,3,3,3,3,2,3,2,0,2,2,2,3,3,3,2,0,0,3,3,3,1,0,3,0,0,2,0,0,3,2,0,0,3,2,0,3,0,3,3,3,1,2,3,2,1,0,2,1,3,1,1,0,0,0,1,3,0,3,2,0,3,3,3,2,2,2,2,2,3,3,2,0,0,3,2,2,0,0,0,3,0
        ele = i.rstrip().split()
        mark = ele[0]
        sig = ele[1:]
        ids, spos, seq = mark.split("|")
        # ~ ids=namechange[ids]
        # ~ print(ids,int(spos))
        storepos[ids][int(spos)] = sig

    fl = args.output + ".feature.fa"
    store = {}
    lines = open(fl, "r").readlines()
    for index, i in enumerate(lines):
        if i.startswith(">"):
            ids = i.rstrip().lstrip(">")
            read = lines[index + 1].rstrip()
            store[ids] = read

    fl = "{0}/extract.reference.bed12".format(basefl)

    readgene = {}
    for i in open(fl, "r"):
        ele = i.rstrip().split()
        readgene[ele[3]] = ele[0]
    fl = "%s/extract.sort.bam.tsv.gz" % (basefl)
    ##########################################
    ##########################################
    # ~ chr04	W_003002_20180416_FAH83697_MN23410_sequencing_run_20180415_FAH83697_mRNA_WT_Col0_2918_23801_read_57_ch_290_strand.fast5	-	10201753	10201753	236|10203065|GAACA	275|10202917|AGACC	991|10202201|GGACA	1003|10202189|AGACC	1373|10201819|AAACA
    # ~ c1,c2,c3=0,0,0

    pbar = tqdm(total=len(store.keys()), position=0, leave=True)
    pre1, pre2 = "", ""
    results = []
    for i in gzip.open(fl, "r"):
        i = i.decode("utf-8").rstrip()

        # ~ NR_002323.2	0	chr22	7541	A	I	31375381	a	M
        if i.startswith("#"):
            pre1 = "#"
            continue
        ele = i.rstrip().split()

        if ele[3] == "." or ele[6] == ".":
            continue

        align = [0, 0, 0, 0, 0]  # mat,mis,ins,del,qual
        if ele[-1] in ['M', 'm']:
            align[4] = ord(ele[-4]) - 33
            if (ele[-2] != ele[4]):
                align[1] = 1
            else:
                align[0] = 1

        if ele[-1] == 'D':
            align[3] = 1

        if ele[-1] == 'I':
            align[2] = 1

        ids, chro, idspos, gpos = ele[0], ele[2], int(ele[3]), ele[6]

        if ids != pre1:
            pbar.update(1)
            pre1 = ids

        if ele[1] == "0":
            strand = "+"

        elif ele[1] == "16":
            strand = "-"
            lens = len(store[ids])
            idspos = lens - idspos - 1

        if ids in storepos and idspos in storepos[ids] and ids in readgene:
            # kmer = store[ids][idspos - 2:idspos + 3]
            # line = "%s|%s|%s" % (idspos, gpos, kmer)
            # total_m6A_reads["%s\t%s\t%s\t%s\tNA\t" % (chro, ids, strand, readgene[ids])][line] = 1

            # ids|chro|strand|idspos|gpos|gene, base|kmer, mean, std, md_intense, length, align
            lines = "%s|%s|%s|%s|%s|%s\t%s|%s\t%s\t%s\t%s\t%s\t%s\n" % (ids, chro, strand, idspos, gpos, readgene[ids],
                                                                        storepos[ids][idspos][0],
                                                                        storepos[ids][idspos][1],
                                                                        storepos[ids][idspos][2],
                                                                        storepos[ids][idspos][3],
                                                                        storepos[ids][idspos][4],
                                                                        storepos[ids][idspos][5],
                                                                        "|".join(str(item) for item in align))
            results.append(lines)

    return results
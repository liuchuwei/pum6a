#!/usr/bin/env python
# -*- coding: utf-8 -*-
'Modify from: https://github.com/enovoa/EpiNano'


import os
import re
import sys
from collections import OrderedDict
from collections import defaultdict
from tqdm import tqdm

def print_last_consecutive_lines(lines, outfh):
    contents = OrderedDict()
    for line in lines:
        ary = line.strip().split(',')
        ref, pos, strand = (ary[0], ary[1], ary[3])
        contents[(ref, pos, strand)] = line.rstrip()

    win = len(lines)
    middle = lines[win // 2].rstrip().split(',')
    window = str(int(middle[1]) - win // 2) + '-' + str(int(middle[1]) + win // 2)
    kmer = ''

    consecutive_lines = []
    ref, pos, base, strand = middle[:4]
    for i in reversed(list(range(1, win // 2 + 1))):
        k = (ref, str(int(pos) - i), strand)
        relative_pos = '-' + str(i)
        if k in contents:
            kmer = kmer + contents[k].split(',')[2]
            consecutive_lines.append(window + ',' + relative_pos + ',' + contents[k])
        else:
            kmer = kmer + 'N'
            consecutive_lines.append(window + ',' + relative_pos + ',' + "," + ",".join(['NA'] * 12))

    consecutive_lines.append(window + ',' + '+0' + ',' + ",".join(middle))
    kmer = kmer + middle[2]
    for i in range(1, win // 2 + 1):
        k = (ref, str(int(pos) + i), strand)
        relative_pos = '+' + str(i)
        if k in contents:
            kmer = kmer + contents[k].split(',')[2]
            consecutive_lines.append(window + ',' + relative_pos + ',' + contents[k])
        else:
            kmer = kmer + 'N'
            consecutive_lines.append(window + ',' + relative_pos + ',' + ",".join(['NA'] * 12))
    for l in consecutive_lines:
        print(kmer + ',' + l, file=outfh)


def slide_RRACH(per_site_var, win=5):

    fh = open(per_site_var, 'rb')
    eof = fh.seek(-1, 2)
    fh.seek(0, 0)
    head = fh.readline()
    lines = []

    head_num = fh.tell()
    for _ in range(win):
        l = fh.readline().decode('utf-8').rstrip().split(",")
        if l:
            lines.append(l)
    line_num = (fh.tell() - head_num)/5
    total_num = int((eof - head_num)//line_num)

    kmer_fillter = "[AG][AG]AC[ACT]"
    siteInfo = defaultdict(dict)


    # par = tqdm(total = eof//(fh.tell()/4))

    print("Extract RRACH matching variant")
    # total_num = 10000
    for i in tqdm(range(total_num),position=0):
    # while (fh.tell() <= eof):
        try:
            motif = "".join([item[2] for item in lines])
        except:
            continue

        if re.search(kmer_fillter, motif):
            trans = [item[0] for item in lines]
            if len(set(trans)) == 1:
                site_motif = "".join([item[2] for item in lines])
                site_pos = lines[0][1] + "-" + lines[-1][1]
                site_qmean = ",".join([item[5] for item in lines])
                mis = ",".join([item[8] for item in lines])
                ins = ",".join([item[9] for item in lines])
                _del = ",".join([item[10] for item in lines])
                trans = lines[0][0]
                siteInfo[trans][site_pos] = "%s|%s|%s|%s|%s|%s" % (site_motif, site_pos, site_qmean, mis, ins, _del)

        lines = lines[1:]
        new_line = fh.readline().decode('utf-8').rstrip().split(",")
        lines.append(new_line)
        # par.update(1)
    # par.close()

    return siteInfo


def slide_per_site_var(per_site_var, win=5):
    '''
    #Ref,pos,base,strand,cov,q_mean,q_median,q_std,mis,ins,del

    cc6m_2244_T7_ecorv,7,A,+,1.0,15.0,15.0,0.0,0.0,0.0,0.0
    kmer sequences will be reversed if reads aligned on the minus strand
    bases mapped to reverse strand have alredy been complemented during above processing
    '''
    # Ref,pos,base,strand,cov,q_mean,q_median,q_std,mis,ins,del
    prefix = re.sub(r'.per.site.\S+', '',
                    per_site_var)  # , .replace ('.per.site.csv','') # ".".join (per_site_var.split('.')[:-1])
    out_tmp = prefix + '.per_site_var.{}mer.tmp'.format(win)
    if os.path.exists(out_tmp):
        os.remove(out_tmp)
    outfh = open(out_tmp, 'w')

    fh = open(per_site_var, 'rb')
    eof = fh.seek(-1, 2)
    fh.seek(0, 0)
    head = fh.readline()
    lines = []

    for _ in range(win):
        l = fh.readline().decode('utf-8').rstrip()
        if l:
            lines.append(l)
    if len(lines) < win:
        print('not enough sites to be slided', file=sys.stderr)

    contents = OrderedDict()

    for line in lines:
        ary = line.strip().split(',')
        ref, pos, strand = (ary[0], ary[1], ary[3])
        contents[(ref, pos, strand)] = line.rstrip()

    while (fh.tell() <= eof):
        middle = lines[win // 2].split(',')
        window = str(int(middle[1]) - win // 2) + '-' + str(int(middle[1]) + win // 2)
        consecutive_lines = []
        kmer = ''
        ref, pos, base, strand = middle[:4]
        k_to_del = (ref, str(int(pos) - win), strand)
        for i in reversed(list(range(1, win // 2 + 1))):
            k = (ref, str(int(pos) - i), strand)
            relative_pos = '-' + str(i)
            if k in contents:
                kmer = kmer + contents[k].split(',')[2]
                consecutive_lines.append(window + ',' + relative_pos + ',' + contents[k])
            else:
                consecutive_lines.append(window + ',' + relative_pos + ',' + ",".join(
                    [ref, str(int(pos) - i), 'N', strand, '0', 'NaN,NaN,NaN,NaN,NaN,NaN']))
                kmer = kmer + 'N'
        consecutive_lines.append(window + ',+0' + ',' + ",".join(middle))
        kmer = kmer + middle[2]
        for i in range(1, win // 2 + 1):
            k = (ref, str(int(pos) + i), strand)
            relative_pos = '+' + str(i)
            if k in contents:
                kmer = kmer + contents[k].split(',')[2]
                consecutive_lines.append(window + ',' + relative_pos + ',' + contents[k])
            else:
                kmer = kmer + 'N'
                consecutive_lines.append(window + ',' + relative_pos + ',' + ",".join(
                    [ref, str(int(pos) + i), 'N', strand, '0', 'NaN,NaN,NaN,NaN,NaN,NaN']))
            # consecutive_lines.append (window+','+relative_pos+','+ "," . join (['NaN']*11))

        for l in consecutive_lines:
            print(kmer + ',' + l, file=outfh)
        keys = list(contents.keys())
        del consecutive_lines
        if k_to_del in contents:
            del contents[k_to_del]
        lines = lines[1:]
        new_line = fh.readline().decode('utf-8').rstrip()
        lines.append(new_line)
        ref, pos, base, strand = new_line.split(',')[:4]
        contents[(ref, pos, strand)] = new_line
    print_last_consecutive_lines(lines, outfh)
    outfh.close()

    # out2 = prefix + '.per_site.{}mer.csv'.format(win)
    out2 = prefix + '.per.site.{}mer.csv'.format(win)
    outh2 = open(out2, 'w')
    q_in_head = ",".join(["q{}".format(i) for i in range(1, win + 1)])
    mis_in_head = ",".join(["mis{}".format(i) for i in range(1, win + 1)])
    ins_in_head = ",".join(["ins{}".format(i) for i in range(1, win + 1)])
    del_in_head = ",".join(["del{}".format(i) for i in range(1, win + 1)])
    outh2.write(
        '#Kmer,Window,Ref,Strand,Coverage,{},{},{},{}\n'.format(q_in_head, mis_in_head, ins_in_head, del_in_head))

    tmpfh = open(out_tmp, 'r')
    cov, q, mis, ins, dele = [], [], [], [], []
    firstline = tmpfh.readline().rstrip().split(',')
    current_win = (firstline[0], firstline[1], firstline[3], firstline[6])
    lines = []
    lines.append(firstline)
    ary = []
    for l in tmpfh:
        ary = l.rstrip().split(',')
        try:
            window = (ary[0], ary[1], ary[3], ary[6])
        except:
            print(l.rstrip())
        if window != current_win:
            for ele in lines:
                q.append(ele[8])
                mis.append(ele[11])
                ins.append(ele[12])
                dele.append(ele[13])
                cov.append(ele[7])
            Qs = ",".join(q)
            Mis = ",".join(mis)
            Ins = ",".join(ins)
            Del = ",".join(dele)
            Cov = ":".join(cov)
            print(",".join(current_win), Cov, Qs, Mis, Ins, Del, sep=",", file=outh2)
            cov, q, mis, ins, dele = [], [], [], [], []
            current_win = window
            lines = []
        lines.append(ary)
    # last 5 lines
    cov, q, mis, ins, dele = [], [], [], [], []
    for ele in lines:
        q.append(ele[8])
        mis.append(ele[11])
        ins.append(ele[12])
        dele.append(ele[13])
        cov.append(ele[7])
    Qs = ",".join(q)
    Mis = ",".join(mis)
    Ins = ",".join(ins)
    Del = ",".join(dele)
    Cov = ":".join(cov)
    print(",".join(window), Cov, Qs, Mis, Ins, Del, sep=",", file=outh2)

    tmpfh.close()
    outh2.close()
    os.remove(out_tmp)
    return (out2)


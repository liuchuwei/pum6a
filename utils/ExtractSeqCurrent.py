#!/usr/bin/env python
# -*- coding: utf-8 -*-
'Modify from: https://github.com/gaoyubang/nanom6A'

import re

import h5py
import numpy as np
from statsmodels import robust


def get_label(fl, args):

    'Get raw data from single fast5 file'
    'https://github.com/gaoyubang/nanom6A'

    ## 1.Open file
    try:
        fast5_data = h5py.File(fl, 'r')
    except IOError:
        raise IOError('Error opening file. Likely a corrupted file.')
    # 2.Get raw data
    try:
        raw_dat = list(fast5_data['/Raw/Reads/'].values())[0]
        # raw_attrs = raw_dat.attrs
        raw_dat = raw_dat['Signal'][()]
    # ~ .value
    except:
        raise RuntimeError(
            'Raw data is not stored in Raw/Reads/Read_[read#] so ' +
            'new segments cannot be identified.')
    # 3.Read corrected data
    try:
        corr_data = fast5_data['/Analyses/' + args.basecall_group + '/' + args.basecall_subgroup + '/Events']
        corr_attrs = dict(list(corr_data.attrs.items()))
        corr_data = corr_data[()]
    # ~ .value
    except:
        raise RuntimeError(('Corrected data not found.'))

    # 4. Extra information
    corr_start_rel_to_raw = corr_attrs['read_start_rel_to_raw']  #
    if len(raw_dat) > 99999999:
        raise ValueError(fl + ": max signal length exceed 99999999")
    if any(len(vals) <= 1 for vals in (corr_data, raw_dat)):
        raise NotImplementedError(('One or no segments or signal present in read.'))
    event_starts = corr_data['start'] + corr_start_rel_to_raw
    event_lengths = corr_data['length']
    event_bases = corr_data['base']
    fast5_data.close()

    return (raw_dat, event_bases, event_starts, event_lengths)


# def search_RRACH(signal,start,length,base,fn_string, args):

def search_kmer(signal,start,length,base,fn_string, args):

    uniq_arr=np.unique(signal)
    signal = (signal - np.median(uniq_arr)) / np.float(robust.mad(uniq_arr))
    raw_signal = signal.tolist()
    # kmer_fillter="[AG][AG]AC[ACT]"
    kmer_fillter=args.kmer_filter
    windows = [i for i in range(-int(args.kmer)//2+1, int(args.kmer)//2+1)]
    line=""
    total_seq="".join([x.decode() for x in base])
    clipnum=int(args.clip)
    # ~ print(length)
    # ~ print(base)
    # ~ print(len(length))
    for indx in range(len(length)):
        if 2+clipnum<=indx<=len(length)-3-clipnum:
            if args.kmer == '5':
                base0,base1,base2,base3,base4=[base[indx+x].decode() for x in windows]
                kmer_now_t="%s%s%s%s%s"%(base0,base1,base2,base3,base4)
            if args.kmer == '6':
                base0,base1,base2,base3,base4,base5=[base[indx+x].decode() for x in windows]
                kmer_now_t="%s%s%s%s%s%s"%(base0,base1,base2,base3,base4,base5)
            if args.kmer == '7':
                base0, base1, base2, base3, base4, base5, base6 = [base[indx + x].decode() for x in windows]
                kmer_now_t = "%s%s%s%s%s%s%s" % (base0, base1, base2, base3, base4, base5, base6)
            if args.kmer == '9':
                base0, base1, base2, base3, base4, base5, base6, base7, base8 = [base[indx + x].decode() for x in windows]
                kmer_now_t = "%s%s%s%s%s%s%s%s%s" % (base0, base1, base2, base3, base4, base5, base6, base7, base8)

            # ~ print(kmer_now_t)
            # ~ print(indx,kmer_now_t)
            list_have=[x.start() for x in re.finditer(kmer_fillter,kmer_now_t)]
            if len(list_have)==0:
                continue
            raw_signal_every=[raw_signal[start[indx+x]:start[indx+x]+length[indx+x]] for x in windows]
            mean=[np.mean(x) for x in raw_signal_every]
            std=[np.std(x) for x in raw_signal_every]
            md_intense = [np.median(x) for x in raw_signal_every]
            length2=[length[indx+x] for x in windows]
            #############
            if args.kmer == '5':
                line+="%s|%s|%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n"%(str(fn_string).split("/")[-1],indx,"N",base2,kmer_now_t,"|".join([str(x) for x in mean]),"|".join([str(x) for x in  std]),"|".join([str(x) for x in  md_intense]),"|".join([str(x) for x in length2]),kmer_now_t)
            if args.kmer == '6':
                line+="%s|%s|%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n"%(str(fn_string).split("/")[-1],indx,"N",base3,kmer_now_t,"|".join([str(x) for x in mean]),"|".join([str(x) for x in  std]),"|".join([str(x) for x in  md_intense]),"|".join([str(x) for x in length2]),kmer_now_t)
            if args.kmer == '7':
                line+="%s|%s|%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n"%(str(fn_string).split("/")[-1],indx,"N",base3,kmer_now_t,"|".join([str(x) for x in mean]),"|".join([str(x) for x in  std]),"|".join([str(x) for x in  md_intense]),"|".join([str(x) for x in length2]),kmer_now_t)
            if args.kmer == '9':
                line+="%s|%s|%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n"%(str(fn_string).split("/")[-1],indx,"N",base4,kmer_now_t,"|".join([str(x) for x in mean]),"|".join([str(x) for x in  std]),"|".join([str(x) for x in  md_intense]),"|".join([str(x) for x in length2]),kmer_now_t)
    # ~ print(line)
    return line

def extract_feature(fl, args):

    try:
        (raw_data, raw_label, raw_start, raw_length) = get_label(fl, args)
    except Exception as e:
        # ~ print(str(e))
        return False, (None, None)

    raw_data = raw_data[::-1]
    # ~ print(input_file,raw_start,raw_length,raw_label)
    total_seq = "".join([x.decode() for x in raw_label])
    ids = fl.split("/")[-1]
    total_seq = ">%s\n%s\n" % (ids, total_seq)
    line = search_kmer(raw_data,raw_start,raw_length,raw_label,fl, args)
    return line, total_seq


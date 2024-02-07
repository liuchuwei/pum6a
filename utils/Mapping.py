import os
from utils.tookit import Tookits


def mapping(args, nums):
    ## output fasta file
    output = open(args.output + ".feature.fa", "w")
    output.write("".join([str(x[1]) for x in nums]))
    output.close()

    ## mapping
    tools = Tookits()

    fl1 = args.output + ".feature.fa"
    basefl = '/'.join(args.output.split("/")[:-1])

    cmd = "%s faidx %s" % (tools.samtools, fl1)
    os.system(cmd)
    cmd = "%s --secondary=no -ax splice -uf -k14 -t %s %s  %s|%s view -@ %s -bS - |%s sort -@ %s - >%s/extract.sort.bam" % (
        tools.minimap2, args.cpu, args.genome, fl1, tools.samtools, args.cpu, tools.samtools, args.cpu, basefl)
    os.system(cmd)
    cmd = "%s index %s/extract.sort.bam" % (tools.samtools, basefl)
    os.system(cmd)
    # cmd = ' ~/anaconda3/envs/Nanom6A/bin/sam2tsv -r {1} {0}/extract.sort.bam|gzip -c >{0}/extract.sort.bam.tsv.gz'.format(
    #     basefl, args.genome)
    cmd = 'java -jar {2} -r {1} {0}/extract.sort.bam|gzip -c >{0}/extract.sort.bam.tsv.gz'.format(
        basefl, args.genome, tools.sam2tsv)
    os.system(cmd)
    cmd = "%s -bed12 -split -i %s/extract.sort.bam >%s/extract.bed12" % (
        tools.bamtobed, basefl, basefl)
    os.system(cmd)
    print("gene annotation")
    cmd = "%s --secondary=no -ax splice -uf -k14 -t %s %s  %s|%s view -@ %s -bS - |%s sort -@ %s - >%s/extract.reference.sort.bam" % (
        tools.minimap2, args.cpu, args.reference, fl1, tools.samtools, args.cpu, tools.samtools, args.cpu, basefl)
    os.system(cmd)
    cmd = "%s -bed12 -split -i %s/extract.reference.sort.bam >%s/extract.reference.isoform.bed12" % (
        tools.bamtobed, basefl, basefl)
    os.system(cmd)

    ## replace_gene
    fl = args.isoform
    # ~ head gene2transcripts.txt
    # ~ DDX11L1	NR_046018.2
    rs_gene = {}
    for i in open(fl, "r"):
        ele = i.rstrip().split()
        for item in ele[1:]:
            rs_gene[item] = ele[0]
    fl = "%s/extract.reference.isoform.bed12" % (basefl)
    output = open("%s/extract.reference.bed12" % (basefl), "w")
    # NM_001197125.1	30	1440	1e0208b3-8061-451f-9415-73df9654f9da.fast5	0	+	30	1440	255,0,0	1	1410	0
    for i in open(fl, "r"):
        ele = i.rstrip().split()
        if ele[0] in rs_gene:
            ele[0] = rs_gene[ele[0]]
            output.write("\t".join(ele) + "\n")
    output.close()
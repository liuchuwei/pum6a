# define config
class Tookits(object):

    def __init__(self):

        self.basecall = "~/biotools/ont-guppy/bin/guppy_basecaller"
        self.model = "~/biotools/ont-guppy/data/rna_r9.4.1_70bps_hac.cfg"
        # self.minimap2 = "~/biotools/minimap2-2.26_x64-linux/minimap2"
        self.minimap2 = "~/biotools/minimap2/minimap2"
        self.samtools = "~/biotools/samtools-1.16.1/samtools"
        self.bamtobed = "~/biotools/bedtools2/bin/bamToBed"
        self.nanopolish = "~/biotools/nanopolish/nanopolish"

        # self.multi_to_single_fast5 = "/home/zdj/anaconda3/envs/mDRS/bin/multi_to_single_fast5"
        self.multi_to_single_fast5 = "/home/wan/anaconda3/envs/mDRS/bin/multi_to_single_fast5"
        self.removel_fail_fast = "remove_failed_fast5.R"

        # self.tombo = "/home/zdj/anaconda3/envs/tombo/bin/tombo"
        self.tombo = "/home/wan/anaconda3/envs/tombo/bin/tombo"

        # self.sam2tsv = '/home/zdj/biotools/sam2tsv.jar'
        self.sam2tsv = '/home/wan/biotools/sam2tsv.jar'

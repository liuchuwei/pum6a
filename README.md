# pum6a
## Dependence
![](https://img.shields.io/badge/software-version-blue)  
[![](https://img.shields.io/badge/Guppy-v6.5.7-green)](https://community.nanoporetech.com/downloads)
[![](https://img.shields.io/badge/Minimap2-v2.24-green)](https://github.com/lh3/minimap2)
[![](https://img.shields.io/badge/samtools-v1.1.7-green)](https://github.com/samtools/samtools)
[![](https://img.shields.io/badge/bedtools-v2.29.1-green)](https://bedtools.readthedocs.io/en/latest/)

![](https://img.shields.io/badge/Genome-version-blue)  
[![](https://img.shields.io/badge/GRCm39-17.07.23-orange)](https://hgdownload.soe.ucsc.edu/goldenPath/mm39/bigZips/)
[![](https://img.shields.io/badge/GRCh38.p14-17.07.23-orange)](https://hgdownload.soe.ucsc.edu/goldenPath/hg38/bigZips/)


<!-- TABLE OF CONTENTS -->
<details open="open">
  <summary><h2 style="display: inline-block">Table of Contents</h2></summary>
  <ol>
    <li>
      <a href="#Intallation">Intallation</a>
    </li>
    <li><a href="#Usage">Usage</a></li>
    <li><a href="#References">References</a></li>
    <li><a href="#license">License</a></li>
    <li><a href="#Contact">Contact</a></li>
  </ol>
</details>


## Intallation
1.Clone the project
   ```shell
   git clone https://github.com/liuchuwei/pum6a.git
   ```
2.Install conda environment
   ```shell
   conda env create -f caNano.yaml
   conda env create -f tombo.yaml
   ```
3.prepare tookit: 

check and modify the tookit.py file (in 'utils' directory).
    
## Usage
1.Basecalling
   ```shell
   python 01.basecalling.py -i $fast5 -o $out
   ```
2.Resguiggle

preprocess

   ```shell
   conda activate tombo
   python 02.resquiggle_pre.py -f $fast5 -o $out
   ```
annotate_raw_with_fastqs

   ```shell
   cat *.fastq > merge.fastq
   python 03.resquiggle.py preprocess annotate_raw_with_fastqs \
   --fast5-basedir $single \
   --fastq-filenames $merge_fastq \
   --overwrite \
   --processes 8
   ```
resquiggling
   ```shell
    python 03.resquiggle.py resquiggle $fast5 $reference \
    --rna \
    --corrected-group RawGenomeCorrected_000 \
    --basecall-group Basecall_1D_000 \
    --overwrite \
    --processes 16 \
    --fit-global-scale \
    --include-event-stdev
   ```

3.Minimap
   ```shell
   python 04.minimap.py -i <directory of fastq files> -o <output directory> -r <path of reference>
   ```

4.m6a detection

4.1 activate environment
   ```shell
   conda activate pum6a
   ```

4.2 preprocess
   ```shell
   python run.py preprocess --single $single_fast5 -o $output -g $genome.fa -r $transcript.fa -i $gene2transcripts.txt -b $bam
   ```

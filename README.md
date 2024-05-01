# pum6a
## Dependence
![](https://img.shields.io/badge/software-version-blue)  
[![](https://img.shields.io/badge/Guppy-v6.5.7-green)](https://community.nanoporetech.com/downloads)
[![](https://img.shields.io/badge/Minimap2-v2.24-green)](https://github.com/lh3/minimap2)
[![](https://img.shields.io/badge/samtools-v1.1.7-green)](https://github.com/samtools/samtools)
[![](https://img.shields.io/badge/bedtools-v2.29.1-green)](https://bedtools.readthedocs.io/en/latest/)

[![](https://img.shields.io/badge/ELIGOS-v2.0.1-blue)](https://gitlab.com/piroonj/eligos2)
[![](https://img.shields.io/badge/Epinano-v1.2.0-blue)](https://github.com/novoalab/EpiNano)  
[![](https://img.shields.io/badge/MINES-v0.0-orange)](https://github.com/YeoLab/MINES.git)
[![](https://img.shields.io/badge/Tombo-v1.5.1-orange)](https://github.com/nanoporetech/tombo)
[![](https://img.shields.io/badge/Nanom6A-v2.0-orange)](https://github.com/gaoyubang/nanom6A)  
[![](https://img.shields.io/badge/m6Anet-v1.0-purple)](https://github.com/GoekeLab/m6anet) 
[![](https://img.shields.io/badge/nanopolish-v0.14.0-purple)](https://github.com/jts/nanopolish)  

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
   conda create -n pum6a python=3.8
   conda activate pum6a
   pip install torch==2.2.1 torchvision==0.17.1 torchaudio==2.2.1 --index-url https://download.pytorch.org/whl/cu118
   pip install dask==2023.5.0 h5py==3.10.0 numpy==1.24.3 pandas==2.0.3 scikit-learn==1.3.2 tqdm==4.66.1 toml==0.10.2 statsmodels==0.14.1
   ```

The usage of pum6a require the installation of [tombo](https://github.com/nanoporetech/tombo) environment.

You can install [MINES](https://github.com/YeoLab/MINES.git), [m6Anet](https://github.com/GoekeLab/m6anet), [ELIGOS](https://gitlab.com/piroonj/eligos2), [Nanom6A](https://github.com/gaoyubang/nanom6A), and [Epinano](https://github.com/novoalab/EpiNano) environment according to your need. 
Usage example can be found in the m6a_detection directory.

3.prepare tookit:
check and modify the tool paths of tookit.py file (in 'utils' directory).
    
## Usage

### Experiment
Experiment of pum6a framework for different positive and unlable bags datasets.

```shell
python run.py experiment --config $*.toml
```
for example: python run.py experiment --config log/Internet_pum6a_0.5Freq_88888.toml

### m6A modification detection
1.Basecalling
   ```shell
   python process/01.basecalling.py -i $fast5 -o $out
   ```
2.Resguiggle

preprocess

   ```shell
   conda activate tombo
   python process/02.resquiggle_pre.py -f $fast5 -o $out
   ```
annotate_raw_with_fastqs

   ```shell
   cat *.fastq > merge.fastq
   python process/03.resquiggle.py preprocess annotate_raw_with_fastqs \
   --fast5-basedir $single \
   --fastq-filenames $merge_fastq \
   --overwrite \
   --processes 8
   ```
resquiggling
   ```shell
    python process/3.resquiggle.py resquiggle $fast5 $reference \
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
   python process/04.minimap.py -i <directory of fastq files> -o <output directory> -r <path of reference>
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

4.3 train model
   ```shell
   python run.py train --config $config.toml
   ```

4.4 predict
   ```shell
   python run.py predict --config $config.toml
   ```
4.5 evaluate
   ```shell
   python run.py evaluate --config $config.toml
   ```

## License
Distributed under the MIT License. See LICENSE for more information.

## Contact
liuchw3@mail2.sysu.edu.cn

## Reference
## 1.Tombo
activate environment
```shell
conda activate tombo
```
modification detection
```shell
python 01.Tombo.py detect_modifications de_novo 
--fast5-basedirs <directory of fast5 files> \
--statistics-file-basename <output name> \
--corrected-group RawGenomeCorrected_000 \
--processes 16
```
text output
```shell
python 01.Tombo.py text_output browser_files --fast5-basedirs <directory of fast5 files> \
--statistics-filename <output name>  \
--browser-file-basename wt_rrach \
--genome-fasta <path of reference> \
--motif-descriptions RRACH:3:m6A \
--file-types coverage dampened_fraction fraction \
--corrected-group RawGenomeCorrected_000
```
postprocess
```shell
awk '{if($0!=""){print $0}}' $.plus.wig > $_wig2bed.wig
wig2bed --multisplit tombo --sort-tmpdir tmp <$_wig2bed.wig > $.wig.bed
python 01.Tombo_postpresss.py -o $ouput.csv -bed $.wig.bed -ref $reference_directory -min_read 3
```

## 2.Mines

```shell
conda activate tombo
python 01.Tombo.py text_output browser_files --fast5-basedirs <directory of fast5 files> \
--statistics-filename <output name>  \
--browser-file-basename wt_rrach \
--file-types coverage dampened_fraction fraction \
--corrected-group RawGenomeCorrected_000
```

```shell
awk '{if($0!=null){print $0}}' $.plus.wig > $_wig2bed.wig
wig2bed < $_wig2bed.wig > $.wig.bed --multisplit=mines
```

```shell
conda activate Mines
python 02.Mines.py --fraction_modified $tombo/wt.fraction_modified_reads.plus.wig.bed \
--coverage $tombo/wt.coverage.plus.bedgraph \
--output wt.bed \
--ref $ref \
--kmer_models $MINES/Final_Models/names.txt
```

```shell
awk 'BEGIN{OFS="\t"}{if($9==""){print $1,$2,$3,$4,$5,$6,$7,$8,0}else{print $0}}' $.bed > $_1.bed
awk 'BEGIN{OFS=""}{print $1,"\t",$2,"\t",$9,"\t",$8,"\t",$1,"|",$2,"\t",$4}' $_1.bed > $_C5_RRACH.bed
python 02.Mines_postpresss.py -o $ouput.csv -bed $.wig.bed -ref $reference_directory -min_read 3
```
## 3.Nanom6A

list all fast5 file
```shell
find single -name "*.fast5" >files.txt
```

extracting signals
```shell
conda activate Nanom6A
python 03.Nanom6A_extract_raw_and_feature_fast.py --cpu=$int --fl=files.txt -o result --clip=10
```
predicting m6A site

```shell
python 03.Nanom6A_predict_site.py --cpu $int -i result -o result_final -r $.fa -g $.genome.fa -b $.gene2transcripts.txt --model Nanom6A/model
```
postprocess
```shell
python 03.Nanom6A_postpresss.py -o $ouput.csv -bed $.total_prediction.csv -ref $reference_directory -min_read 3
```

## 4.Epinano
slide kmer
```shell
python 04.Epinano_slide.py <per.site.csv> 5
```
predicting m6A site
```shell
python 04.Epinano_predict.py \
--model $Epinano/models/rrach.q3.mis3.del3.linear.dump \
--predict $5mer.csv \
--columns 8,13,23 \
--out_prefix $prefix
```
postprocess
```shell
python 03.Epinano_postpresss.py -o $ouput.csv -bed $.total_prediction.csv -ref $reference_directory -min_read 3
```

## 5.Eligos
```shell
conda activate eligos
python 05.ELIGOS.py rna_mod -i <bam file> -reg <bed files> -ref <REFERENCE> -m <rBEM5+2 model> -p <output file prefix> \
-o <output file directory> --sub_bam_dir <SUB_BAM_DIR> \
--max_depth 2000000 --min_depth 5 --esb 0 --oddR 1 --pval 1 -t 16
```
```shell
python 05.ELIGOS_postpresss.py -o $ouput.csv -bed $.combine.txt -ref $reference_directory 
```
## 6.m6Anet
eventalign
```shell
python 06.m6anet_eventalign.py -f <directory of fast5 files> -o <output directory> \
 -fq <path of fastq> -r <path of reference> -bam <path of bam files> -o <output directory>
```
data preprocess
```shell
python 06.m6anet.py dataprep --eventalign wt_eventalign.txt
--out_dir wt
--n_processes 16
--readcount_max 2000000
```

#detect m6A
```shell
python 06.m6anet.py inference --input_dir wt
--out_dir run/wt
--n_processes 16
```

postprocess
```shell
python 06.m6anet_postpresss.py -o $ouput.csv -bed $.total_prediction.csv -ref $reference_directory
```

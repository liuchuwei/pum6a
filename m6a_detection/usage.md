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

## 2.Mine
```shell
awk '{if($0!=null){print $0}}' wt.fraction_modified_reads.plus.wig > wt.wig
wig2bed < wt.wig > wt.fraction_modified_reads.plus.wig.bed --multisplit=mines
```
```shell
python 02.Mines.py --fraction_modified $tombo/wt.fraction_modified_reads.plus.wig.bed \
--coverage $tombo/wt.coverage.plus.bedgraph \
--output wt.bed \
--ref $ref \
--kmer_models $MINES/Final_Models/names.txt
```
## 3.Nanom6A

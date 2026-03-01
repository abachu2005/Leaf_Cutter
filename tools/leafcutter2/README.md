# LeafCutter2

## Introduction

LeafCutter2 is a tool for clustering, functional characterization and quantification of splice junction read counts. It implements a novel dynamic programming algorithm over the standard LeafCutter output to classify splice junctions and alternative splicing events according to their molecular function (i.e.: protein-coding or unproductive).

#### Prerequisites

All the necessary python libraries can be installed in a conda envoronment with the `leafcutter2_env.yml` file in this repository. You can do this by simply running:

```
conda env create -f leafcutter2_env.yml
```

The environemnt can also be installed using [Mamba](https://mamba.readthedocs.io/en/latest/), which is generally faster:

```
mamba env create -f leafcutter2_env.yml
```

After creating the environment (irrespective of which method was used), you can activate it by running:

```
conda activate leafcutter2
```


## Clustering, classifying and quantifying splice junctions

### Input:
- Splice junction BED files. They should contain at least six columns (chrom, start, end, name, score and strand), with the fifth column corresponding to the splice junction read counts. 
- GTF annotation with genes, start codons and stop codons.
- Genome assembly FASTA file. It must correspond to the same assembly as the GTF file. Make sure it has an index in a [faidx file](https://www.htslib.org/doc/samtools-faidx.html) in the same directory.

We recommend using the BED-formatted `.junc` files from obtained from BAM files using [regtools junctions extract](https://regtools.readthedocs.io/en/latest/commands/junctions-extract/) as input. 

E.g.: `regtools junctions extract -a 8 -i 50 -I 500000 bamfile.bam -o outfile.junc`. 

The BED files can also be obtained from [STAR's](https://github.com/alexdobin/STAR/blob/master/doc/STARmanual.pdf) `SJ.out.tab` files with minimal modifications. We include the script `scripts/add_on_scripts/STAR2Junc.py` to facilitate preprocessing. 

Usage: `python STAR2Junc.py sample.SJ.out.tab sample.bed.gz`. 

### Running LeafCutter2

We provide a basic example from GTEx in the `example/` directory. Assuming that we're in the `example/` directory, a basic LeafCutter2 run works as follows:

```
python ../scripts/leafcutter2.py \
    -j junction_files.txt \
    -r output_dir \
    -A annotation/chr10.gtf.gz \
    -G annotation/chr10.fa.gz
```

-    `-j junction_files.txt` should be a text file listing path to each junction file, one path per line.
-    `-r output_dir` specifies the directory of output (default is current directory, `./`). 
-    `-A annotation/chr10.gtf.gz` is a gtf file of chromosome 10 obtained from Gencode v43.
-    `-G annotation/chr10.fa.gz` a FASTA file of chromosome 10 (GRCh38 assembly). 

**Note:** Make sure that the chromosome names match between the BED, GTF and FASTA files.

### Output:
- `leafcutter2.cluster_ratios.gz` a table quantifying splice junction read counts for each intron, divided by the total number of reads observed for the intron cluster to which the intron belongs. Each row corresponds to a splice junction, with the first column indicating the splice junction ID and all subsequent columns corresponding to each sample. The splice junction ID has the following format: `chr10:134786:179993:clu_1_+:PR`, indicating the chromosome, start and end of the splice junction, the LeafCutter intron cluster to which it belongs (in this case, `clu_1_+`, and a label indicating the splice junction's function: **PR** (productive/protein-coding), **UP** (unproductive), **NE** (ambiguous in their functional effect) or **IN** (intergenic).
- `leafcutter2.junction_counts.gz` a table similar to `leafcutter2.cluster_ratios.gz`, except it only keeps track of the splice junction read counts for each intron. Useful for differential splicing analysis with [LeafCutter's R package](https://davidaknowles.github.io/leafcutter/).
- `clustering/` a directory containing files relevant for clustering and annotation. 
    - `clustering/leafcutter2_clusters` contains the intron clusters. Useful for skipping the clustering step in repeated runs.
    - Other files documenting stats from the clustering and classification algorithm. Useful for debugging.
    
### A note on GTF files:

For classification, LeafCutter2 requires information from a GTF file. Please ensure that the features column (third column on a GTF) includes the following type of information, and that the names match:
- `gene`
- `transcript`
- `CDS`
- `start_codon`
- `stop_codon`

In addition, LeafCutter2 requires information on gene and transcript types. These come in the 9th column of a GTF file, but different annotations can use different tags (e.g., Gencode typically uses `gene_type`, while Ensembl uses `gene_biotype`). You can specify what tag your GTF uses by using the following parameters:
- `--gene_type` (default: `gene_type`)
- `--transcript_type` (default: `transcript_type`)

You can also specify what tag your GTF uses for gene and transcript name (or if you prefer, change from names to gene and transcript IDs) by using:
- `--gene_name` (default: `gene_name`)
- `--transcript_name` (default: `transcript_name`)

Finally, some GTFs might lack some or all of these features and information. E.g, if you're working with an assembled GTF with StringTie, or some GTFs downloaded from the UCSC Genome Browser. In this case, you can reformat your GTF file using our add on script [Reformat_gtf.py](https://github.com/cfbuenabadn/leafcutter2/blob/main/scripts/add_on_scripts/Reformat_gtf.py). See instructions [here](https://github.com/cfbuenabadn/leafcutter2/blob/main/scripts/add_on_scripts/Reformat_gtf.README.md). 


### Parameters

```
python scripts/leafcutter2.py -h

usage: leafcutter2.py [-h] -j JUNCFILES 
                              [-o OUTPREFIX] [-q] [-r RUNDIR]
                              [-l MAXINTRONLEN] [-m MINCLUREADS]
                              [-M MINREADS] [-p MINCLURATIO]
                              [-c CLUSTER] [-k] [-C]
                              [-N NOISECLASS] [-f OFFSET] [-T]

required arguments:
  -j JUNCFILES, --juncfiles JUNCFILES
                        text file with all junction files to be processed
                        
optional arguments:
  -h, --help            show this help message and exit
  -r RUNDIR, --rundir RUNDIR
                        write to directory (default ./)
  -o OUTPREFIX, --outprefix OUTPREFIX
                        output prefix (default leafcutter2)
  -q, --quiet           don't print status messages to stdout
  -A ANNOTATION, --annot ANNOTATION
                        GTF annotation file
  -G GENOME, --genome GENOME
                        Genome fasta file
  -N MAXJUNCS, --max_juncs MAXJUNCS
                        skip solveNMD function if gene contains more than N juncs. 
                        Juncs in skipped genes are assigned Coding=False. Default 10000
  -l MAXINTRONLEN, --maxintronlen MAXINTRONLEN
                        maximum intron length in bp (default 100,000bp)
  -m MINCLUREADS, --minclureads MINCLUREADS
                        minimum reads in a cluster (default 30 reads)
  -M MINREADS, --minreads MINREADS
                        minimum reads for a junction to be considered for
                        clustering(default 5 reads)
  -p MINCLURATIO, --mincluratio MINCLURATIO
                        minimum fraction of reads in a cluster that support a
                        junction (default 0.001)
  -D MINREADSTD, --minreadstd MINREADSTD
                        minimum standard deviation of reads across samples for a \
                        junction to be included in output (default 0.5)
  -p MINCLURATIO, --mincluratio MINCLURATIO
                        minimum fraction of reads in a cluster that support a junction \
                        (default 0.001)
  -c CLUSTER, --cluster CLUSTER
                        refined cluster file when clusters are already made
  -k, --checkchrom      check that the chromosomes are well formated e.g. chr1,
                        chr2, ..., or 1, 2, ...
  -C, --includeconst    also include constitutive introns
  -f OFFSET, --offset OFFSET
                        Offset sometimes useful for off by 1 annotations. (default 0)
  -T, --keeptemp        keep temporary files. (default false)
  -L, --keepleafcutter1 keep temporary LeafCutter1 files. Useful for running differential splicing 
                        analysis with leafcutter's R package. (default false)
  -P, --keepannot       save parsed annotations to .pckle files. (default false)
  -g, --gene_type GENE_TYPE
                        tag for gene type in GTF file (default gene_type)
  -t, --transcript_type GENE_TYPE
                        tag for transcript type in GTF file (default transcript_type)
  -gn, --gene_type GENE_TYPE
                        tag for gene name or ID in GTF file (default gene_name)
  -tn, --transcript_type GENE_TYPE
                        tag for transcript name or ID in GTF file (default transcript_name)
```

## Pre-clustering splice junctions (optional)

Generating intron clusters first can save time form multiple subsequent runs. The script `scripts/leafcutter_make_clusters.py`
makes intron clusters separately that can be later used as input for `scripts/leafcutter2.py`. You can generate clusters by running: 

```
python scripts/leafcutter_make_clusters.py \
    -j junction_files.txt \
    -r output_dir \
```
This will generate a file named `output_dir/clustering/leafcutter2_clusters` that can be later used as an input for LeafCutter2. This will skip the clustering step:

```
python ../scripts/leafcutter2.py \
    -j junction_files.txt \
    -r output_dir \
    -A annotation/chr10.gtf.gz \
    -G annotation/chr10.fa.gz
    -c output_dir/clustering/leafcutter2_clusters
```

**Note:** The junction-filtering options (--MINCLUREADS, --MINREADS, --MINCLURATIO) will be ignored by `leafcutter2.py` if a pre-defined set of intron clusters is provided.


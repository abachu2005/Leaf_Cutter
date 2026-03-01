# Reformat_gtf.py

Reformat_gtf.py is a script used to reformat gtf files as needed for the leafcutter classification script. That is, each transcript feature must have child features for start_codon, stop_codon, and the transcript (and all its child features) should have attribute tags for "gene_name", "gene_type", "transcript_name", and "transcript_type", where the value for gene_type and transcript_type should be "protein_coding" for protein-coding transcripts. Current Gencode gtf files are already like this. This script is more necessary to reformat gtf files from other sources (eg Ensembl, RefSeq, output of cufflinks, stringTie, etc.).

## Dependencies

The necessary python libraries can be installed with
```bash
conda create -f Reformat_gtf.conda_env.yml
```

## Usage

And the `--help` flag can be used for more detail:

```bash
python Reformat_gtf.py --help
```

```

usage: Reformat_gtf.py [-h] -i GTF_IN -o GTF_OUT [-input_type {gtf,bed12}] -fa
                       FASTA_IN [-bed12_out BED12_OUT] [-n N_LINES]
                       [-infer_transcript_type_approach {A,B,C}]
                       [-infer_gene_type_approach {A,B}]
                       [-transcript_type_attribute_name TRANSCRIPT_TYPE_ATTRIBUTE_NAME]
                       [-gene_type_attribute_name GENE_TYPE_ATTRIBUTE_NAME]
                       [-transcript_name_attribute_name TRANSCRIPT_NAME_ATTRIBUTE_NAME]
                       [-gene_name_attribute_name GENE_NAME_ATTRIBUTE_NAME]
                       [-extra_attributes EXTRA_ATTRIBUTES]
                       [-NMDetectiveB_coding_threshold {1,2,3,4,5,6,7}]
                       [-translation_approach {A,B,C,D,E}] [-v]

Helper script to reformat GTF file for compatibility with
SpliceJunctionClassifier.py. More specifically, for a GTF to be compatible
with that script, each gene feature and transcript feature must have a
'gene_type' and 'gene_name' attributes where 'gene_type' attribute value is
'protein_coding' for protein coding genes, and 'gene_name' attribute value is
unique for each gene. Each child transcript feature must also have
'transcript_type' and 'transcript_name' attributes, where 'transcript_type' is
'protein_coding' for protein_coding (productive) transcript isoforms. Also,
each protein_coding isoform must have a child 'exon' 'CDS', 'start_codon' and
'stop_codon' features (which also have the gene_type, gne_name,
transcript_type, and transcript_name attributes of their parent features). As
in Gencode v43 GTFs, transcripts that are not 'protein_coding' may also have
'CDS', 'start_codon' and 'start_codon' child features but if the
transcript_type attribute is not 'protein_coding', these will not determine
inclusion into the set of productive start/stop codons by
SpliceJunctionClassifier.py. Furthermore, this script adds NMDetectiveB
attributes to the transcript (see options for this feature). This is useful in
cases when a GTF may be missing 'transcript_type' attribute, and if it is even
missing 'CDS' features, this script can add the 'CDS', 'start_codon', and
'stop_codon' features after attempting to translate the transcript sequence
and assign values to the 'transcript_type' attribute (eg, 'protein_coding' if
NMDetectiveB result is 'Last exon' and 'noncoding if NMDetectiveB result is
'Trigger NMD'). This script also optionally can output bed12 file for each
transcript. I have tested this on some GTF files from Gencode, Ensembl, and
UCSC. Depending on the exact nature (eg attribute names) in the input GTF, you
may have to alter some options to make the output suitable for
SpliceJunctionClassifier.py.

options:
  -h, --help            show this help message and exit
  -i GTF_IN             Input GTF file. May alternatively use bed12 file to
                        define input transcripts (see "-input_type" argument)
  -o GTF_OUT            Output GTF file
  -input_type {gtf,bed12}
                        "-i" input file that contains transcript structures
                        can be either gtf format, or bed12 format. If bed12
                        format, transcript_name, gene_name, transcript_type,
                        and gene_type attributes must be added as columns 13,
                        14, 15, and 16, respectively. default: "gtf"
  -fa FASTA_IN          Input FASTA file
  -bed12_out BED12_OUT  Optional bed12 out. One line per transcript.
                        Attributes as extra columns
  -n N_LINES            Number of lines to read in gtf. Useful for quick
                        debugging, but reading only n lines may cause buggy
                        behavior if a transcript feature line is processed but
                        not all of its child (eg exon, or CDS) features
  -infer_transcript_type_approach {A,B,C}
                        Approach to determine the transcript_type attribute
                        value in output. (A) Use existing value. (B) Value is
                        "protein_coding" if a the transcript contains child
                        features of type "CDS". Value is "noncoding"
                        otherwise. (C) Translate the transcript and use
                        NMDetectiveB on the ORF. See -translation_approach and
                        -NMDetectiveB_coding_threshold options to specify how
                        this is done. default: "A"
  -infer_gene_type_approach {A,B}
                        Approach to determine the gene_type attribute value in
                        output. (A) Use existing value. (B) Value is
                        "protein_coding" if gene has any child transcripts
                        have attribute transcript_type "protein_coding". Value
                        is "noncoding" otherwise. default: "A"
  -transcript_type_attribute_name TRANSCRIPT_TYPE_ATTRIBUTE_NAME
                        Name of the attribute in the input gtf file that
                        defines the transcript_type attribute in the output
                        file. Only relevant if
                        -infer_transcript_type_approach==A. default:
                        "transcript_type"
  -gene_type_attribute_name GENE_TYPE_ATTRIBUTE_NAME
                        Name of the attribute in the input gtf file that
                        defines the gene_type attribute in the output file.
                        Only relevant if -infer_gene_type_approach==A.
                        default: "gene_type"
  -transcript_name_attribute_name TRANSCRIPT_NAME_ATTRIBUTE_NAME
                        Name of the attribute in the input gtf file that
                        defines the transcript_name attribute in the output
                        file. default: "transcript_name"
  -gene_name_attribute_name GENE_NAME_ATTRIBUTE_NAME
                        Name of the attribute in the input gtf file that
                        defines the gene_name attribute in the output file.
                        default: "gene_name"
  -extra_attributes EXTRA_ATTRIBUTES
                        Extra transcript attributes (comma delimited quoted
                        string). default:
                        "transcript_support_level,tag,ccds_id"
  -NMDetectiveB_coding_threshold {1,2,3,4,5,6,7}
                        NMDetectiveB classifies each transcript into 7 ordinal
                        categories, from the most coding potential to the
                        least coding potential: (1) Last exon (2) Start
                        proximal (3) 50nt rule (4) Long exon (5) Trigger NMD
                        (6) No stop (7) No CDS. Transcripts with classified as
                        this NMDetective value or greater will be assigned
                        transcript_type attribute value of "noncoding", while
                        others will be value of "protein_coding". default: "5"
  -translation_approach {A,B,C,D,E}
                        Approach to use NMDetective to annotate CDS in output
                        for genes where gene_biotype/gene_type ==
                        "protein_coding", some of which may not have annotated
                        CDS in input (eg
                        transcript_type=="processed_transcript"). Possible
                        approaches: (A) using annotated ORF if ORF is present
                        in input with 5UTR and 3UTR. If 3UTR is present and
                        5UTR is absent (suggesting stop codon is annotated but
                        start codon may be outside of the transcript bounds),
                        search for longest ORF within transcript bounds where
                        a start codon is not required at the beginning of ORF.
                        Similarly, if 5UTR is present but 3UTR is absent,
                        search for longest ORF without requiring stop codon.
                        If neither UTR is present, or if no CDS is annotated
                        in input, search for longest ORF, not requiring start
                        or stop within transcript bounds. I think this
                        approach might be useful to correctly identify the
                        ORF, even if transcript bounds are not accurate, but
                        it has the downside that true "processed_transcripts"
                        with a internal TSS that eliminates the correct start
                        codon, may be erronesously be classified as "Last
                        exon" (ie productive) transcripts by NMDFinderB. (B)
                        Use only annotated CDS. In effect, output gtf is the
                        same except start_codon and stop_codon features are
                        added even if not present in input. This would be
                        useful for dealing with "processed_transcripts"
                        properly by NMDFinder, but I havent checked whether
                        CDS annotations in poorly annotated species (eg
                        lamprey, chicken, etc) are reasonable, which could be
                        a problem for Yangs script. (C) use annotated CDS if
                        present, and use first ATG if no CDS present (minimum
                        ORF length of 30 codons, not including start or stop).
                        (D) Same as (C) but no minimum ORF length. (E) Use
                        first ORF with minimum length > 42, regardless of
                        annotation
  -v, --verbose         Increase verbosity
```

### Examples

#### Gencode

Gencode gtf's should already be like this, but you can still run it to reformat the gtf, or to use the python script's built-in ability to translate transcripts and apply NMDFinderB algorithm to re-tag transcript_type and gene_type tags as "protein_coding" or "unproductive", or maybe re-tag the "gene_name" and "transcript_name" values to be replaced with the values for the "gene_id" and transcript_id" tag (the leafcutter2 classify junctions script will use the gene_name and transcript_name tag of the gtf you provide). Here I have the first 1000 lines of the Gencode v gtf.

```bash
## first download the gencode gtf, perhaps just first 1000 lines for this example
wget -O- ftp://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_human/release_43/gencode.v43.annotation.gtf.gz | zcat | head -n 1000 > .test_data/gencode.gtf

## and check the first 20 lines
head -n 20 .test_data/gencode.gtf
```

```
##description: evidence-based annotation of the human genome (GRCh38), version 43 (Ensembl 109)
##provider: GENCODE
##contact: gencode-help@ebi.ac.uk
##format: gtf
##date: 2022-11-29
chr1	HAVANA	gene	11869	14409	.	+	.	gene_id "ENSG00000290825.1"; gene_type "lncRNA"; gene_name "DDX11L2"; level 2; tag "overlaps_pseudogene";
chr1	HAVANA	transcript	11869	14409	.	+	.	gene_id "ENSG00000290825.1"; transcript_id "ENST00000456328.2"; gene_type "lncRNA"; gene_name "DDX11L2"; transcript_type "lncRNA"; transcript_name "DDX11L2-202"; level 2; transcript_support_level "1"; tag "basic"; tag "Ensembl_canonical"; havana_transcript "OTTHUMT00000362751.1";
chr1	HAVANA	exon	11869	12227	.	+	.	gene_id "ENSG00000290825.1"; transcript_id "ENST00000456328.2"; gene_type "lncRNA"; gene_name "DDX11L2"; transcript_type "lncRNA"; transcript_name "DDX11L2-202"; exon_number 1; exon_id "ENSE00002234944.1"; level 2; transcript_support_level "1"; tag "basic"; tag "Ensembl_canonical"; havana_transcript "OTTHUMT00000362751.1";
chr1	HAVANA	exon	12613	12721	.	+	.	gene_id "ENSG00000290825.1"; transcript_id "ENST00000456328.2"; gene_type "lncRNA"; gene_name "DDX11L2"; transcript_type "lncRNA"; transcript_name "DDX11L2-202"; exon_number 2; exon_id "ENSE00003582793.1"; level 2; transcript_support_level "1"; tag "basic"; tag "Ensembl_canonical"; havana_transcript "OTTHUMT00000362751.1";
chr1	HAVANA	exon	13221	14409	.	+	.	gene_id "ENSG00000290825.1"; transcript_id "ENST00000456328.2"; gene_type "lncRNA"; gene_name "DDX11L2"; transcript_type "lncRNA"; transcript_name "DDX11L2-202"; exon_number 3; exon_id "ENSE00002312635.1"; level 2; transcript_support_level "1"; tag "basic"; tag "Ensembl_canonical"; havana_transcript "OTTHUMT00000362751.1";
chr1	HAVANA	gene	12010	13670	.	+	.	gene_id "ENSG00000223972.6"; gene_type "transcribed_unprocessed_pseudogene"; gene_name "DDX11L1"; level 2; hgnc_id "HGNC:37102"; havana_gene "OTTHUMG00000000961.2";
chr1	HAVANA	transcript	12010	13670	.	+	.	gene_id "ENSG00000223972.6"; transcript_id "ENST00000450305.2"; gene_type "transcribed_unprocessed_pseudogene"; gene_name "DDX11L1"; transcript_type "transcribed_unprocessed_pseudogene"; transcript_name "DDX11L1-201"; level 2; transcript_support_level "NA"; hgnc_id "HGNC:37102"; ont "PGO:0000005"; ont "PGO:0000019"; tag "basic"; tag "Ensembl_canonical"; havana_gene "OTTHUMG00000000961.2"; havana_transcript "OTTHUMT00000002844.2";
chr1	HAVANA	exon	12010	12057	.	+	.	gene_id "ENSG00000223972.6"; transcript_id "ENST00000450305.2"; gene_type "transcribed_unprocessed_pseudogene"; gene_name "DDX11L1"; transcript_type "transcribed_unprocessed_pseudogene"; transcript_name "DDX11L1-201"; exon_number 1; exon_id "ENSE00001948541.1"; level 2; transcript_support_level "NA"; hgnc_id "HGNC:37102"; ont "PGO:0000005"; ont "PGO:0000019"; tag "basic"; tag "Ensembl_canonical"; havana_gene "OTTHUMG00000000961.2"; havana_transcript "OTTHUMT00000002844.2";
chr1	HAVANA	exon	12179	12227	.	+	.	gene_id "ENSG00000223972.6"; transcript_id "ENST00000450305.2"; gene_type "transcribed_unprocessed_pseudogene"; gene_name "DDX11L1"; transcript_type "transcribed_unprocessed_pseudogene"; transcript_name "DDX11L1-201"; exon_number 2; exon_id "ENSE00001671638.2"; level 2; transcript_support_level "NA"; hgnc_id "HGNC:37102"; ont "PGO:0000005"; ont "PGO:0000019"; tag "basic"; tag "Ensembl_canonical"; havana_gene "OTTHUMG00000000961.2"; havana_transcript "OTTHUMT00000002844.2";
chr1	HAVANA	exon	12613	12697	.	+	.	gene_id "ENSG00000223972.6"; transcript_id "ENST00000450305.2"; gene_type "transcribed_unprocessed_pseudogene"; gene_name "DDX11L1"; transcript_type "transcribed_unprocessed_pseudogene"; transcript_name "DDX11L1-201"; exon_number 3; exon_id "ENSE00001758273.2"; level 2; transcript_support_level "NA"; hgnc_id "HGNC:37102"; ont "PGO:0000005"; ont "PGO:0000019"; tag "basic"; tag "Ensembl_canonical"; havana_gene "OTTHUMG00000000961.2"; havana_transcript "OTTHUMT00000002844.2";
chr1	HAVANA	exon	12975	13052	.	+	.	gene_id "ENSG00000223972.6"; transcript_id "ENST00000450305.2"; gene_type "transcribed_unprocessed_pseudogene"; gene_name "DDX11L1"; transcript_type "transcribed_unprocessed_pseudogene"; transcript_name "DDX11L1-201"; exon_number 4; exon_id "ENSE00001799933.2"; level 2; transcript_support_level "NA"; hgnc_id "HGNC:37102"; ont "PGO:0000005"; ont "PGO:0000019"; tag "basic"; tag "Ensembl_canonical"; havana_gene "OTTHUMG00000000961.2"; havana_transcript "OTTHUMT00000002844.2";
chr1	HAVANA	exon	13221	13374	.	+	.	gene_id "ENSG00000223972.6"; transcript_id "ENST00000450305.2"; gene_type "transcribed_unprocessed_pseudogene"; gene_name "DDX11L1"; transcript_type "transcribed_unprocessed_pseudogene"; transcript_name "DDX11L1-201"; exon_number 5; exon_id "ENSE00001746346.2"; level 2; transcript_support_level "NA"; hgnc_id "HGNC:37102"; ont "PGO:0000005"; ont "PGO:0000019"; tag "basic"; tag "Ensembl_canonical"; havana_gene "OTTHUMG00000000961.2"; havana_transcript "OTTHUMT00000002844.2";
chr1	HAVANA	exon	13453	13670	.	+	.	gene_id "ENSG00000223972.6"; transcript_id "ENST00000450305.2"; gene_type "transcribed_unprocessed_pseudogene"; gene_name "DDX11L1"; transcript_type "transcribed_unprocessed_pseudogene"; transcript_name "DDX11L1-201"; exon_number 6; exon_id "ENSE00001863096.1"; level 2; transcript_support_level "NA"; hgnc_id "HGNC:37102"; ont "PGO:0000005"; ont "PGO:0000019"; tag "basic"; tag "Ensembl_canonical"; havana_gene "OTTHUMG00000000961.2"; havana_transcript "OTTHUMT00000002844.2";
chr1	HAVANA	gene	14404	29570	.	-	.	gene_id "ENSG00000227232.5"; gene_type "unprocessed_pseudogene"; gene_name "WASH7P"; level 2; hgnc_id "HGNC:38034"; havana_gene "OTTHUMG00000000958.1";
chr1	HAVANA	transcript	14404	29570	.	-	.	gene_id "ENSG00000227232.5"; transcript_id "ENST00000488147.1"; gene_type "unprocessed_pseudogene"; gene_name "WASH7P"; transcript_type "unprocessed_pseudogene"; transcript_name "WASH7P-201"; level 2; transcript_support_level "NA"; hgnc_id "HGNC:38034"; ont "PGO:0000005"; tag "basic"; tag "Ensembl_canonical"; havana_gene "OTTHUMG00000000958.1"; havana_transcript "OTTHUMT00000002839.1";
```

While this is already good to go for the leafcutter2 classify junctions script, we can also use the script to add NMDetectiveB attributes, output in a bed12 format (which might be easier to parse later than a gtf), and re-tag the gene_name and transcript_name attributes to be the same as the gene_id and transcript_id attributes.

I'm not including a large fasta file in this test_data directory, so you will have to download an appropriate fasta and appropriately reference it to use this script.

```bash
# Reformat the gtf with the script
python Reformat_gtf.py -i .test_data/gencode.gtf -o .test_data/gencode_reformatted.gtf -fa /project2/yangili1/bjf79/ReferenceGenomes/Human_UCSC.hg38_GencodeBasic46/Reference.fa -gene_name_attribute_name gene_id -transcript_name_attribute_name transcript_id -bed12_out .test_data/gencode.bed12

## and check the first 20 lines
head -n 20 .test_data/gencode_reformatted.gtf
```

Note the extra tags added to the transcript features, and the new gene_name and transcript_name attributes. The bed12 file is also created.

```
##description: evidence-based annotation of the human genome (GRCh38), version 43 (Ensembl 109)
##provider: GENCODE
##contact: gencode-help@ebi.ac.uk
##format: gtf
##date: 2022-11-29
#! args: Namespace(gtf_in='.test_data/gencode.gtf', gtf_out='.test_data/gencode_reformatted.gtf', input_type='gtf', fasta_in='/project2/yangili1/bjf79/ReferenceGenomes/Human_UCSC.hg38_GencodeBasic46/Reference.fa', bed12_out='.test_data/gencode.bed12', n_lines=None, infer_transcript_type_approach='A', infer_gene_type_approach='A', transcript_type_attribute_name='transcript_type', gene_type_attribute_name='gene_type', transcript_name_attribute_name='transcript_id', gene_name_attribute_name='gene_id', extra_attributes='transcript_support_level,tag,ccds_id', NMDetectiveB_coding_threshold=5, translation_approach='B', verbose=False)
chr1	input_gtf	gene	11869	14409	.	+	.	gene_name "ENSG00000290825.1"; gene_type "lncRNA";
chr1	input_gtf	transcript	11869	14409	0	+	.	transcript_name "ENST00000456328.2"; gene_name "ENSG00000290825.1"; transcript_type "lncRNA"; tag "NMDFinderB:No_CDS";transcript_name "ENST00000456328.2"; gene_name "ENSG00000290825.1"; transcript_type "lncRNA"; tag "NMDFinderB:No_CDS"; tag "FiveUTR_nEx":"0"; tag "FiveUTR_len":"0"; tag "ThreeUTR_nEx":"0"; tag "ThreeUTR_len":"0"; tag "ThreeUTR_nEx_AfterFirst50":"0"; tag "CDSLen":"0"; tag "Introns":"chr1:12227-12612:+,chr1:12721-13220:+"; gene_type "lncRNA";
chr1	input_gtf	exon	11869	12227	0	+	.	transcript_name "ENST00000456328.2"; gene_name "ENSG00000290825.1"; transcript_type "lncRNA"; tag "NMDFinderB:No_CDS"; exon_id "ENST00000456328.2_Exon001"; gene_type "lncRNA";
chr1	input_gtf	exon	12613	12721	0	+	.	transcript_name "ENST00000456328.2"; gene_name "ENSG00000290825.1"; transcript_type "lncRNA"; tag "NMDFinderB:No_CDS"; exon_id "ENST00000456328.2_Exon002"; gene_type "lncRNA";
chr1	input_gtf	exon	13221	14409	0	+	.	transcript_name "ENST00000456328.2"; gene_name "ENSG00000290825.1"; transcript_type "lncRNA"; tag "NMDFinderB:No_CDS"; exon_id "ENST00000456328.2_Exon003"; gene_type "lncRNA";
chr1	input_gtf	gene	12010	13670	.	+	.	gene_name "ENSG00000223972.6"; gene_type "transcribed_unprocessed_pseudogene";
chr1	input_gtf	transcript	12010	13670	0	+	.	transcript_name "ENST00000450305.2"; gene_name "ENSG00000223972.6"; transcript_type "transcribed_unprocessed_pseudogene"; tag "NMDFinderB:No_CDS";transcript_name "ENST00000450305.2"; gene_name "ENSG00000223972.6"; transcript_type "transcribed_unprocessed_pseudogene"; tag "NMDFinderB:No_CDS"; tag "FiveUTR_nEx":"0"; tag "FiveUTR_len":"0"; tag "ThreeUTR_nEx":"0"; tag "ThreeUTR_len":"0"; tag "ThreeUTR_nEx_AfterFirst50":"0"; tag "CDSLen":"0"; tag "Introns":"chr1:12057-12178:+,chr1:12227-12612:+,chr1:12697-12974:+,chr1:13052-13220:+,chr1:13374-13452:+"; gene_type "transcribed_unprocessed_pseudogene";
chr1	input_gtf	exon	12010	12057	0	+	.	transcript_name "ENST00000450305.2"; gene_name "ENSG00000223972.6"; transcript_type "transcribed_unprocessed_pseudogene"; tag "NMDFinderB:No_CDS"; exon_id "ENST00000450305.2_Exon001"; gene_type "transcribed_unprocessed_pseudogene";
chr1	input_gtf	exon	12179	12227	0	+	.	transcript_name "ENST00000450305.2"; gene_name "ENSG00000223972.6"; transcript_type "transcribed_unprocessed_pseudogene"; tag "NMDFinderB:No_CDS"; exon_id "ENST00000450305.2_Exon002"; gene_type "transcribed_unprocessed_pseudogene";
chr1	input_gtf	exon	12613	12697	0	+	.	transcript_name "ENST00000450305.2"; gene_name "ENSG00000223972.6"; transcript_type "transcribed_unprocessed_pseudogene"; tag "NMDFinderB:No_CDS"; exon_id "ENST00000450305.2_Exon003"; gene_type "transcribed_unprocessed_pseudogene";
chr1	input_gtf	exon	12975	13052	0	+	.	transcript_name "ENST00000450305.2"; gene_name "ENSG00000223972.6"; transcript_type "transcribed_unprocessed_pseudogene"; tag "NMDFinderB:No_CDS"; exon_id "ENST00000450305.2_Exon004"; gene_type "transcribed_unprocessed_pseudogene";
chr1	input_gtf	exon	13221	13374	0	+	.	transcript_name "ENST00000450305.2"; gene_name "ENSG00000223972.6"; transcript_type "transcribed_unprocessed_pseudogene"; tag "NMDFinderB:No_CDS"; exon_id "ENST00000450305.2_Exon005"; gene_type "transcribed_unprocessed_pseudogene";
chr1	input_gtf	exon	13453	13670	0	+	.	transcript_name "ENST00000450305.2"; gene_name "ENSG00000223972.6"; transcript_type "transcribed_unprocessed_pseudogene"; tag "NMDFinderB:No_CDS"; exon_id "ENST00000450305.2_Exon006"; gene_type "transcribed_unprocessed_pseudogene";
chr1	input_gtf	gene	14404	29570	.	-	.	gene_name "ENSG00000227232.5"; gene_type "unprocessed_pseudogene";
```

#### Human gtf, from UCSC/RefSeq
Gtf files downloaded from other sources (eg RefSeq, UCSC, Ensembl) or from different releases (eg Ensembl release v101, Ensembl release v75) can change a bit in format. The following example I downloaded a gtf file from UCSC for human (hg38), which from which it sources annotations from RefSeq and these gtf files are not formatted like the Gencode gtf files. The script can be used to reformat these gtf files as well. For gtf files actually downloaded from NCBI RefSeq, or from Ensembl, the original gtf format might be a bit different than this example but with some tweaking of the options, this script should be able to reformat them as well.

```bash
## first download the gencode gtf, perhaps just first 1000 lines for this example
wget -O- https://hgdownload.soe.ucsc.edu/goldenPath/hg38/bigZips/genes/hg38.ncbiRefSeq.gtf.gz | zcat | head -n 1000 > .test_data/refseq.gtf

## and check the first 20 lines
head -n 20 .test_data/refseq.gtf
```

```
chrM	ncbiRefSeq.2022-10-28	transcript	15956	16023	.	-	.	gene_id "TRNP"; transcript_id "rna-TRNP";  gene_name "TRNP";
chrM	ncbiRefSeq.2022-10-28	exon	15956	16023	.	-	.	gene_id "TRNP"; transcript_id "rna-TRNP"; exon_number "1"; exon_id "rna-TRNP.1"; gene_name "TRNP";
chrM	ncbiRefSeq.2022-10-28	transcript	15888	15953	.	+	.	gene_id "TRNT"; transcript_id "rna-TRNT";  gene_name "TRNT";
chrM	ncbiRefSeq.2022-10-28	exon	15888	15953	.	+	.	gene_id "TRNT"; transcript_id "rna-TRNT"; exon_number "1"; exon_id "rna-TRNT.1"; gene_name "TRNT";
chrM	ncbiRefSeq.2022-10-28	transcript	14747	15887	.	+	.	gene_id "CYTB"; transcript_id "rna-CYTB";  gene_name "CYTB";
chrM	ncbiRefSeq.2022-10-28	exon	14747	15887	.	+	.	gene_id "CYTB"; transcript_id "rna-CYTB"; exon_number "1"; exon_id "rna-CYTB.1"; gene_name "CYTB";
chrM	ncbiRefSeq.2022-10-28	CDS	14747	15887	.	+	0	gene_id "CYTB"; transcript_id "rna-CYTB"; exon_number "1"; exon_id "rna-CYTB.1"; gene_name "CYTB";
chrM	ncbiRefSeq.2022-10-28	start_codon	14747	14749	.	+	0	gene_id "CYTB"; transcript_id "rna-CYTB"; exon_number "1"; exon_id "rna-CYTB.1"; gene_name "CYTB";
chrM	ncbiRefSeq.2022-10-28	transcript	14674	14742	.	-	.	gene_id "TRNE"; transcript_id "rna-TRNE";  gene_name "TRNE";
chrM	ncbiRefSeq.2022-10-28	exon	14674	14742	.	-	.	gene_id "TRNE"; transcript_id "rna-TRNE"; exon_number "1"; exon_id "rna-TRNE.1"; gene_name "TRNE";
chrM	ncbiRefSeq.2022-10-28	transcript	14149	14673	.	-	.	gene_id "ND6"; transcript_id "rna-ND6";  gene_name "ND6";
chrM	ncbiRefSeq.2022-10-28	exon	14149	14673	.	-	.	gene_id "ND6"; transcript_id "rna-ND6"; exon_number "1"; exon_id "rna-ND6.1"; gene_name "ND6";
chrM	ncbiRefSeq.2022-10-28	CDS	14152	14673	.	-	0	gene_id "ND6"; transcript_id "rna-ND6"; exon_number "1"; exon_id "rna-ND6.1"; gene_name "ND6";
chrM	ncbiRefSeq.2022-10-28	start_codon	14671	14673	.	-	0	gene_id "ND6"; transcript_id "rna-ND6"; exon_number "1"; exon_id "rna-ND6.1"; gene_name "ND6";
chrM	ncbiRefSeq.2022-10-28	stop_codon	14149	14151	.	-	0	gene_id "ND6"; transcript_id "rna-ND6"; exon_number "1"; exon_id "rna-ND6.1"; gene_name "ND6";
chrM	ncbiRefSeq.2022-10-28	transcript	12337	14148	.	+	.	gene_id "ND5"; transcript_id "rna-ND5";  gene_name "ND5";
chrM	ncbiRefSeq.2022-10-28	exon	12337	14148	.	+	.	gene_id "ND5"; transcript_id "rna-ND5"; exon_number "1"; exon_id "rna-ND5.1"; gene_name "ND5";
chrM	ncbiRefSeq.2022-10-28	CDS	12337	14145	.	+	0	gene_id "ND5"; transcript_id "rna-ND5"; exon_number "1"; exon_id "rna-ND5.1"; gene_name "ND5";
chrM	ncbiRefSeq.2022-10-28	start_codon	12337	12339	.	+	0	gene_id "ND5"; transcript_id "rna-ND5"; exon_number "1"; exon_id "rna-ND5.1"; gene_name "ND5";
chrM	ncbiRefSeq.2022-10-28	stop_codon	14146	14148	.	+	0	gene_id "ND5"; transcript_id "rna-ND5"; exon_number "1"; exon_id "rna-ND5.1"; gene_name "ND5";
```

Note in this case there is no transcript_name, gene_type or transcript_type attributes. Let's use the script to add these attributes in the output. For defining the gene_type and transcript_type, there are some options to choose from (see the help message above).

```bash
# Reformat the gtf with the script
python Reformat_gtf.py -i .test_data/refseq.gtf -o .test_data/refseq_reformatted.gtf -fa /project2/yangili1/bjf79/ReferenceGenomes/Human_UCSC.hg38_GencodeBasic46/Reference.fa -gene_name_attribute_name gene_id -transcript_name_attribute_name transcript_id -infer_transcript_type_approach B -infer_gene_type_approach B

## and check the first 20 lines
head -n 20 .test_data/refseq_reformatted.gtf
```

```
#! args: Namespace(gtf_in='.test_data/refseq.gtf', gtf_out='.test_data/refseq_reformatted.gtf', input_type='gtf', fasta_in='/project2/yangili1/bjf79/ReferenceGenomes/Human_UCSC.hg38_GencodeBasic46/Reference.fa', bed12_out=None, n_lines=None, infer_transcript_type_approach='B', infer_gene_type_approach='B', transcript_type_attribute_name='transcript_type', gene_type_attribute_name='gene_type', transcript_name_attribute_name='transcript_id', gene_name_attribute_name='gene_id', extra_attributes='transcript_support_level,tag,ccds_id', NMDetectiveB_coding_threshold=5, translation_approach='B', verbose=False)
chr19_GL000209v2_alt	input_gtf	gene	2842	11554	.	-	.	gene_name "KIR3DL2"; gene_type "protein_coding";
chr19_GL000209v2_alt	input_gtf	transcript	2842	11554	0	-	.	transcript_name "NM_006737.4"; gene_name "KIR3DL2"; transcript_type "protein_coding"; tag "NMDFinderB:Last_exon";transcript_name "NM_006737.4"; gene_name "KIR3DL2"; transcript_type "protein_coding"; tag "NMDFinderB:Last_exon"; tag "FiveUTR_nEx":"1"; tag "FiveUTR_len":"33"; tag "ThreeUTR_nEx":"0"; tag "ThreeUTR_len":"0"; tag "ThreeUTR_nEx_AfterFirst50":"0"; tag "CDSLen":"1002"; tag "Introns":"chr19_GL000209v2_alt:2894-6058:-,chr19_GL000209v2_alt:6352-7931:-,chr19_GL000209v2_alt:8231-9695:-,chr19_GL000209v2_alt:9980-10722:-,chr19_GL000209v2_alt:10758-11487:-"; gene_type "protein_coding";
chr19_GL000209v2_alt	input_gtf	exon	2842	2894	0	-	.	transcript_name "NM_006737.4"; gene_name "KIR3DL2"; transcript_type "protein_coding"; tag "NMDFinderB:Last_exon"; exon_id "NM_006737.4_Exon001"; gene_type "protein_coding";
chr19_GL000209v2_alt	input_gtf	CDS	2842	2894	0	-	2	transcript_name "NM_006737.4"; gene_name "KIR3DL2"; transcript_type "protein_coding"; tag "NMDFinderB:Last_exon";transcript_name "NM_006737.4"; gene_name "KIR3DL2"; transcript_type "protein_coding"; tag "NMDFinderB:Last_exon"; gene_type "protein_coding";
chr19_GL000209v2_alt	input_gtf	stop_codon	2842	2844	0	-	0	transcript_name "NM_006737.4"; gene_name "KIR3DL2"; transcript_type "protein_coding"; tag "NMDFinderB:Last_exon"; gene_type "protein_coding";
chr19_GL000209v2_alt	input_gtf	exon	6059	6352	0	-	.	transcript_name "NM_006737.4"; gene_name "KIR3DL2"; transcript_type "protein_coding"; tag "NMDFinderB:Last_exon"; exon_id "NM_006737.4_Exon002"; gene_type "protein_coding";
chr19_GL000209v2_alt	input_gtf	CDS	6059	6352	0	-	2	transcript_name "NM_006737.4"; gene_name "KIR3DL2"; transcript_type "protein_coding"; tag "NMDFinderB:Last_exon";transcript_name "NM_006737.4"; gene_name "KIR3DL2"; transcript_type "protein_coding"; tag "NMDFinderB:Last_exon"; gene_type "protein_coding";
chr19_GL000209v2_alt	input_gtf	exon	7932	8231	0	-	.	transcript_name "NM_006737.4"; gene_name "KIR3DL2"; transcript_type "protein_coding"; tag "NMDFinderB:Last_exon"; exon_id "NM_006737.4_Exon003"; gene_type "protein_coding";
chr19_GL000209v2_alt	input_gtf	CDS	7932	8231	0	-	2	transcript_name "NM_006737.4"; gene_name "KIR3DL2"; transcript_type "protein_coding"; tag "NMDFinderB:Last_exon";transcript_name "NM_006737.4"; gene_name "KIR3DL2"; transcript_type "protein_coding"; tag "NMDFinderB:Last_exon"; gene_type "protein_coding";
chr19_GL000209v2_alt	input_gtf	exon	9696	9980	0	-	.	transcript_name "NM_006737.4"; gene_name "KIR3DL2"; transcript_type "protein_coding"; tag "NMDFinderB:Last_exon"; exon_id "NM_006737.4_Exon004"; gene_type "protein_coding";
chr19_GL000209v2_alt	input_gtf	CDS	9696	9980	0	-	2	transcript_name "NM_006737.4"; gene_name "KIR3DL2"; transcript_type "protein_coding"; tag "NMDFinderB:Last_exon";transcript_name "NM_006737.4"; gene_name "KIR3DL2"; transcript_type "protein_coding"; tag "NMDFinderB:Last_exon"; gene_type "protein_coding";
chr19_GL000209v2_alt	input_gtf	exon	10723	10758	0	-	.	transcript_name "NM_006737.4"; gene_name "KIR3DL2"; transcript_type "protein_coding"; tag "NMDFinderB:Last_exon"; exon_id "NM_006737.4_Exon005"; gene_type "protein_coding";
chr19_GL000209v2_alt	input_gtf	CDS	10723	10758	0	-	2	transcript_name "NM_006737.4"; gene_name "KIR3DL2"; transcript_type "protein_coding"; tag "NMDFinderB:Last_exon";transcript_name "NM_006737.4"; gene_name "KIR3DL2"; transcript_type "protein_coding"; tag "NMDFinderB:Last_exon"; gene_type "protein_coding";
chr19_GL000209v2_alt	input_gtf	exon	11488	11554	0	-	.	transcript_name "NM_006737.4"; gene_name "KIR3DL2"; transcript_type "protein_coding"; tag "NMDFinderB:Last_exon"; exon_id "NM_006737.4_Exon006"; gene_type "protein_coding";
chr19_GL000209v2_alt	input_gtf	CDS	11488	11521	0	-	0	transcript_name "NM_006737.4"; gene_name "KIR3DL2"; transcript_type "protein_coding"; tag "NMDFinderB:Last_exon";transcript_name "NM_006737.4"; gene_name "KIR3DL2"; transcript_type "protein_coding"; tag "NMDFinderB:Last_exon"; gene_type "protein_coding";
chr19_GL000209v2_alt	input_gtf	start_codon	11519	11521	0	-	0	transcript_name "NM_006737.4"; gene_name "KIR3DL2"; transcript_type "protein_coding"; tag "NMDFinderB:Last_exon"; gene_type "protein_coding";
chr19_GL000209v2_alt	input_gtf	UTR	11522	11554	0	-	.	transcript_name "NM_006737.4"; gene_name "KIR3DL2"; transcript_type "protein_coding"; tag "NMDFinderB:Last_exon"; gene_type "protein_coding";
chr19_GL000209v2_alt	input_gtf	transcript	6057	11554	0	-	.	transcript_name "NM_001242867.2"; gene_name "KIR3DL2"; transcript_type "protein_coding"; tag "NMDFinderB:Last_exon";transcript_name "NM_001242867.2"; gene_name "KIR3DL2"; transcript_type "protein_coding"; tag "NMDFinderB:Last_exon"; tag "FiveUTR_nEx":"1"; tag "FiveUTR_len":"33"; tag "ThreeUTR_nEx":"0"; tag "ThreeUTR_len":"0"; tag "ThreeUTR_nEx_AfterFirst50":"0"; tag "CDSLen":"951"; tag "Introns":"chr19_GL000209v2_alt:6352-7931:-,chr19_GL000209v2_alt:8231-9695:-,chr19_GL000209v2_alt:9980-10722:-,chr19_GL000209v2_alt:10758-11487:-"; gene_type "protein_coding";
chr19_GL000209v2_alt	input_gtf	exon	6057	6352	0	-	.	transcript_name "NM_001242867.2"; gene_name "KIR3DL2"; transcript_type "protein_coding"; tag "NMDFinderB:Last_exon"; exon_id "NM_001242867.2_Exon001"; gene_type "protein_coding";
```

#### StringTie
Finally, let's do an example using the output of stringtie, which does not as far as I know contain CDS features, or start_codon or stop_codon features, or transcript_type or gene_type attributes. There are various tools downstream of stringtie that can be used to better annotate gtf files (AGAT, sqanti3, etc), but this script can also be used to add these attributes. Here is an example

```
#First look at the example stringtie-generated gtf file for a small region of the genome
cat .test_data/stringtie.gtf
```

Note the lack of gene, CDS, start_codon, stop_codon, gene_type and transcript_type attributes. The gene_name and transcript_name attributes are also not the same as the gene_id and transcript_id attributes. So given transcript boundaries, one can infer open reading frame, and add the missing attributes, and consider whether transcript is predicted NMD target. Using the script options (see `--help` output above), we can add these attributes to the gtf file, with a little bit of flexibility in the options.

```
# stringtie -o scratch/test.merge.gtf --fr -G /project2/yangili1/bjf79/ChromatinSplicingQTLs/code/ReferenceGenome/Annotations/gencode.v34.chromasomal.annotation.gtf --ref /project2/yangili1/bjf79/ChromatinSplicingQTLs/code/ReferenceGenome/Fasta/GRCh38.primary_assembly.genome.fa scratch/test.merge.bam
# StringTie version 2.2.1
chr12	StringTie	transcript	131894622	131923150	1000	+	.	gene_id "STRG.1"; transcript_id "STRG.1.1"; reference_id "ENST00000321867.6"; ref_gene_id "ENSG00000177169.10"; ref_gene_name "ULK1"; cov "59125.789062"; FPKM "99528.601562"; TPM "498897.531250";
chr12	StringTie	exon	131894622	131895112	1000	+	.	gene_id "STRG.1"; transcript_id "STRG.1.1"; exon_number "1"; reference_id "ENST00000321867.6"; ref_gene_id "ENSG00000177169.10"; ref_gene_name "ULK1"; cov "5596.674316";
chr12	StringTie	exon	131895601	131895693	1000	+	.	gene_id "STRG.1"; transcript_id "STRG.1.1"; exon_number "2"; reference_id "ENST00000321867.6"; ref_gene_id "ENSG00000177169.10"; ref_gene_name "ULK1"; cov "31352.257812";
chr12	StringTie	exon	131895783	131895824	1000	+	.	gene_id "STRG.1"; transcript_id "STRG.1.1"; exon_number "3"; reference_id "ENST00000321867.6"; ref_gene_id "ENSG00000177169.10"; ref_gene_name "ULK1"; cov "60338.976562";
chr12	StringTie	exon	131906892	131906924	1000	+	.	gene_id "STRG.1"; transcript_id "STRG.1.1"; exon_number "4"; reference_id "ENST00000321867.6"; ref_gene_id "ENSG00000177169.10"; ref_gene_name "ULK1"; cov "66670.695312";
chr12	StringTie	exon	131907495	131907531	1000	+	.	gene_id "STRG.1"; transcript_id "STRG.1.1"; exon_number "5"; reference_id "ENST00000321867.6"; ref_gene_id "ENSG00000177169.10"; ref_gene_name "ULK1"; cov "72943.625000";
chr12	StringTie	exon	131908644	131908817	1000	+	.	gene_id "STRG.1"; transcript_id "STRG.1.1"; exon_number "6"; reference_id "ENST00000321867.6"; ref_gene_id "ENSG00000177169.10"; ref_gene_name "ULK1"; cov "70412.312500";
chr12	StringTie	exon	131908898	131908971	1000	+	.	gene_id "STRG.1"; transcript_id "STRG.1.1"; exon_number "7"; reference_id "ENST00000321867.6"; ref_gene_id "ENSG00000177169.10"; ref_gene_name "ULK1"; cov "67140.031250";
chr12	StringTie	exon	131909136	131909237	1000	+	.	gene_id "STRG.1"; transcript_id "STRG.1.1"; exon_number "8"; reference_id "ENST00000321867.6"; ref_gene_id "ENSG00000177169.10"; ref_gene_name "ULK1"; cov "58031.765625";
chr12	StringTie	exon	131909775	131909833	1000	+	.	gene_id "STRG.1"; transcript_id "STRG.1.1"; exon_number "9"; reference_id "ENST00000321867.6"; ref_gene_id "ENSG00000177169.10"; ref_gene_name "ULK1"; cov "52475.296875";
chr12	StringTie	exon	131909919	131910001	1000	+	.	gene_id "STRG.1"; transcript_id "STRG.1.1"; exon_number "10"; reference_id "ENST00000321867.6"; ref_gene_id "ENSG00000177169.10"; ref_gene_name "ULK1"; cov "53820.222656";
chr12	StringTie	exon	131910254	131910304	1000	+	.	gene_id "STRG.1"; transcript_id "STRG.1.1"; exon_number "11"; reference_id "ENST00000321867.6"; ref_gene_id "ENSG00000177169.10"; ref_gene_name "ULK1"; cov "52807.921875";
chr12	StringTie	exon	131910712	131910800	1000	+	.	gene_id "STRG.1"; transcript_id "STRG.1.1"; exon_number "12"; reference_id "ENST00000321867.6"; ref_gene_id "ENSG00000177169.10"; ref_gene_name "ULK1"; cov "55465.664062";
chr12	StringTie	exon	131911942	131912089	1000	+	.	gene_id "STRG.1"; transcript_id "STRG.1.1"; exon_number "13"; reference_id "ENST00000321867.6"; ref_gene_id "ENSG00000177169.10"; ref_gene_name "ULK1"; cov "56328.164062";
chr12	StringTie	exon	131913198	131913258	1000	+	.	gene_id "STRG.1"; transcript_id "STRG.1.1"; exon_number "14"; reference_id "ENST00000321867.6"; ref_gene_id "ENSG00000177169.10"; ref_gene_name "ULK1"; cov "56330.296875";
chr12	StringTie	exon	131913747	131913836	1000	+	.	gene_id "STRG.1"; transcript_id "STRG.1.1"; exon_number "15"; reference_id "ENST00000321867.6"; ref_gene_id "ENSG00000177169.10"; ref_gene_name "ULK1"; cov "59891.199219";
chr12	StringTie	exon	131914352	131914477	1000	+	.	gene_id "STRG.1"; transcript_id "STRG.1.1"; exon_number "16"; reference_id "ENST00000321867.6"; ref_gene_id "ENSG00000177169.10"; ref_gene_name "ULK1"; cov "65586.796875";
chr12	StringTie	exon	131915083	131915231	1000	+	.	gene_id "STRG.1"; transcript_id "STRG.1.1"; exon_number "17"; reference_id "ENST00000321867.6"; ref_gene_id "ENSG00000177169.10"; ref_gene_name "ULK1"; cov "65277.812500";
chr12	StringTie	exon	131915335	131915421	1000	+	.	gene_id "STRG.1"; transcript_id "STRG.1.1"; exon_number "18"; reference_id "ENST00000321867.6"; ref_gene_id "ENSG00000177169.10"; ref_gene_name "ULK1"; cov "60823.355469";
chr12	StringTie	exon	131915891	131916159	1000	+	.	gene_id "STRG.1"; transcript_id "STRG.1.1"; exon_number "19"; reference_id "ENST00000321867.6"; ref_gene_id "ENSG00000177169.10"; ref_gene_name "ULK1"; cov "60507.070312";
chr12	StringTie	exon	131916398	131916591	1000	+	.	gene_id "STRG.1"; transcript_id "STRG.1.1"; exon_number "20"; reference_id "ENST00000321867.6"; ref_gene_id "ENSG00000177169.10"; ref_gene_name "ULK1"; cov "62592.519531";
chr12	StringTie	exon	131916953	131917062	1000	+	.	gene_id "STRG.1"; transcript_id "STRG.1.1"; exon_number "21"; reference_id "ENST00000321867.6"; ref_gene_id "ENSG00000177169.10"; ref_gene_name "ULK1"; cov "59059.410156";
chr12	StringTie	exon	131917411	131917554	1000	+	.	gene_id "STRG.1"; transcript_id "STRG.1.1"; exon_number "22"; reference_id "ENST00000321867.6"; ref_gene_id "ENSG00000177169.10"; ref_gene_name "ULK1"; cov "58239.757812";
chr12	StringTie	exon	131918497	131918681	1000	+	.	gene_id "STRG.1"; transcript_id "STRG.1.1"; exon_number "23"; reference_id "ENST00000321867.6"; ref_gene_id "ENSG00000177169.10"; ref_gene_name "ULK1"; cov "53761.750000";
chr12	StringTie	exon	131919212	131919384	1000	+	.	gene_id "STRG.1"; transcript_id "STRG.1.1"; exon_number "24"; reference_id "ENST00000321867.6"; ref_gene_id "ENSG00000177169.10"; ref_gene_name "ULK1"; cov "55082.761719";
chr12	StringTie	exon	131919472	131919590	1000	+	.	gene_id "STRG.1"; transcript_id "STRG.1.1"; exon_number "25"; reference_id "ENST00000321867.6"; ref_gene_id "ENSG00000177169.10"; ref_gene_name "ULK1"; cov "66974.125000";
chr12	StringTie	exon	131919979	131920136	1000	+	.	gene_id "STRG.1"; transcript_id "STRG.1.1"; exon_number "26"; reference_id "ENST00000321867.6"; ref_gene_id "ENSG00000177169.10"; ref_gene_name "ULK1"; cov "71130.859375";
chr12	StringTie	exon	131921100	131921235	1000	+	.	gene_id "STRG.1"; transcript_id "STRG.1.1"; exon_number "27"; reference_id "ENST00000321867.6"; ref_gene_id "ENSG00000177169.10"; ref_gene_name "ULK1"; cov "80458.429688";
chr12	StringTie	exon	131921306	131923150	1000	+	.	gene_id "STRG.1"; transcript_id "STRG.1.1"; exon_number "28"; reference_id "ENST00000321867.6"; ref_gene_id "ENSG00000177169.10"; ref_gene_name "ULK1"; cov "70387.460938";
chr12	StringTie	transcript	131896500	131898239	1000	+	.	gene_id "STRG.2"; transcript_id "STRG.2.1"; reference_id "ENST00000624048.1"; ref_gene_id "ENSG00000279283.1"; ref_gene_name "AC131009.4"; cov "1203.300049"; FPKM "2025.558960"; TPM "10153.327148";
chr12	StringTie	exon	131896500	131898239	1000	+	.	gene_id "STRG.2"; transcript_id "STRG.2.1"; exon_number "1"; reference_id "ENST00000624048.1"; ref_gene_id "ENSG00000279283.1"; ref_gene_name "AC131009.4"; cov "1203.300049";
chr12	StringTie	transcript	131906490	131909996	1000	+	.	gene_id "STRG.1"; transcript_id "STRG.1.2"; reference_id "ENST00000537421.5"; ref_gene_id "ENSG00000177169.10"; ref_gene_name "ULK1"; cov "591.564575"; FPKM "995.802246"; TPM "4991.562988";
chr12	StringTie	exon	131906490	131906924	1000	+	.	gene_id "STRG.1"; transcript_id "STRG.1.2"; exon_number "1"; reference_id "ENST00000537421.5"; ref_gene_id "ENSG00000177169.10"; ref_gene_name "ULK1"; cov "1295.367798";
chr12	StringTie	exon	131907495	131907531	1000	+	.	gene_id "STRG.1"; transcript_id "STRG.1.2"; exon_number "2"; reference_id "ENST00000537421.5"; ref_gene_id "ENSG00000177169.10"; ref_gene_name "ULK1"; cov "8.420218";
chr12	StringTie	exon	131908644	131908817	1000	+	.	gene_id "STRG.1"; transcript_id "STRG.1.2"; exon_number "3"; reference_id "ENST00000537421.5"; ref_gene_id "ENSG00000177169.10"; ref_gene_name "ULK1"; cov "8.128017";
chr12	StringTie	exon	131908898	131908971	1000	+	.	gene_id "STRG.1"; transcript_id "STRG.1.2"; exon_number "4"; reference_id "ENST00000537421.5"; ref_gene_id "ENSG00000177169.10"; ref_gene_name "ULK1"; cov "7.750283";
chr12	StringTie	exon	131909136	131909237	1000	+	.	gene_id "STRG.1"; transcript_id "STRG.1.2"; exon_number "5"; reference_id "ENST00000537421.5"; ref_gene_id "ENSG00000177169.10"; ref_gene_name "ULK1"; cov "6.698874";
chr12	StringTie	exon	131909775	131909833	1000	+	.	gene_id "STRG.1"; transcript_id "STRG.1.2"; exon_number "6"; reference_id "ENST00000537421.5"; ref_gene_id "ENSG00000177169.10"; ref_gene_name "ULK1"; cov "6.057465";
chr12	StringTie	exon	131909919	131909996	1000	+	.	gene_id "STRG.1"; transcript_id "STRG.1.2"; exon_number "7"; reference_id "ENST00000537421.5"; ref_gene_id "ENSG00000177169.10"; ref_gene_name "ULK1"; cov "6.223817";
chr12	StringTie	transcript	131908619	131909169	1000	+	.	gene_id "STRG.1"; transcript_id "STRG.1.3"; reference_id "ENST00000542313.2"; ref_gene_id "ENSG00000177169.10"; ref_gene_name "ULK1"; cov "45.668560"; FPKM "76.875557"; TPM "385.346771";
chr12	StringTie	exon	131908619	131908817	1000	+	.	gene_id "STRG.1"; transcript_id "STRG.1.3"; exon_number "1"; reference_id "ENST00000542313.2"; ref_gene_id "ENSG00000177169.10"; ref_gene_name "ULK1"; cov "69.606125";
chr12	StringTie	exon	131908898	131908971	1000	+	.	gene_id "STRG.1"; transcript_id "STRG.1.3"; exon_number "2"; reference_id "ENST00000542313.2"; ref_gene_id "ENSG00000177169.10"; ref_gene_name "ULK1"; cov "1.620012";
chr12	StringTie	exon	131909136	131909169	1000	+	.	gene_id "STRG.1"; transcript_id "STRG.1.3"; exon_number "3"; reference_id "ENST00000542313.2"; ref_gene_id "ENSG00000177169.10"; ref_gene_name "ULK1"; cov "1.433761";
chr12	StringTie	transcript	131915943	131919385	1000	+	.	gene_id "STRG.1"; transcript_id "STRG.1.4"; reference_id "ENST00000541761.2"; ref_gene_id "ENSG00000177169.10"; ref_gene_name "ULK1"; cov "171.283447"; FPKM "288.327698"; TPM "1445.272705";
chr12	StringTie	exon	131915943	131916591	1000	+	.	gene_id "STRG.1"; transcript_id "STRG.1.4"; exon_number "1"; reference_id "ENST00000541761.2"; ref_gene_id "ENSG00000177169.10"; ref_gene_name "ULK1"; cov "332.826508";
chr12	StringTie	exon	131916953	131917062	1000	+	.	gene_id "STRG.1"; transcript_id "STRG.1.4"; exon_number "2"; reference_id "ENST00000541761.2"; ref_gene_id "ENSG00000177169.10"; ref_gene_name "ULK1"; cov "0.189004";
chr12	StringTie	exon	131917411	131917554	1000	+	.	gene_id "STRG.1"; transcript_id "STRG.1.4"; exon_number "3"; reference_id "ENST00000541761.2"; ref_gene_id "ENSG00000177169.10"; ref_gene_name "ULK1"; cov "0.186381";
chr12	StringTie	exon	131918497	131918681	1000	+	.	gene_id "STRG.1"; transcript_id "STRG.1.4"; exon_number "4"; reference_id "ENST00000541761.2"; ref_gene_id "ENSG00000177169.10"; ref_gene_name "ULK1"; cov "0.172050";
chr12	StringTie	exon	131919212	131919385	1000	+	.	gene_id "STRG.1"; transcript_id "STRG.1.4"; exon_number "5"; reference_id "ENST00000541761.2"; ref_gene_id "ENSG00000177169.10"; ref_gene_name "ULK1"; cov "0.435755";
chr12	StringTie	transcript	131918302	131921550	1000	+	.	gene_id "STRG.1"; transcript_id "STRG.1.5"; reference_id "ENST00000540647.5"; ref_gene_id "ENSG00000177169.10"; ref_gene_name "ULK1"; cov "1337.152100"; FPKM "2250.876953"; TPM "11282.756836";
chr12	StringTie	exon	131918302	131918681	1000	+	.	gene_id "STRG.1"; transcript_id "STRG.1.5"; exon_number "1"; reference_id "ENST00000540647.5"; ref_gene_id "ENSG00000177169.10"; ref_gene_name "ULK1"; cov "3816.895508";
chr12	StringTie	exon	131919212	131919384	1000	+	.	gene_id "STRG.1"; transcript_id "STRG.1.5"; exon_number "2"; reference_id "ENST00000540647.5"; ref_gene_id "ENSG00000177169.10"; ref_gene_name "ULK1"; cov "154.429565";
chr12	StringTie	exon	131919472	131919590	1000	+	.	gene_id "STRG.1"; transcript_id "STRG.1.5"; exon_number "3"; reference_id "ENST00000540647.5"; ref_gene_id "ENSG00000177169.10"; ref_gene_name "ULK1"; cov "187.768082";
chr12	StringTie	exon	131919979	131920136	1000	+	.	gene_id "STRG.1"; transcript_id "STRG.1.5"; exon_number "4"; reference_id "ENST00000540647.5"; ref_gene_id "ENSG00000177169.10"; ref_gene_name "ULK1"; cov "199.421890";
chr12	StringTie	exon	131921100	131921235	1000	+	.	gene_id "STRG.1"; transcript_id "STRG.1.5"; exon_number "5"; reference_id "ENST00000540647.5"; ref_gene_id "ENSG00000177169.10"; ref_gene_name "ULK1"; cov "225.572571";
chr12	StringTie	exon	131921306	131921550	1000	+	.	gene_id "STRG.1"; transcript_id "STRG.1.5"; exon_number "6"; reference_id "ENST00000540647.5"; ref_gene_id "ENSG00000177169.10"; ref_gene_name "ULK1"; cov "235.199020";
chr12	StringTie	transcript	131919241	131921951	1000	+	.	gene_id "STRG.1"; transcript_id "STRG.1.6"; reference_id "ENST00000544718.1"; ref_gene_id "ENSG00000177169.10"; ref_gene_name "ULK1"; cov "270.849304"; FPKM "455.930542"; TPM "2285.399658";
chr12	StringTie	exon	131919241	131920136	1000	+	.	gene_id "STRG.1"; transcript_id "STRG.1.6"; exon_number "1"; reference_id "ENST00000544718.1"; ref_gene_id "ENSG00000177169.10"; ref_gene_name "ULK1"; cov "506.459167";
chr12	StringTie	exon	131921100	131921235	1000	+	.	gene_id "STRG.1"; transcript_id "STRG.1.6"; exon_number "2"; reference_id "ENST00000544718.1"; ref_gene_id "ENSG00000177169.10"; ref_gene_name "ULK1"; cov "0.878679";
chr12	StringTie	exon	131921306	131921951	1000	+	.	gene_id "STRG.1"; transcript_id "STRG.1.6"; exon_number "3"; reference_id "ENST00000544718.1"; ref_gene_id "ENSG00000177169.10"; ref_gene_name "ULK1"; cov "0.895097";
chr12	StringTie	transcript	131920067	131922281	1000	+	.	gene_id "STRG.1"; transcript_id "STRG.1.7"; reference_id "ENST00000540568.1"; ref_gene_id "ENSG00000177169.10"; ref_gene_name "ULK1"; cov "1541.480103"; FPKM "2594.829834"; TPM "13006.856445";
chr12	StringTie	exon	131920067	131921235	1000	+	.	gene_id "STRG.1"; transcript_id "STRG.1.7"; exon_number "1"; reference_id "ENST00000540568.1"; ref_gene_id "ENSG00000177169.10"; ref_gene_name "ULK1"; cov "2827.322754";
chr12	StringTie	exon	131921306	131922281	1000	+	.	gene_id "STRG.1"; transcript_id "STRG.1.7"; exon_number "2"; reference_id "ENST00000540568.1"; ref_gene_id "ENSG00000177169.10"; ref_gene_name "ULK1"; cov "1.367344";
chr12	StringTie	transcript	131924736	131929351	1000	-	.	gene_id "STRG.3"; transcript_id "STRG.3.1"; reference_id "ENST00000539078.1"; ref_gene_id "ENSG00000255992.1"; ref_gene_name "AC131009.1"; cov "631.823914"; FPKM "1063.572266"; TPM "5331.267578";
chr12	StringTie	exon	131924736	131924766	1000	-	.	gene_id "STRG.3"; transcript_id "STRG.3.1"; exon_number "1"; reference_id "ENST00000539078.1"; ref_gene_id "ENSG00000255992.1"; ref_gene_name "AC131009.1"; cov "434.612885";
chr12	StringTie	exon	131927963	131928190	1000	-	.	gene_id "STRG.3"; transcript_id "STRG.3.1"; exon_number "2"; reference_id "ENST00000539078.1"; ref_gene_id "ENSG00000255992.1"; ref_gene_name "AC131009.1"; cov "795.434265";
chr12	StringTie	exon	131929185	131929351	1000	-	.	gene_id "STRG.3"; transcript_id "STRG.3.1"; exon_number "3"; reference_id "ENST00000539078.1"; ref_gene_id "ENSG00000255992.1"; ref_gene_name "AC131009.1"; cov "445.059753";
chr12	StringTie	transcript	131929200	131943859	1000	+	.	gene_id "STRG.4"; transcript_id "STRG.4.1"; reference_id "ENST00000443358.6"; ref_gene_id "ENSG00000177192.14"; ref_gene_name "PUS1"; cov "6788.194336"; FPKM "11426.815430"; TPM "57278.109375";
chr12	StringTie	exon	131929200	131929380	1000	+	.	gene_id "STRG.4"; transcript_id "STRG.4.1"; exon_number "1"; reference_id "ENST00000443358.6"; ref_gene_id "ENSG00000177192.14"; ref_gene_name "PUS1"; cov "1060.815552";
chr12	StringTie	exon	131929907	131930135	1000	+	.	gene_id "STRG.4"; transcript_id "STRG.4.1"; exon_number "2"; reference_id "ENST00000443358.6"; ref_gene_id "ENSG00000177192.14"; ref_gene_name "PUS1"; cov "6163.019043";
chr12	StringTie	exon	131932175	131932312	1000	+	.	gene_id "STRG.4"; transcript_id "STRG.4.1"; exon_number "3"; reference_id "ENST00000443358.6"; ref_gene_id "ENSG00000177192.14"; ref_gene_name "PUS1"; cov "7207.108887";
chr12	StringTie	exon	131939173	131939275	1000	+	.	gene_id "STRG.4"; transcript_id "STRG.4.1"; exon_number "4"; reference_id "ENST00000443358.6"; ref_gene_id "ENSG00000177192.14"; ref_gene_name "PUS1"; cov "6596.350586";
chr12	StringTie	exon	131941292	131941983	1000	+	.	gene_id "STRG.4"; transcript_id "STRG.4.1"; exon_number "5"; reference_id "ENST00000443358.6"; ref_gene_id "ENSG00000177192.14"; ref_gene_name "PUS1"; cov "10554.355469";
chr12	StringTie	exon	131943539	131943859	1000	+	.	gene_id "STRG.4"; transcript_id "STRG.4.1"; exon_number "6"; reference_id "ENST00000443358.6"; ref_gene_id "ENSG00000177192.14"; ref_gene_name "PUS1"; cov "2226.160156";
chr12	StringTie	transcript	131929276	131945896	1000	+	.	gene_id "STRG.4"; transcript_id "STRG.4.2"; reference_id "ENST00000376649.8"; ref_gene_id "ENSG00000177192.14"; ref_gene_name "PUS1"; cov "31503.429688"; FPKM "53030.875000"; TPM "265822.812500";
chr12	StringTie	exon	131929276	131929796	1000	+	.	gene_id "STRG.4"; transcript_id "STRG.4.2"; exon_number "1"; reference_id "ENST00000376649.8"; ref_gene_id "ENSG00000177192.14"; ref_gene_name "PUS1"; cov "21614.521484";
chr12	StringTie	exon	131929907	131930135	1000	+	.	gene_id "STRG.4"; transcript_id "STRG.4.2"; exon_number "2"; reference_id "ENST00000376649.8"; ref_gene_id "ENSG00000177192.14"; ref_gene_name "PUS1"; cov "64407.667969";
chr12	StringTie	exon	131932175	131932312	1000	+	.	gene_id "STRG.4"; transcript_id "STRG.4.2"; exon_number "3"; reference_id "ENST00000376649.8"; ref_gene_id "ENSG00000177192.14"; ref_gene_name "PUS1"; cov "75319.109375";
chr12	StringTie	exon	131939173	131939275	1000	+	.	gene_id "STRG.4"; transcript_id "STRG.4.2"; exon_number "4"; reference_id "ENST00000376649.8"; ref_gene_id "ENSG00000177192.14"; ref_gene_name "PUS1"; cov "68936.273438";
chr12	StringTie	exon	131941292	131941983	1000	+	.	gene_id "STRG.4"; transcript_id "STRG.4.2"; exon_number "5"; reference_id "ENST00000376649.8"; ref_gene_id "ENSG00000177192.14"; ref_gene_name "PUS1"; cov "110300.078125";
chr12	StringTie	exon	131943539	131945896	1000	+	.	gene_id "STRG.4"; transcript_id "STRG.4.2"; exon_number "6"; reference_id "ENST00000376649.8"; ref_gene_id "ENSG00000177192.14"; ref_gene_name "PUS1"; cov "3169.084473";
chr12	StringTie	transcript	131929276	131943859	1000	+	.	gene_id "STRG.4"; transcript_id "STRG.4.3"; reference_id "ENST00000542167.2"; ref_gene_id "ENSG00000177192.14"; ref_gene_name "PUS1"; cov "769.425720"; FPKM "1295.202393"; TPM "6492.337891";
chr12	StringTie	exon	131929276	131929512	1000	+	.	gene_id "STRG.4"; transcript_id "STRG.4.3"; exon_number "1"; reference_id "ENST00000542167.2"; ref_gene_id "ENSG00000177192.14"; ref_gene_name "PUS1"; cov "3.251697";
chr12	StringTie	exon	131931515	131932312	1000	+	.	gene_id "STRG.4"; transcript_id "STRG.4.3"; exon_number "2"; reference_id "ENST00000542167.2"; ref_gene_id "ENSG00000177192.14"; ref_gene_name "PUS1"; cov "2053.054199";
chr12	StringTie	exon	131939173	131939275	1000	+	.	gene_id "STRG.4"; transcript_id "STRG.4.3"; exon_number "3"; reference_id "ENST00000542167.2"; ref_gene_id "ENSG00000177192.14"; ref_gene_name "PUS1"; cov "12.079113";
chr12	StringTie	exon	131941292	131941983	1000	+	.	gene_id "STRG.4"; transcript_id "STRG.4.3"; exon_number "4"; reference_id "ENST00000542167.2"; ref_gene_id "ENSG00000177192.14"; ref_gene_name "PUS1"; cov "19.326939";
chr12	StringTie	exon	131943539	131943859	1000	+	.	gene_id "STRG.4"; transcript_id "STRG.4.3"; exon_number "5"; reference_id "ENST00000542167.2"; ref_gene_id "ENSG00000177192.14"; ref_gene_name "PUS1"; cov "4.076503";
chr12	StringTie	transcript	131929282	131939275	1000	+	.	gene_id "STRG.4"; transcript_id "STRG.4.4"; reference_id "ENST00000538037.5"; ref_gene_id "ENSG00000177192.14"; ref_gene_name "PUS1"; cov "367.705933"; FPKM "618.972839"; TPM "3102.666016";
chr12	StringTie	exon	131929282	131929380	1000	+	.	gene_id "STRG.4"; transcript_id "STRG.4.4"; exon_number "1"; reference_id "ENST00000538037.5"; ref_gene_id "ENSG00000177192.14"; ref_gene_name "PUS1"; cov "103.768829";
chr12	StringTie	exon	131929735	131929796	1000	+	.	gene_id "STRG.4"; transcript_id "STRG.4.4"; exon_number "2"; reference_id "ENST00000538037.5"; ref_gene_id "ENSG00000177192.14"; ref_gene_name "PUS1"; cov "313.002747";
chr12	StringTie	exon	131929907	131930135	1000	+	.	gene_id "STRG.4"; transcript_id "STRG.4.4"; exon_number "3"; reference_id "ENST00000538037.5"; ref_gene_id "ENSG00000177192.14"; ref_gene_name "PUS1"; cov "404.184357";
chr12	StringTie	exon	131932175	131932312	1000	+	.	gene_id "STRG.4"; transcript_id "STRG.4.4"; exon_number "4"; reference_id "ENST00000538037.5"; ref_gene_id "ENSG00000177192.14"; ref_gene_name "PUS1"; cov "472.658112";
chr12	StringTie	exon	131939173	131939275	1000	+	.	gene_id "STRG.4"; transcript_id "STRG.4.4"; exon_number "5"; reference_id "ENST00000538037.5"; ref_gene_id "ENSG00000177192.14"; ref_gene_name "PUS1"; cov "432.603241";
chr12	StringTie	transcript	131929304	131932680	1000	+	.	gene_id "STRG.4"; transcript_id "STRG.4.5"; reference_id "ENST00000544662.1"; ref_gene_id "ENSG00000177192.14"; ref_gene_name "PUS1"; cov "1126.412476"; FPKM "1896.131226"; TPM "9504.556641";
chr12	StringTie	exon	131929304	131929404	1000	+	.	gene_id "STRG.4"; transcript_id "STRG.4.5"; exon_number "1"; reference_id "ENST00000544662.1"; ref_gene_id "ENSG00000177192.14"; ref_gene_name "PUS1"; cov "2.688002";
chr12	StringTie	exon	131929907	131930135	1000	+	.	gene_id "STRG.4"; transcript_id "STRG.4.5"; exon_number "2"; reference_id "ENST00000544662.1"; ref_gene_id "ENSG00000177192.14"; ref_gene_name "PUS1"; cov "8.886930";
chr12	StringTie	exon	131932175	131932680	1000	+	.	gene_id "STRG.4"; transcript_id "STRG.4.5"; exon_number "3"; reference_id "ENST00000544662.1"; ref_gene_id "ENSG00000177192.14"; ref_gene_name "PUS1"; cov "1856.470703";
chr12	StringTie	transcript	131929307	131932833	1000	+	.	gene_id "STRG.4"; transcript_id "STRG.4.6"; reference_id "ENST00000456665.6"; ref_gene_id "ENSG00000177192.14"; ref_gene_name "PUS1"; cov "1953.258179"; FPKM "3287.990723"; TPM "16481.400391";
chr12	StringTie	exon	131929307	131929404	1000	+	.	gene_id "STRG.4"; transcript_id "STRG.4.6"; exon_number "1"; reference_id "ENST00000456665.6"; ref_gene_id "ENSG00000177192.14"; ref_gene_name "PUS1"; cov "202.592514";
chr12	StringTie	exon	131929907	131930135	1000	+	.	gene_id "STRG.4"; transcript_id "STRG.4.6"; exon_number "2"; reference_id "ENST00000456665.6"; ref_gene_id "ENSG00000177192.14"; ref_gene_name "PUS1"; cov "663.997681";
chr12	StringTie	exon	131932175	131932312	1000	+	.	gene_id "STRG.4"; transcript_id "STRG.4.6"; exon_number "3"; reference_id "ENST00000456665.6"; ref_gene_id "ENSG00000177192.14"; ref_gene_name "PUS1"; cov "776.487000";
chr12	StringTie	exon	131932722	131932833	1000	+	.	gene_id "STRG.4"; transcript_id "STRG.4.6"; exon_number "4"; reference_id "ENST00000456665.6"; ref_gene_id "ENSG00000177192.14"; ref_gene_name "PUS1"; cov "7571.118652";
chr12	StringTie	transcript	131929589	131932833	1000	+	.	gene_id "STRG.4"; transcript_id "STRG.4.7"; reference_id "ENST00000544213.5"; ref_gene_id "ENSG00000177192.14"; ref_gene_name "PUS1"; cov "438.360535"; FPKM "737.908325"; TPM "3698.842773";
chr12	StringTie	exon	131929589	131929796	1000	+	.	gene_id "STRG.4"; transcript_id "STRG.4.7"; exon_number "1"; reference_id "ENST00000544213.5"; ref_gene_id "ENSG00000177192.14"; ref_gene_name "PUS1"; cov "66.386703";
chr12	StringTie	exon	131929907	131930135	1000	+	.	gene_id "STRG.4"; transcript_id "STRG.4.7"; exon_number "2"; reference_id "ENST00000544213.5"; ref_gene_id "ENSG00000177192.14"; ref_gene_name "PUS1"; cov "172.327240";
chr12	StringTie	exon	131932175	131932312	1000	+	.	gene_id "STRG.4"; transcript_id "STRG.4.7"; exon_number "3"; reference_id "ENST00000544213.5"; ref_gene_id "ENSG00000177192.14"; ref_gene_name "PUS1"; cov "201.521576";
chr12	StringTie	exon	131932722	131932833	1000	+	.	gene_id "STRG.4"; transcript_id "STRG.4.7"; exon_number "4"; reference_id "ENST00000544213.5"; ref_gene_id "ENSG00000177192.14"; ref_gene_name "PUS1"; cov "1964.931519";
chr12	StringTie	transcript	131929755	131943857	1000	+	.	gene_id "STRG.4"; transcript_id "STRG.4.8"; reference_id "ENST00000535067.5"; ref_gene_id "ENSG00000177192.14"; ref_gene_name "PUS1"; cov "37.523735"; FPKM "63.165073"; TPM "316.621552";
chr12	StringTie	exon	131929755	131930135	1000	+	.	gene_id "STRG.4"; transcript_id "STRG.4.8"; exon_number "1"; reference_id "ENST00000535067.5"; ref_gene_id "ENSG00000177192.14"; ref_gene_name "PUS1"; cov "75.970001";
chr12	StringTie	exon	131932175	131932312	1000	+	.	gene_id "STRG.4"; transcript_id "STRG.4.8"; exon_number "2"; reference_id "ENST00000535067.5"; ref_gene_id "ENSG00000177192.14"; ref_gene_name "PUS1"; cov "10.543117";
chr12	StringTie	exon	131943539	131943857	1000	+	.	gene_id "STRG.4"; transcript_id "STRG.4.8"; exon_number "3"; reference_id "ENST00000535067.5"; ref_gene_id "ENSG00000177192.14"; ref_gene_name "PUS1"; cov "3.277017";
chr12	StringTie	transcript	131929765	131941422	1000	+	.	gene_id "STRG.4"; transcript_id "STRG.4.9"; reference_id "ENST00000537484.1"; ref_gene_id "ENSG00000177192.14"; ref_gene_name "PUS1"; cov "432.875000"; FPKM "728.674316"; TPM "3652.556396";
chr12	StringTie	exon	131929765	131929796	1000	+	.	gene_id "STRG.4"; transcript_id "STRG.4.9"; exon_number "1"; reference_id "ENST00000537484.1"; ref_gene_id "ENSG00000177192.14"; ref_gene_name "PUS1"; cov "127.098045";
chr12	StringTie	exon	131929907	131930135	1000	+	.	gene_id "STRG.4"; transcript_id "STRG.4.9"; exon_number "2"; reference_id "ENST00000537484.1"; ref_gene_id "ENSG00000177192.14"; ref_gene_name "PUS1"; cov "147.042633";
chr12	StringTie	exon	131932175	131932312	1000	+	.	gene_id "STRG.4"; transcript_id "STRG.4.9"; exon_number "3"; reference_id "ENST00000537484.1"; ref_gene_id "ENSG00000177192.14"; ref_gene_name "PUS1"; cov "171.953415";
chr12	StringTie	exon	131932722	131932833	1000	+	.	gene_id "STRG.4"; transcript_id "STRG.4.9"; exon_number "4"; reference_id "ENST00000537484.1"; ref_gene_id "ENSG00000177192.14"; ref_gene_name "PUS1"; cov "1676.627930";
chr12	StringTie	exon	131941292	131941422	1000	+	.	gene_id "STRG.4"; transcript_id "STRG.4.9"; exon_number "5"; reference_id "ENST00000537484.1"; ref_gene_id "ENSG00000177192.14"; ref_gene_name "PUS1"; cov "218.732346";
chr12	StringTie	transcript	131934642	131934928	1000	+	.	gene_id "STRG.5"; transcript_id "STRG.5.1"; reference_id "ENST00000621809.1"; ref_gene_id "ENSG00000273568.1"; ref_gene_name "AC131009.3"; cov "7427.000000"; FPKM "12502.139648"; TPM "62668.289062";
chr12	StringTie	exon	131934642	131934928	1000	+	.	gene_id "STRG.5"; transcript_id "STRG.5.1"; exon_number "1"; reference_id "ENST00000621809.1"; ref_gene_id "ENSG00000273568.1"; ref_gene_name "AC131009.3"; cov "7427.000000";
chr12	StringTie	transcript	131940927	131943859	1000	+	.	gene_id "STRG.4"; transcript_id "STRG.4.10"; reference_id "ENST00000543754.1"; ref_gene_id "ENSG00000177192.14"; ref_gene_name "PUS1"; cov "956.938477"; FPKM "1610.849487"; TPM "8074.552246";
chr12	StringTie	exon	131940927	131941983	1000	+	.	gene_id "STRG.4"; transcript_id "STRG.4.10"; exon_number "1"; reference_id "ENST00000543754.1"; ref_gene_id "ENSG00000177192.14"; ref_gene_name "PUS1"; cov "1244.444824";
chr12	StringTie	exon	131943539	131943859	1000	+	.	gene_id "STRG.4"; transcript_id "STRG.4.10"; exon_number "2"; reference_id "ENST00000543754.1"; ref_gene_id "ENSG00000177192.14"; ref_gene_name "PUS1"; cov "10.227993";
```

```bash
# Reformat the gtf with the script
python Reformat_gtf.py -i .test_data/stringtie.gtf -o .test_data/stringtie_reformatted.gtf -fa /project2/yangili1/bjf79/ReferenceGenomes/Human_UCSC.hg38_GencodeBasic46/Reference.fa -gene_name_attribute_name gene_id -transcript_name_attribute_name transcript_id -infer_transcript_type_approach C -infer_gene_type_approach B -translation_approach C

## and check output
cat .test_data/stringtie_reformatted.gtf
```

```

# stringtie -o scratch/test.merge.gtf --fr -G /project2/yangili1/bjf79/ChromatinSplicingQTLs/code/ReferenceGenome/Annotations/gencode.v34.chromasomal.annotation.gtf --ref /project2/yangili1/bjf79/ChromatinSplicingQTLs/code/ReferenceGenome/Fasta/GRCh38.primary_assembly.genome.fa scratch/test.merge.bam
# StringTie version 2.2.1
#! args: Namespace(gtf_in='.test_data/stringtie.gtf', gtf_out='.test_data/stringtie_reformatted.gtf', input_type='gtf', fasta_in='/project2/yangili1/bjf79/ReferenceGenomes/Human_UCSC.hg38_GencodeBasic46/Reference.fa', bed12_out=None, n_lines=None, infer_transcript_type_approach='C', infer_gene_type_approach='B', transcript_type_attribute_name='transcript_type', gene_type_attribute_name='gene_type', transcript_name_attribute_name='transcript_id', gene_name_attribute_name='gene_id', extra_attributes='transcript_support_level,tag,ccds_id', NMDetectiveB_coding_threshold=5, translation_approach='C', verbose=False)
chr12	input_gtf	gene	131894622	131923150	.	+	.	gene_name "ENSG00000177169.10"; gene_type "protein_coding";
chr12	FirstORF_NoStopRequired	transcript	131894622	131923150	0	+	.	transcript_name "STRG.1.1"; gene_name "ENSG00000177169.10"; transcript_type "protein_coding"; tag "NMDFinderB:Start_proximal";transcript_name "STRG.1.1"; gene_name "ENSG00000177169.10"; transcript_type "protein_coding"; tag "NMDFinderB:Start_proximal"; tag "FiveUTR_nEx":"1"; tag "FiveUTR_len":"228"; tag "ThreeUTR_nEx":"28"; tag "ThreeUTR_len":"4998"; tag "ThreeUTR_nEx_AfterFirst50":"28"; tag "CDSLen":"96"; tag "Introns":"chr12:131895112-131895600:+,chr12:131895693-131895782:+,chr12:131895824-131906891:+,chr12:131906924-131907494:+,chr12:131907531-131908643:+,chr12:131908817-131908897:+,chr12:131908971-131909135:+,chr12:131909237-131909774:+,chr12:131909833-131909918:+,chr12:131910001-131910253:+,chr12:131910304-131910711:+,chr12:131910800-131911941:+,chr12:131912089-131913197:+,chr12:131913258-131913746:+,chr12:131913836-131914351:+,chr12:131914477-131915082:+,chr12:131915231-131915334:+,chr12:131915421-131915890:+,chr12:131916159-131916397:+,chr12:131916591-131916952:+,chr12:131917062-131917410:+,chr12:131917554-131918496:+,chr12:131918681-131919211:+,chr12:131919384-131919471:+,chr12:131919590-131919978:+,chr12:131920136-131921099:+,chr12:131921235-131921305:+"; gene_type "protein_coding";
chr12	FirstORF_NoStopRequired	exon	131894622	131895112	0	+	.	transcript_name "STRG.1.1"; gene_name "ENSG00000177169.10"; transcript_type "protein_coding"; tag "NMDFinderB:Start_proximal"; exon_id "STRG.1.1_Exon001"; gene_type "protein_coding";
chr12	FirstORF_NoStopRequired	UTR	131894622	131894849	0	+	.	transcript_name "STRG.1.1"; gene_name "ENSG00000177169.10"; transcript_type "protein_coding"; tag "NMDFinderB:Start_proximal"; gene_type "protein_coding";
chr12	FirstORF_NoStopRequired	CDS	131894850	131894945	0	+	0	transcript_name "STRG.1.1"; gene_name "ENSG00000177169.10"; transcript_type "protein_coding"; tag "NMDFinderB:Start_proximal";transcript_name "STRG.1.1"; gene_name "ENSG00000177169.10"; transcript_type "protein_coding"; tag "NMDFinderB:Start_proximal"; gene_type "protein_coding";
chr12	FirstORF_NoStopRequired	start_codon	131894850	131894852	0	+	0	transcript_name "STRG.1.1"; gene_name "ENSG00000177169.10"; transcript_type "protein_coding"; tag "NMDFinderB:Start_proximal"; gene_type "protein_coding";
chr12	FirstORF_NoStopRequired	stop_codon	131894943	131894945	0	+	0	transcript_name "STRG.1.1"; gene_name "ENSG00000177169.10"; transcript_type "protein_coding"; tag "NMDFinderB:Start_proximal"; gene_type "protein_coding";
chr12	FirstORF_NoStopRequired	UTR	131894946	131895112	0	+	.	transcript_name "STRG.1.1"; gene_name "ENSG00000177169.10"; transcript_type "protein_coding"; tag "NMDFinderB:Start_proximal"; gene_type "protein_coding";
chr12	FirstORF_NoStopRequired	exon	131895601	131895693	0	+	.	transcript_name "STRG.1.1"; gene_name "ENSG00000177169.10"; transcript_type "protein_coding"; tag "NMDFinderB:Start_proximal"; exon_id "STRG.1.1_Exon002"; gene_type "protein_coding";
chr12	FirstORF_NoStopRequired	UTR	131895601	131895693	0	+	.	transcript_name "STRG.1.1"; gene_name "ENSG00000177169.10"; transcript_type "protein_coding"; tag "NMDFinderB:Start_proximal"; gene_type "protein_coding";
chr12	FirstORF_NoStopRequired	exon	131895783	131895824	0	+	.	transcript_name "STRG.1.1"; gene_name "ENSG00000177169.10"; transcript_type "protein_coding"; tag "NMDFinderB:Start_proximal"; exon_id "STRG.1.1_Exon003"; gene_type "protein_coding";
chr12	FirstORF_NoStopRequired	UTR	131895783	131895824	0	+	.	transcript_name "STRG.1.1"; gene_name "ENSG00000177169.10"; transcript_type "protein_coding"; tag "NMDFinderB:Start_proximal"; gene_type "protein_coding";
chr12	FirstORF_NoStopRequired	exon	131906892	131906924	0	+	.	transcript_name "STRG.1.1"; gene_name "ENSG00000177169.10"; transcript_type "protein_coding"; tag "NMDFinderB:Start_proximal"; exon_id "STRG.1.1_Exon004"; gene_type "protein_coding";
chr12	FirstORF_NoStopRequired	UTR	131906892	131906924	0	+	.	transcript_name "STRG.1.1"; gene_name "ENSG00000177169.10"; transcript_type "protein_coding"; tag "NMDFinderB:Start_proximal"; gene_type "protein_coding";
chr12	FirstORF_NoStopRequired	exon	131907495	131907531	0	+	.	transcript_name "STRG.1.1"; gene_name "ENSG00000177169.10"; transcript_type "protein_coding"; tag "NMDFinderB:Start_proximal"; exon_id "STRG.1.1_Exon005"; gene_type "protein_coding";
chr12	FirstORF_NoStopRequired	UTR	131907495	131907531	0	+	.	transcript_name "STRG.1.1"; gene_name "ENSG00000177169.10"; transcript_type "protein_coding"; tag "NMDFinderB:Start_proximal"; gene_type "protein_coding";
chr12	FirstORF_NoStopRequired	exon	131908644	131908817	0	+	.	transcript_name "STRG.1.1"; gene_name "ENSG00000177169.10"; transcript_type "protein_coding"; tag "NMDFinderB:Start_proximal"; exon_id "STRG.1.1_Exon006"; gene_type "protein_coding";
chr12	FirstORF_NoStopRequired	UTR	131908644	131908817	0	+	.	transcript_name "STRG.1.1"; gene_name "ENSG00000177169.10"; transcript_type "protein_coding"; tag "NMDFinderB:Start_proximal"; gene_type "protein_coding";
chr12	FirstORF_NoStopRequired	exon	131908898	131908971	0	+	.	transcript_name "STRG.1.1"; gene_name "ENSG00000177169.10"; transcript_type "protein_coding"; tag "NMDFinderB:Start_proximal"; exon_id "STRG.1.1_Exon007"; gene_type "protein_coding";
chr12	FirstORF_NoStopRequired	UTR	131908898	131908971	0	+	.	transcript_name "STRG.1.1"; gene_name "ENSG00000177169.10"; transcript_type "protein_coding"; tag "NMDFinderB:Start_proximal"; gene_type "protein_coding";
chr12	FirstORF_NoStopRequired	exon	131909136	131909237	0	+	.	transcript_name "STRG.1.1"; gene_name "ENSG00000177169.10"; transcript_type "protein_coding"; tag "NMDFinderB:Start_proximal"; exon_id "STRG.1.1_Exon008"; gene_type "protein_coding";
chr12	FirstORF_NoStopRequired	UTR	131909136	131909237	0	+	.	transcript_name "STRG.1.1"; gene_name "ENSG00000177169.10"; transcript_type "protein_coding"; tag "NMDFinderB:Start_proximal"; gene_type "protein_coding";
chr12	FirstORF_NoStopRequired	exon	131909775	131909833	0	+	.	transcript_name "STRG.1.1"; gene_name "ENSG00000177169.10"; transcript_type "protein_coding"; tag "NMDFinderB:Start_proximal"; exon_id "STRG.1.1_Exon009"; gene_type "protein_coding";
chr12	FirstORF_NoStopRequired	UTR	131909775	131909833	0	+	.	transcript_name "STRG.1.1"; gene_name "ENSG00000177169.10"; transcript_type "protein_coding"; tag "NMDFinderB:Start_proximal"; gene_type "protein_coding";
chr12	FirstORF_NoStopRequired	exon	131909919	131910001	0	+	.	transcript_name "STRG.1.1"; gene_name "ENSG00000177169.10"; transcript_type "protein_coding"; tag "NMDFinderB:Start_proximal"; exon_id "STRG.1.1_Exon010"; gene_type "protein_coding";
chr12	FirstORF_NoStopRequired	UTR	131909919	131910001	0	+	.	transcript_name "STRG.1.1"; gene_name "ENSG00000177169.10"; transcript_type "protein_coding"; tag "NMDFinderB:Start_proximal"; gene_type "protein_coding";
chr12	FirstORF_NoStopRequired	exon	131910254	131910304	0	+	.	transcript_name "STRG.1.1"; gene_name "ENSG00000177169.10"; transcript_type "protein_coding"; tag "NMDFinderB:Start_proximal"; exon_id "STRG.1.1_Exon011"; gene_type "protein_coding";
chr12	FirstORF_NoStopRequired	UTR	131910254	131910304	0	+	.	transcript_name "STRG.1.1"; gene_name "ENSG00000177169.10"; transcript_type "protein_coding"; tag "NMDFinderB:Start_proximal"; gene_type "protein_coding";
chr12	FirstORF_NoStopRequired	exon	131910712	131910800	0	+	.	transcript_name "STRG.1.1"; gene_name "ENSG00000177169.10"; transcript_type "protein_coding"; tag "NMDFinderB:Start_proximal"; exon_id "STRG.1.1_Exon012"; gene_type "protein_coding";
chr12	FirstORF_NoStopRequired	UTR	131910712	131910800	0	+	.	transcript_name "STRG.1.1"; gene_name "ENSG00000177169.10"; transcript_type "protein_coding"; tag "NMDFinderB:Start_proximal"; gene_type "protein_coding";
chr12	FirstORF_NoStopRequired	exon	131911942	131912089	0	+	.	transcript_name "STRG.1.1"; gene_name "ENSG00000177169.10"; transcript_type "protein_coding"; tag "NMDFinderB:Start_proximal"; exon_id "STRG.1.1_Exon013"; gene_type "protein_coding";
chr12	FirstORF_NoStopRequired	UTR	131911942	131912089	0	+	.	transcript_name "STRG.1.1"; gene_name "ENSG00000177169.10"; transcript_type "protein_coding"; tag "NMDFinderB:Start_proximal"; gene_type "protein_coding";
chr12	FirstORF_NoStopRequired	exon	131913198	131913258	0	+	.	transcript_name "STRG.1.1"; gene_name "ENSG00000177169.10"; transcript_type "protein_coding"; tag "NMDFinderB:Start_proximal"; exon_id "STRG.1.1_Exon014"; gene_type "protein_coding";
chr12	FirstORF_NoStopRequired	UTR	131913198	131913258	0	+	.	transcript_name "STRG.1.1"; gene_name "ENSG00000177169.10"; transcript_type "protein_coding"; tag "NMDFinderB:Start_proximal"; gene_type "protein_coding";
chr12	FirstORF_NoStopRequired	exon	131913747	131913836	0	+	.	transcript_name "STRG.1.1"; gene_name "ENSG00000177169.10"; transcript_type "protein_coding"; tag "NMDFinderB:Start_proximal"; exon_id "STRG.1.1_Exon015"; gene_type "protein_coding";
chr12	FirstORF_NoStopRequired	UTR	131913747	131913836	0	+	.	transcript_name "STRG.1.1"; gene_name "ENSG00000177169.10"; transcript_type "protein_coding"; tag "NMDFinderB:Start_proximal"; gene_type "protein_coding";
chr12	FirstORF_NoStopRequired	exon	131914352	131914477	0	+	.	transcript_name "STRG.1.1"; gene_name "ENSG00000177169.10"; transcript_type "protein_coding"; tag "NMDFinderB:Start_proximal"; exon_id "STRG.1.1_Exon016"; gene_type "protein_coding";
chr12	FirstORF_NoStopRequired	UTR	131914352	131914477	0	+	.	transcript_name "STRG.1.1"; gene_name "ENSG00000177169.10"; transcript_type "protein_coding"; tag "NMDFinderB:Start_proximal"; gene_type "protein_coding";
chr12	FirstORF_NoStopRequired	exon	131915083	131915231	0	+	.	transcript_name "STRG.1.1"; gene_name "ENSG00000177169.10"; transcript_type "protein_coding"; tag "NMDFinderB:Start_proximal"; exon_id "STRG.1.1_Exon017"; gene_type "protein_coding";
chr12	FirstORF_NoStopRequired	UTR	131915083	131915231	0	+	.	transcript_name "STRG.1.1"; gene_name "ENSG00000177169.10"; transcript_type "protein_coding"; tag "NMDFinderB:Start_proximal"; gene_type "protein_coding";
chr12	FirstORF_NoStopRequired	exon	131915335	131915421	0	+	.	transcript_name "STRG.1.1"; gene_name "ENSG00000177169.10"; transcript_type "protein_coding"; tag "NMDFinderB:Start_proximal"; exon_id "STRG.1.1_Exon018"; gene_type "protein_coding";
chr12	FirstORF_NoStopRequired	UTR	131915335	131915421	0	+	.	transcript_name "STRG.1.1"; gene_name "ENSG00000177169.10"; transcript_type "protein_coding"; tag "NMDFinderB:Start_proximal"; gene_type "protein_coding";
chr12	FirstORF_NoStopRequired	exon	131915891	131916159	0	+	.	transcript_name "STRG.1.1"; gene_name "ENSG00000177169.10"; transcript_type "protein_coding"; tag "NMDFinderB:Start_proximal"; exon_id "STRG.1.1_Exon019"; gene_type "protein_coding";
chr12	FirstORF_NoStopRequired	UTR	131915891	131916159	0	+	.	transcript_name "STRG.1.1"; gene_name "ENSG00000177169.10"; transcript_type "protein_coding"; tag "NMDFinderB:Start_proximal"; gene_type "protein_coding";
chr12	FirstORF_NoStopRequired	exon	131916398	131916591	0	+	.	transcript_name "STRG.1.1"; gene_name "ENSG00000177169.10"; transcript_type "protein_coding"; tag "NMDFinderB:Start_proximal"; exon_id "STRG.1.1_Exon020"; gene_type "protein_coding";
chr12	FirstORF_NoStopRequired	UTR	131916398	131916591	0	+	.	transcript_name "STRG.1.1"; gene_name "ENSG00000177169.10"; transcript_type "protein_coding"; tag "NMDFinderB:Start_proximal"; gene_type "protein_coding";
chr12	FirstORF_NoStopRequired	exon	131916953	131917062	0	+	.	transcript_name "STRG.1.1"; gene_name "ENSG00000177169.10"; transcript_type "protein_coding"; tag "NMDFinderB:Start_proximal"; exon_id "STRG.1.1_Exon021"; gene_type "protein_coding";
chr12	FirstORF_NoStopRequired	UTR	131916953	131917062	0	+	.	transcript_name "STRG.1.1"; gene_name "ENSG00000177169.10"; transcript_type "protein_coding"; tag "NMDFinderB:Start_proximal"; gene_type "protein_coding";
chr12	FirstORF_NoStopRequired	exon	131917411	131917554	0	+	.	transcript_name "STRG.1.1"; gene_name "ENSG00000177169.10"; transcript_type "protein_coding"; tag "NMDFinderB:Start_proximal"; exon_id "STRG.1.1_Exon022"; gene_type "protein_coding";
chr12	FirstORF_NoStopRequired	UTR	131917411	131917554	0	+	.	transcript_name "STRG.1.1"; gene_name "ENSG00000177169.10"; transcript_type "protein_coding"; tag "NMDFinderB:Start_proximal"; gene_type "protein_coding";
chr12	FirstORF_NoStopRequired	exon	131918497	131918681	0	+	.	transcript_name "STRG.1.1"; gene_name "ENSG00000177169.10"; transcript_type "protein_coding"; tag "NMDFinderB:Start_proximal"; exon_id "STRG.1.1_Exon023"; gene_type "protein_coding";
chr12	FirstORF_NoStopRequired	UTR	131918497	131918681	0	+	.	transcript_name "STRG.1.1"; gene_name "ENSG00000177169.10"; transcript_type "protein_coding"; tag "NMDFinderB:Start_proximal"; gene_type "protein_coding";
chr12	FirstORF_NoStopRequired	exon	131919212	131919384	0	+	.	transcript_name "STRG.1.1"; gene_name "ENSG00000177169.10"; transcript_type "protein_coding"; tag "NMDFinderB:Start_proximal"; exon_id "STRG.1.1_Exon024"; gene_type "protein_coding";
chr12	FirstORF_NoStopRequired	UTR	131919212	131919384	0	+	.	transcript_name "STRG.1.1"; gene_name "ENSG00000177169.10"; transcript_type "protein_coding"; tag "NMDFinderB:Start_proximal"; gene_type "protein_coding";
chr12	FirstORF_NoStopRequired	exon	131919472	131919590	0	+	.	transcript_name "STRG.1.1"; gene_name "ENSG00000177169.10"; transcript_type "protein_coding"; tag "NMDFinderB:Start_proximal"; exon_id "STRG.1.1_Exon025"; gene_type "protein_coding";
chr12	FirstORF_NoStopRequired	UTR	131919472	131919590	0	+	.	transcript_name "STRG.1.1"; gene_name "ENSG00000177169.10"; transcript_type "protein_coding"; tag "NMDFinderB:Start_proximal"; gene_type "protein_coding";
chr12	FirstORF_NoStopRequired	exon	131919979	131920136	0	+	.	transcript_name "STRG.1.1"; gene_name "ENSG00000177169.10"; transcript_type "protein_coding"; tag "NMDFinderB:Start_proximal"; exon_id "STRG.1.1_Exon026"; gene_type "protein_coding";
chr12	FirstORF_NoStopRequired	UTR	131919979	131920136	0	+	.	transcript_name "STRG.1.1"; gene_name "ENSG00000177169.10"; transcript_type "protein_coding"; tag "NMDFinderB:Start_proximal"; gene_type "protein_coding";
chr12	FirstORF_NoStopRequired	exon	131921100	131921235	0	+	.	transcript_name "STRG.1.1"; gene_name "ENSG00000177169.10"; transcript_type "protein_coding"; tag "NMDFinderB:Start_proximal"; exon_id "STRG.1.1_Exon027"; gene_type "protein_coding";
chr12	FirstORF_NoStopRequired	UTR	131921100	131921235	0	+	.	transcript_name "STRG.1.1"; gene_name "ENSG00000177169.10"; transcript_type "protein_coding"; tag "NMDFinderB:Start_proximal"; gene_type "protein_coding";
chr12	FirstORF_NoStopRequired	exon	131921306	131923150	0	+	.	transcript_name "STRG.1.1"; gene_name "ENSG00000177169.10"; transcript_type "protein_coding"; tag "NMDFinderB:Start_proximal"; exon_id "STRG.1.1_Exon028"; gene_type "protein_coding";
chr12	FirstORF_NoStopRequired	UTR	131921306	131923150	0	+	.	transcript_name "STRG.1.1"; gene_name "ENSG00000177169.10"; transcript_type "protein_coding"; tag "NMDFinderB:Start_proximal"; gene_type "protein_coding";
chr12	FirstORF_NoStopRequired	transcript	131906490	131909996	0	+	.	transcript_name "STRG.1.2"; gene_name "ENSG00000177169.10"; transcript_type "protein_coding"; tag "NMDFinderB:Long_exon";transcript_name "STRG.1.2"; gene_name "ENSG00000177169.10"; transcript_type "protein_coding"; tag "NMDFinderB:Long_exon"; tag "FiveUTR_nEx":"1"; tag "FiveUTR_len":"138"; tag "ThreeUTR_nEx":"7"; tag "ThreeUTR_len":"665"; tag "ThreeUTR_nEx_AfterFirst50":"7"; tag "CDSLen":"156"; tag "Introns":"chr12:131906924-131907494:+,chr12:131907531-131908643:+,chr12:131908817-131908897:+,chr12:131908971-131909135:+,chr12:131909237-131909774:+,chr12:131909833-131909918:+"; gene_type "protein_coding";
chr12	FirstORF_NoStopRequired	exon	131906490	131906924	0	+	.	transcript_name "STRG.1.2"; gene_name "ENSG00000177169.10"; transcript_type "protein_coding"; tag "NMDFinderB:Long_exon"; exon_id "STRG.1.2_Exon001"; gene_type "protein_coding";
chr12	FirstORF_NoStopRequired	UTR	131906490	131906627	0	+	.	transcript_name "STRG.1.2"; gene_name "ENSG00000177169.10"; transcript_type "protein_coding"; tag "NMDFinderB:Long_exon"; gene_type "protein_coding";
chr12	FirstORF_NoStopRequired	CDS	131906628	131906783	0	+	0	transcript_name "STRG.1.2"; gene_name "ENSG00000177169.10"; transcript_type "protein_coding"; tag "NMDFinderB:Long_exon";transcript_name "STRG.1.2"; gene_name "ENSG00000177169.10"; transcript_type "protein_coding"; tag "NMDFinderB:Long_exon"; gene_type "protein_coding";
chr12	FirstORF_NoStopRequired	start_codon	131906628	131906630	0	+	0	transcript_name "STRG.1.2"; gene_name "ENSG00000177169.10"; transcript_type "protein_coding"; tag "NMDFinderB:Long_exon"; gene_type "protein_coding";
chr12	FirstORF_NoStopRequired	stop_codon	131906781	131906783	0	+	0	transcript_name "STRG.1.2"; gene_name "ENSG00000177169.10"; transcript_type "protein_coding"; tag "NMDFinderB:Long_exon"; gene_type "protein_coding";
chr12	FirstORF_NoStopRequired	UTR	131906784	131906924	0	+	.	transcript_name "STRG.1.2"; gene_name "ENSG00000177169.10"; transcript_type "protein_coding"; tag "NMDFinderB:Long_exon"; gene_type "protein_coding";
chr12	FirstORF_NoStopRequired	exon	131907495	131907531	0	+	.	transcript_name "STRG.1.2"; gene_name "ENSG00000177169.10"; transcript_type "protein_coding"; tag "NMDFinderB:Long_exon"; exon_id "STRG.1.2_Exon002"; gene_type "protein_coding";
chr12	FirstORF_NoStopRequired	UTR	131907495	131907531	0	+	.	transcript_name "STRG.1.2"; gene_name "ENSG00000177169.10"; transcript_type "protein_coding"; tag "NMDFinderB:Long_exon"; gene_type "protein_coding";
chr12	FirstORF_NoStopRequired	exon	131908644	131908817	0	+	.	transcript_name "STRG.1.2"; gene_name "ENSG00000177169.10"; transcript_type "protein_coding"; tag "NMDFinderB:Long_exon"; exon_id "STRG.1.2_Exon003"; gene_type "protein_coding";
chr12	FirstORF_NoStopRequired	UTR	131908644	131908817	0	+	.	transcript_name "STRG.1.2"; gene_name "ENSG00000177169.10"; transcript_type "protein_coding"; tag "NMDFinderB:Long_exon"; gene_type "protein_coding";
chr12	FirstORF_NoStopRequired	exon	131908898	131908971	0	+	.	transcript_name "STRG.1.2"; gene_name "ENSG00000177169.10"; transcript_type "protein_coding"; tag "NMDFinderB:Long_exon"; exon_id "STRG.1.2_Exon004"; gene_type "protein_coding";
chr12	FirstORF_NoStopRequired	UTR	131908898	131908971	0	+	.	transcript_name "STRG.1.2"; gene_name "ENSG00000177169.10"; transcript_type "protein_coding"; tag "NMDFinderB:Long_exon"; gene_type "protein_coding";
chr12	FirstORF_NoStopRequired	exon	131909136	131909237	0	+	.	transcript_name "STRG.1.2"; gene_name "ENSG00000177169.10"; transcript_type "protein_coding"; tag "NMDFinderB:Long_exon"; exon_id "STRG.1.2_Exon005"; gene_type "protein_coding";
chr12	FirstORF_NoStopRequired	UTR	131909136	131909237	0	+	.	transcript_name "STRG.1.2"; gene_name "ENSG00000177169.10"; transcript_type "protein_coding"; tag "NMDFinderB:Long_exon"; gene_type "protein_coding";
chr12	FirstORF_NoStopRequired	exon	131909775	131909833	0	+	.	transcript_name "STRG.1.2"; gene_name "ENSG00000177169.10"; transcript_type "protein_coding"; tag "NMDFinderB:Long_exon"; exon_id "STRG.1.2_Exon006"; gene_type "protein_coding";
chr12	FirstORF_NoStopRequired	UTR	131909775	131909833	0	+	.	transcript_name "STRG.1.2"; gene_name "ENSG00000177169.10"; transcript_type "protein_coding"; tag "NMDFinderB:Long_exon"; gene_type "protein_coding";
chr12	FirstORF_NoStopRequired	exon	131909919	131909996	0	+	.	transcript_name "STRG.1.2"; gene_name "ENSG00000177169.10"; transcript_type "protein_coding"; tag "NMDFinderB:Long_exon"; exon_id "STRG.1.2_Exon007"; gene_type "protein_coding";
chr12	FirstORF_NoStopRequired	UTR	131909919	131909996	0	+	.	transcript_name "STRG.1.2"; gene_name "ENSG00000177169.10"; transcript_type "protein_coding"; tag "NMDFinderB:Long_exon"; gene_type "protein_coding";
chr12	FirstORF_NoStopRequired	transcript	131908619	131909169	0	+	.	transcript_name "STRG.1.3"; gene_name "ENSG00000177169.10"; transcript_type "noncoding"; tag "NMDFinderB:No_stop";transcript_name "STRG.1.3"; gene_name "ENSG00000177169.10"; transcript_type "noncoding"; tag "NMDFinderB:No_stop"; tag "FiveUTR_nEx":"0"; tag "FiveUTR_len":"0"; tag "ThreeUTR_nEx":"0"; tag "ThreeUTR_len":"0"; tag "ThreeUTR_nEx_AfterFirst50":"0"; tag "CDSLen":"0"; tag "Introns":"chr12:131908817-131908897:+,chr12:131908971-131909135:+"; gene_type "protein_coding";
chr12	FirstORF_NoStopRequired	exon	131908619	131908817	0	+	.	transcript_name "STRG.1.3"; gene_name "ENSG00000177169.10"; transcript_type "noncoding"; tag "NMDFinderB:No_stop"; exon_id "STRG.1.3_Exon001"; gene_type "protein_coding";
chr12	FirstORF_NoStopRequired	exon	131908898	131908971	0	+	.	transcript_name "STRG.1.3"; gene_name "ENSG00000177169.10"; transcript_type "noncoding"; tag "NMDFinderB:No_stop"; exon_id "STRG.1.3_Exon002"; gene_type "protein_coding";
chr12	FirstORF_NoStopRequired	exon	131909136	131909169	0	+	.	transcript_name "STRG.1.3"; gene_name "ENSG00000177169.10"; transcript_type "noncoding"; tag "NMDFinderB:No_stop"; exon_id "STRG.1.3_Exon003"; gene_type "protein_coding";
chr12	FirstORF_NoStopRequired	transcript	131915943	131919385	0	+	.	transcript_name "STRG.1.4"; gene_name "ENSG00000177169.10"; transcript_type "protein_coding"; tag "NMDFinderB:Start_proximal";transcript_name "STRG.1.4"; gene_name "ENSG00000177169.10"; transcript_type "protein_coding"; tag "NMDFinderB:Start_proximal"; tag "FiveUTR_nEx":"1"; tag "FiveUTR_len":"220"; tag "ThreeUTR_nEx":"5"; tag "ThreeUTR_len":"940"; tag "ThreeUTR_nEx_AfterFirst50":"5"; tag "CDSLen":"102"; tag "Introns":"chr12:131916591-131916952:+,chr12:131917062-131917410:+,chr12:131917554-131918496:+,chr12:131918681-131919211:+"; gene_type "protein_coding";
chr12	FirstORF_NoStopRequired	exon	131915943	131916591	0	+	.	transcript_name "STRG.1.4"; gene_name "ENSG00000177169.10"; transcript_type "protein_coding"; tag "NMDFinderB:Start_proximal"; exon_id "STRG.1.4_Exon001"; gene_type "protein_coding";
chr12	FirstORF_NoStopRequired	UTR	131915943	131916162	0	+	.	transcript_name "STRG.1.4"; gene_name "ENSG00000177169.10"; transcript_type "protein_coding"; tag "NMDFinderB:Start_proximal"; gene_type "protein_coding";
chr12	FirstORF_NoStopRequired	CDS	131916163	131916264	0	+	0	transcript_name "STRG.1.4"; gene_name "ENSG00000177169.10"; transcript_type "protein_coding"; tag "NMDFinderB:Start_proximal";transcript_name "STRG.1.4"; gene_name "ENSG00000177169.10"; transcript_type "protein_coding"; tag "NMDFinderB:Start_proximal"; gene_type "protein_coding";
chr12	FirstORF_NoStopRequired	start_codon	131916163	131916165	0	+	0	transcript_name "STRG.1.4"; gene_name "ENSG00000177169.10"; transcript_type "protein_coding"; tag "NMDFinderB:Start_proximal"; gene_type "protein_coding";
chr12	FirstORF_NoStopRequired	stop_codon	131916262	131916264	0	+	0	transcript_name "STRG.1.4"; gene_name "ENSG00000177169.10"; transcript_type "protein_coding"; tag "NMDFinderB:Start_proximal"; gene_type "protein_coding";
chr12	FirstORF_NoStopRequired	UTR	131916265	131916591	0	+	.	transcript_name "STRG.1.4"; gene_name "ENSG00000177169.10"; transcript_type "protein_coding"; tag "NMDFinderB:Start_proximal"; gene_type "protein_coding";
chr12	FirstORF_NoStopRequired	exon	131916953	131917062	0	+	.	transcript_name "STRG.1.4"; gene_name "ENSG00000177169.10"; transcript_type "protein_coding"; tag "NMDFinderB:Start_proximal"; exon_id "STRG.1.4_Exon002"; gene_type "protein_coding";
chr12	FirstORF_NoStopRequired	UTR	131916953	131917062	0	+	.	transcript_name "STRG.1.4"; gene_name "ENSG00000177169.10"; transcript_type "protein_coding"; tag "NMDFinderB:Start_proximal"; gene_type "protein_coding";
chr12	FirstORF_NoStopRequired	exon	131917411	131917554	0	+	.	transcript_name "STRG.1.4"; gene_name "ENSG00000177169.10"; transcript_type "protein_coding"; tag "NMDFinderB:Start_proximal"; exon_id "STRG.1.4_Exon003"; gene_type "protein_coding";
chr12	FirstORF_NoStopRequired	UTR	131917411	131917554	0	+	.	transcript_name "STRG.1.4"; gene_name "ENSG00000177169.10"; transcript_type "protein_coding"; tag "NMDFinderB:Start_proximal"; gene_type "protein_coding";
chr12	FirstORF_NoStopRequired	exon	131918497	131918681	0	+	.	transcript_name "STRG.1.4"; gene_name "ENSG00000177169.10"; transcript_type "protein_coding"; tag "NMDFinderB:Start_proximal"; exon_id "STRG.1.4_Exon004"; gene_type "protein_coding";
chr12	FirstORF_NoStopRequired	UTR	131918497	131918681	0	+	.	transcript_name "STRG.1.4"; gene_name "ENSG00000177169.10"; transcript_type "protein_coding"; tag "NMDFinderB:Start_proximal"; gene_type "protein_coding";
chr12	FirstORF_NoStopRequired	exon	131919212	131919385	0	+	.	transcript_name "STRG.1.4"; gene_name "ENSG00000177169.10"; transcript_type "protein_coding"; tag "NMDFinderB:Start_proximal"; exon_id "STRG.1.4_Exon005"; gene_type "protein_coding";
chr12	FirstORF_NoStopRequired	UTR	131919212	131919385	0	+	.	transcript_name "STRG.1.4"; gene_name "ENSG00000177169.10"; transcript_type "protein_coding"; tag "NMDFinderB:Start_proximal"; gene_type "protein_coding";
chr12	FirstORF_NoStopRequired	transcript	131918302	131921550	0	+	.	transcript_name "STRG.1.5"; gene_name "ENSG00000177169.10"; transcript_type "protein_coding"; tag "NMDFinderB:Last_exon";transcript_name "STRG.1.5"; gene_name "ENSG00000177169.10"; transcript_type "protein_coding"; tag "NMDFinderB:Last_exon"; tag "FiveUTR_nEx":"1"; tag "FiveUTR_len":"134"; tag "ThreeUTR_nEx":"1"; tag "ThreeUTR_len":"189"; tag "ThreeUTR_nEx_AfterFirst50":"1"; tag "CDSLen":"888"; tag "Introns":"chr12:131918681-131919211:+,chr12:131919384-131919471:+,chr12:131919590-131919978:+,chr12:131920136-131921099:+,chr12:131921235-131921305:+"; gene_type "protein_coding";
chr12	FirstORF_NoStopRequired	exon	131918302	131918681	0	+	.	transcript_name "STRG.1.5"; gene_name "ENSG00000177169.10"; transcript_type "protein_coding"; tag "NMDFinderB:Last_exon"; exon_id "STRG.1.5_Exon001"; gene_type "protein_coding";
chr12	FirstORF_NoStopRequired	UTR	131918302	131918435	0	+	.	transcript_name "STRG.1.5"; gene_name "ENSG00000177169.10"; transcript_type "protein_coding"; tag "NMDFinderB:Last_exon"; gene_type "protein_coding";
chr12	FirstORF_NoStopRequired	CDS	131918436	131918681	0	+	0	transcript_name "STRG.1.5"; gene_name "ENSG00000177169.10"; transcript_type "protein_coding"; tag "NMDFinderB:Last_exon";transcript_name "STRG.1.5"; gene_name "ENSG00000177169.10"; transcript_type "protein_coding"; tag "NMDFinderB:Last_exon"; gene_type "protein_coding";
chr12	FirstORF_NoStopRequired	start_codon	131918436	131918438	0	+	0	transcript_name "STRG.1.5"; gene_name "ENSG00000177169.10"; transcript_type "protein_coding"; tag "NMDFinderB:Last_exon"; gene_type "protein_coding";
chr12	FirstORF_NoStopRequired	exon	131919212	131919384	0	+	.	transcript_name "STRG.1.5"; gene_name "ENSG00000177169.10"; transcript_type "protein_coding"; tag "NMDFinderB:Last_exon"; exon_id "STRG.1.5_Exon002"; gene_type "protein_coding";
chr12	FirstORF_NoStopRequired	CDS	131919212	131919384	0	+	0	transcript_name "STRG.1.5"; gene_name "ENSG00000177169.10"; transcript_type "protein_coding"; tag "NMDFinderB:Last_exon";transcript_name "STRG.1.5"; gene_name "ENSG00000177169.10"; transcript_type "protein_coding"; tag "NMDFinderB:Last_exon"; gene_type "protein_coding";
chr12	FirstORF_NoStopRequired	exon	131919472	131919590	0	+	.	transcript_name "STRG.1.5"; gene_name "ENSG00000177169.10"; transcript_type "protein_coding"; tag "NMDFinderB:Last_exon"; exon_id "STRG.1.5_Exon003"; gene_type "protein_coding";
chr12	FirstORF_NoStopRequired	CDS	131919472	131919590	0	+	1	transcript_name "STRG.1.5"; gene_name "ENSG00000177169.10"; transcript_type "protein_coding"; tag "NMDFinderB:Last_exon";transcript_name "STRG.1.5"; gene_name "ENSG00000177169.10"; transcript_type "protein_coding"; tag "NMDFinderB:Last_exon"; gene_type "protein_coding";
chr12	FirstORF_NoStopRequired	exon	131919979	131920136	0	+	.	transcript_name "STRG.1.5"; gene_name "ENSG00000177169.10"; transcript_type "protein_coding"; tag "NMDFinderB:Last_exon"; exon_id "STRG.1.5_Exon004"; gene_type "protein_coding";
chr12	FirstORF_NoStopRequired	CDS	131919979	131920136	0	+	2	transcript_name "STRG.1.5"; gene_name "ENSG00000177169.10"; transcript_type "protein_coding"; tag "NMDFinderB:Last_exon";transcript_name "STRG.1.5"; gene_name "ENSG00000177169.10"; transcript_type "protein_coding"; tag "NMDFinderB:Last_exon"; gene_type "protein_coding";
chr12	FirstORF_NoStopRequired	exon	131921100	131921235	0	+	.	transcript_name "STRG.1.5"; gene_name "ENSG00000177169.10"; transcript_type "protein_coding"; tag "NMDFinderB:Last_exon"; exon_id "STRG.1.5_Exon005"; gene_type "protein_coding";
chr12	FirstORF_NoStopRequired	CDS	131921100	131921235	0	+	0	transcript_name "STRG.1.5"; gene_name "ENSG00000177169.10"; transcript_type "protein_coding"; tag "NMDFinderB:Last_exon";transcript_name "STRG.1.5"; gene_name "ENSG00000177169.10"; transcript_type "protein_coding"; tag "NMDFinderB:Last_exon"; gene_type "protein_coding";
chr12	FirstORF_NoStopRequired	exon	131921306	131921550	0	+	.	transcript_name "STRG.1.5"; gene_name "ENSG00000177169.10"; transcript_type "protein_coding"; tag "NMDFinderB:Last_exon"; exon_id "STRG.1.5_Exon006"; gene_type "protein_coding";
chr12	FirstORF_NoStopRequired	CDS	131921306	131921361	0	+	2	transcript_name "STRG.1.5"; gene_name "ENSG00000177169.10"; transcript_type "protein_coding"; tag "NMDFinderB:Last_exon";transcript_name "STRG.1.5"; gene_name "ENSG00000177169.10"; transcript_type "protein_coding"; tag "NMDFinderB:Last_exon"; gene_type "protein_coding";
chr12	FirstORF_NoStopRequired	stop_codon	131921359	131921361	0	+	0	transcript_name "STRG.1.5"; gene_name "ENSG00000177169.10"; transcript_type "protein_coding"; tag "NMDFinderB:Last_exon"; gene_type "protein_coding";
chr12	FirstORF_NoStopRequired	UTR	131921362	131921550	0	+	.	transcript_name "STRG.1.5"; gene_name "ENSG00000177169.10"; transcript_type "protein_coding"; tag "NMDFinderB:Last_exon"; gene_type "protein_coding";
chr12	FirstORF_NoStopRequired	transcript	131919241	131921951	0	+	.	transcript_name "STRG.1.6"; gene_name "ENSG00000177169.10"; transcript_type "protein_coding"; tag "NMDFinderB:Long_exon";transcript_name "STRG.1.6"; gene_name "ENSG00000177169.10"; transcript_type "protein_coding"; tag "NMDFinderB:Long_exon"; tag "FiveUTR_nEx":"1"; tag "FiveUTR_len":"538"; tag "ThreeUTR_nEx":"3"; tag "ThreeUTR_len":"921"; tag "ThreeUTR_nEx_AfterFirst50":"3"; tag "CDSLen":"219"; tag "Introns":"chr12:131920136-131921099:+,chr12:131921235-131921305:+"; gene_type "protein_coding";
chr12	FirstORF_NoStopRequired	exon	131919241	131920136	0	+	.	transcript_name "STRG.1.6"; gene_name "ENSG00000177169.10"; transcript_type "protein_coding"; tag "NMDFinderB:Long_exon"; exon_id "STRG.1.6_Exon001"; gene_type "protein_coding";
chr12	FirstORF_NoStopRequired	UTR	131919241	131919778	0	+	.	transcript_name "STRG.1.6"; gene_name "ENSG00000177169.10"; transcript_type "protein_coding"; tag "NMDFinderB:Long_exon"; gene_type "protein_coding";
chr12	FirstORF_NoStopRequired	CDS	131919779	131919997	0	+	0	transcript_name "STRG.1.6"; gene_name "ENSG00000177169.10"; transcript_type "protein_coding"; tag "NMDFinderB:Long_exon";transcript_name "STRG.1.6"; gene_name "ENSG00000177169.10"; transcript_type "protein_coding"; tag "NMDFinderB:Long_exon"; gene_type "protein_coding";
chr12	FirstORF_NoStopRequired	start_codon	131919779	131919781	0	+	0	transcript_name "STRG.1.6"; gene_name "ENSG00000177169.10"; transcript_type "protein_coding"; tag "NMDFinderB:Long_exon"; gene_type "protein_coding";
chr12	FirstORF_NoStopRequired	stop_codon	131919995	131919997	0	+	0	transcript_name "STRG.1.6"; gene_name "ENSG00000177169.10"; transcript_type "protein_coding"; tag "NMDFinderB:Long_exon"; gene_type "protein_coding";
chr12	FirstORF_NoStopRequired	UTR	131919998	131920136	0	+	.	transcript_name "STRG.1.6"; gene_name "ENSG00000177169.10"; transcript_type "protein_coding"; tag "NMDFinderB:Long_exon"; gene_type "protein_coding";
chr12	FirstORF_NoStopRequired	exon	131921100	131921235	0	+	.	transcript_name "STRG.1.6"; gene_name "ENSG00000177169.10"; transcript_type "protein_coding"; tag "NMDFinderB:Long_exon"; exon_id "STRG.1.6_Exon002"; gene_type "protein_coding";
chr12	FirstORF_NoStopRequired	UTR	131921100	131921235	0	+	.	transcript_name "STRG.1.6"; gene_name "ENSG00000177169.10"; transcript_type "protein_coding"; tag "NMDFinderB:Long_exon"; gene_type "protein_coding";
chr12	FirstORF_NoStopRequired	exon	131921306	131921951	0	+	.	transcript_name "STRG.1.6"; gene_name "ENSG00000177169.10"; transcript_type "protein_coding"; tag "NMDFinderB:Long_exon"; exon_id "STRG.1.6_Exon003"; gene_type "protein_coding";
chr12	FirstORF_NoStopRequired	UTR	131921306	131921951	0	+	.	transcript_name "STRG.1.6"; gene_name "ENSG00000177169.10"; transcript_type "protein_coding"; tag "NMDFinderB:Long_exon"; gene_type "protein_coding";
chr12	FirstORF_NoStopRequired	transcript	131920067	131922281	0	+	.	transcript_name "STRG.1.7"; gene_name "ENSG00000177169.10"; transcript_type "protein_coding"; tag "NMDFinderB:Long_exon";transcript_name "STRG.1.7"; gene_name "ENSG00000177169.10"; transcript_type "protein_coding"; tag "NMDFinderB:Long_exon"; tag "FiveUTR_nEx":"1"; tag "FiveUTR_len":"199"; tag "ThreeUTR_nEx":"2"; tag "ThreeUTR_len":"1814"; tag "ThreeUTR_nEx_AfterFirst50":"2"; tag "CDSLen":"132"; tag "Introns":"chr12:131921235-131921305:+"; gene_type "protein_coding";
chr12	FirstORF_NoStopRequired	exon	131920067	131921235	0	+	.	transcript_name "STRG.1.7"; gene_name "ENSG00000177169.10"; transcript_type "protein_coding"; tag "NMDFinderB:Long_exon"; exon_id "STRG.1.7_Exon001"; gene_type "protein_coding";
chr12	FirstORF_NoStopRequired	UTR	131920067	131920265	0	+	.	transcript_name "STRG.1.7"; gene_name "ENSG00000177169.10"; transcript_type "protein_coding"; tag "NMDFinderB:Long_exon"; gene_type "protein_coding";
chr12	FirstORF_NoStopRequired	CDS	131920266	131920397	0	+	0	transcript_name "STRG.1.7"; gene_name "ENSG00000177169.10"; transcript_type "protein_coding"; tag "NMDFinderB:Long_exon";transcript_name "STRG.1.7"; gene_name "ENSG00000177169.10"; transcript_type "protein_coding"; tag "NMDFinderB:Long_exon"; gene_type "protein_coding";
chr12	FirstORF_NoStopRequired	start_codon	131920266	131920268	0	+	0	transcript_name "STRG.1.7"; gene_name "ENSG00000177169.10"; transcript_type "protein_coding"; tag "NMDFinderB:Long_exon"; gene_type "protein_coding";
chr12	FirstORF_NoStopRequired	stop_codon	131920395	131920397	0	+	0	transcript_name "STRG.1.7"; gene_name "ENSG00000177169.10"; transcript_type "protein_coding"; tag "NMDFinderB:Long_exon"; gene_type "protein_coding";
chr12	FirstORF_NoStopRequired	UTR	131920398	131921235	0	+	.	transcript_name "STRG.1.7"; gene_name "ENSG00000177169.10"; transcript_type "protein_coding"; tag "NMDFinderB:Long_exon"; gene_type "protein_coding";
chr12	FirstORF_NoStopRequired	exon	131921306	131922281	0	+	.	transcript_name "STRG.1.7"; gene_name "ENSG00000177169.10"; transcript_type "protein_coding"; tag "NMDFinderB:Long_exon"; exon_id "STRG.1.7_Exon002"; gene_type "protein_coding";
chr12	FirstORF_NoStopRequired	UTR	131921306	131922281	0	+	.	transcript_name "STRG.1.7"; gene_name "ENSG00000177169.10"; transcript_type "protein_coding"; tag "NMDFinderB:Long_exon"; gene_type "protein_coding";
chr12	input_gtf	gene	131896500	131898239	.	+	.	gene_name "ENSG00000279283.1"; gene_type "protein_coding";
chr12	FirstORF_NoStopRequired	transcript	131896500	131898239	0	+	.	transcript_name "STRG.2.1"; gene_name "ENSG00000279283.1"; transcript_type "protein_coding"; tag "NMDFinderB:Last_exon";transcript_name "STRG.2.1"; gene_name "ENSG00000279283.1"; transcript_type "protein_coding"; tag "NMDFinderB:Last_exon"; tag "FiveUTR_nEx":"1"; tag "FiveUTR_len":"294"; tag "ThreeUTR_nEx":"1"; tag "ThreeUTR_len":"1344"; tag "ThreeUTR_nEx_AfterFirst50":"1"; tag "CDSLen":"102"; tag "Introns":""; gene_type "protein_coding";
chr12	FirstORF_NoStopRequired	exon	131896500	131898239	0	+	.	transcript_name "STRG.2.1"; gene_name "ENSG00000279283.1"; transcript_type "protein_coding"; tag "NMDFinderB:Last_exon"; exon_id "STRG.2.1_Exon001"; gene_type "protein_coding";
chr12	FirstORF_NoStopRequired	UTR	131896500	131896793	0	+	.	transcript_name "STRG.2.1"; gene_name "ENSG00000279283.1"; transcript_type "protein_coding"; tag "NMDFinderB:Last_exon"; gene_type "protein_coding";
chr12	FirstORF_NoStopRequired	CDS	131896794	131896895	0	+	0	transcript_name "STRG.2.1"; gene_name "ENSG00000279283.1"; transcript_type "protein_coding"; tag "NMDFinderB:Last_exon";transcript_name "STRG.2.1"; gene_name "ENSG00000279283.1"; transcript_type "protein_coding"; tag "NMDFinderB:Last_exon"; gene_type "protein_coding";
chr12	FirstORF_NoStopRequired	start_codon	131896794	131896796	0	+	0	transcript_name "STRG.2.1"; gene_name "ENSG00000279283.1"; transcript_type "protein_coding"; tag "NMDFinderB:Last_exon"; gene_type "protein_coding";
chr12	FirstORF_NoStopRequired	stop_codon	131896893	131896895	0	+	0	transcript_name "STRG.2.1"; gene_name "ENSG00000279283.1"; transcript_type "protein_coding"; tag "NMDFinderB:Last_exon"; gene_type "protein_coding";
chr12	FirstORF_NoStopRequired	UTR	131896896	131898239	0	+	.	transcript_name "STRG.2.1"; gene_name "ENSG00000279283.1"; transcript_type "protein_coding"; tag "NMDFinderB:Last_exon"; gene_type "protein_coding";
chr12	input_gtf	gene	131924736	131929351	.	-	.	gene_name "ENSG00000255992.1"; gene_type "noncoding";
chr12	FirstORF_NoStopRequired	transcript	131924736	131929351	0	-	.	transcript_name "STRG.3.1"; gene_name "ENSG00000255992.1"; transcript_type "noncoding"; tag "NMDFinderB:No_stop";transcript_name "STRG.3.1"; gene_name "ENSG00000255992.1"; transcript_type "noncoding"; tag "NMDFinderB:No_stop"; tag "FiveUTR_nEx":"0"; tag "FiveUTR_len":"0"; tag "ThreeUTR_nEx":"0"; tag "ThreeUTR_len":"0"; tag "ThreeUTR_nEx_AfterFirst50":"0"; tag "CDSLen":"0"; tag "Introns":"chr12:131924766-131927962:-,chr12:131928190-131929184:-"; gene_type "noncoding";
chr12	FirstORF_NoStopRequired	exon	131924736	131924766	0	-	.	transcript_name "STRG.3.1"; gene_name "ENSG00000255992.1"; transcript_type "noncoding"; tag "NMDFinderB:No_stop"; exon_id "STRG.3.1_Exon001"; gene_type "noncoding";
chr12	FirstORF_NoStopRequired	exon	131927963	131928190	0	-	.	transcript_name "STRG.3.1"; gene_name "ENSG00000255992.1"; transcript_type "noncoding"; tag "NMDFinderB:No_stop"; exon_id "STRG.3.1_Exon002"; gene_type "noncoding";
chr12	FirstORF_NoStopRequired	exon	131929185	131929351	0	-	.	transcript_name "STRG.3.1"; gene_name "ENSG00000255992.1"; transcript_type "noncoding"; tag "NMDFinderB:No_stop"; exon_id "STRG.3.1_Exon003"; gene_type "noncoding";
chr12	input_gtf	gene	131929200	131945896	.	+	.	gene_name "ENSG00000177192.14"; gene_type "protein_coding";
chr12	FirstORF_NoStopRequired	transcript	131929200	131943859	0	+	.	transcript_name "STRG.4.1"; gene_name "ENSG00000177192.14"; transcript_type "noncoding"; tag "NMDFinderB:Trigger_NMD";transcript_name "STRG.4.1"; gene_name "ENSG00000177192.14"; transcript_type "noncoding"; tag "NMDFinderB:Trigger_NMD"; tag "FiveUTR_nEx":"1"; tag "FiveUTR_len":"93"; tag "ThreeUTR_nEx":"4"; tag "ThreeUTR_len":"1154"; tag "ThreeUTR_nEx_AfterFirst50":"3"; tag "CDSLen":"417"; tag "Introns":"chr12:131929380-131929906:+,chr12:131930135-131932174:+,chr12:131932312-131939172:+,chr12:131939275-131941291:+,chr12:131941983-131943538:+"; gene_type "protein_coding";
chr12	FirstORF_NoStopRequired	exon	131929200	131929380	0	+	.	transcript_name "STRG.4.1"; gene_name "ENSG00000177192.14"; transcript_type "noncoding"; tag "NMDFinderB:Trigger_NMD"; exon_id "STRG.4.1_Exon001"; gene_type "protein_coding";
chr12	FirstORF_NoStopRequired	UTR	131929200	131929292	0	+	.	transcript_name "STRG.4.1"; gene_name "ENSG00000177192.14"; transcript_type "noncoding"; tag "NMDFinderB:Trigger_NMD"; gene_type "protein_coding";
chr12	FirstORF_NoStopRequired	CDS	131929293	131929380	0	+	0	transcript_name "STRG.4.1"; gene_name "ENSG00000177192.14"; transcript_type "noncoding"; tag "NMDFinderB:Trigger_NMD";transcript_name "STRG.4.1"; gene_name "ENSG00000177192.14"; transcript_type "noncoding"; tag "NMDFinderB:Trigger_NMD"; gene_type "protein_coding";
chr12	FirstORF_NoStopRequired	start_codon	131929293	131929295	0	+	0	transcript_name "STRG.4.1"; gene_name "ENSG00000177192.14"; transcript_type "noncoding"; tag "NMDFinderB:Trigger_NMD"; gene_type "protein_coding";
chr12	FirstORF_NoStopRequired	exon	131929907	131930135	0	+	.	transcript_name "STRG.4.1"; gene_name "ENSG00000177192.14"; transcript_type "noncoding"; tag "NMDFinderB:Trigger_NMD"; exon_id "STRG.4.1_Exon002"; gene_type "protein_coding";
chr12	FirstORF_NoStopRequired	CDS	131929907	131930135	0	+	2	transcript_name "STRG.4.1"; gene_name "ENSG00000177192.14"; transcript_type "noncoding"; tag "NMDFinderB:Trigger_NMD";transcript_name "STRG.4.1"; gene_name "ENSG00000177192.14"; transcript_type "noncoding"; tag "NMDFinderB:Trigger_NMD"; gene_type "protein_coding";
chr12	FirstORF_NoStopRequired	exon	131932175	131932312	0	+	.	transcript_name "STRG.4.1"; gene_name "ENSG00000177192.14"; transcript_type "noncoding"; tag "NMDFinderB:Trigger_NMD"; exon_id "STRG.4.1_Exon003"; gene_type "protein_coding";
chr12	FirstORF_NoStopRequired	CDS	131932175	131932274	0	+	1	transcript_name "STRG.4.1"; gene_name "ENSG00000177192.14"; transcript_type "noncoding"; tag "NMDFinderB:Trigger_NMD";transcript_name "STRG.4.1"; gene_name "ENSG00000177192.14"; transcript_type "noncoding"; tag "NMDFinderB:Trigger_NMD"; gene_type "protein_coding";
chr12	FirstORF_NoStopRequired	stop_codon	131932272	131932274	0	+	0	transcript_name "STRG.4.1"; gene_name "ENSG00000177192.14"; transcript_type "noncoding"; tag "NMDFinderB:Trigger_NMD"; gene_type "protein_coding";
chr12	FirstORF_NoStopRequired	UTR	131932275	131932312	0	+	.	transcript_name "STRG.4.1"; gene_name "ENSG00000177192.14"; transcript_type "noncoding"; tag "NMDFinderB:Trigger_NMD"; gene_type "protein_coding";
chr12	FirstORF_NoStopRequired	exon	131939173	131939275	0	+	.	transcript_name "STRG.4.1"; gene_name "ENSG00000177192.14"; transcript_type "noncoding"; tag "NMDFinderB:Trigger_NMD"; exon_id "STRG.4.1_Exon004"; gene_type "protein_coding";
chr12	FirstORF_NoStopRequired	UTR	131939173	131939275	0	+	.	transcript_name "STRG.4.1"; gene_name "ENSG00000177192.14"; transcript_type "noncoding"; tag "NMDFinderB:Trigger_NMD"; gene_type "protein_coding";
chr12	FirstORF_NoStopRequired	exon	131941292	131941983	0	+	.	transcript_name "STRG.4.1"; gene_name "ENSG00000177192.14"; transcript_type "noncoding"; tag "NMDFinderB:Trigger_NMD"; exon_id "STRG.4.1_Exon005"; gene_type "protein_coding";
chr12	FirstORF_NoStopRequired	UTR	131941292	131941983	0	+	.	transcript_name "STRG.4.1"; gene_name "ENSG00000177192.14"; transcript_type "noncoding"; tag "NMDFinderB:Trigger_NMD"; gene_type "protein_coding";
chr12	FirstORF_NoStopRequired	exon	131943539	131943859	0	+	.	transcript_name "STRG.4.1"; gene_name "ENSG00000177192.14"; transcript_type "noncoding"; tag "NMDFinderB:Trigger_NMD"; exon_id "STRG.4.1_Exon006"; gene_type "protein_coding";
chr12	FirstORF_NoStopRequired	UTR	131943539	131943859	0	+	.	transcript_name "STRG.4.1"; gene_name "ENSG00000177192.14"; transcript_type "noncoding"; tag "NMDFinderB:Trigger_NMD"; gene_type "protein_coding";
chr12	FirstORF_NoStopRequired	transcript	131929276	131945896	0	+	.	transcript_name "STRG.4.2"; gene_name "ENSG00000177192.14"; transcript_type "protein_coding"; tag "NMDFinderB:Start_proximal";transcript_name "STRG.4.2"; gene_name "ENSG00000177192.14"; transcript_type "protein_coding"; tag "NMDFinderB:Start_proximal"; tag "FiveUTR_nEx":"1"; tag "FiveUTR_len":"17"; tag "ThreeUTR_nEx":"6"; tag "ThreeUTR_len":"3907"; tag "ThreeUTR_nEx_AfterFirst50":"6"; tag "CDSLen":"117"; tag "Introns":"chr12:131929796-131929906:+,chr12:131930135-131932174:+,chr12:131932312-131939172:+,chr12:131939275-131941291:+,chr12:131941983-131943538:+"; gene_type "protein_coding";
chr12	FirstORF_NoStopRequired	exon	131929276	131929796	0	+	.	transcript_name "STRG.4.2"; gene_name "ENSG00000177192.14"; transcript_type "protein_coding"; tag "NMDFinderB:Start_proximal"; exon_id "STRG.4.2_Exon001"; gene_type "protein_coding";
chr12	FirstORF_NoStopRequired	UTR	131929276	131929292	0	+	.	transcript_name "STRG.4.2"; gene_name "ENSG00000177192.14"; transcript_type "protein_coding"; tag "NMDFinderB:Start_proximal"; gene_type "protein_coding";
chr12	FirstORF_NoStopRequired	CDS	131929293	131929409	0	+	0	transcript_name "STRG.4.2"; gene_name "ENSG00000177192.14"; transcript_type "protein_coding"; tag "NMDFinderB:Start_proximal";transcript_name "STRG.4.2"; gene_name "ENSG00000177192.14"; transcript_type "protein_coding"; tag "NMDFinderB:Start_proximal"; gene_type "protein_coding";
chr12	FirstORF_NoStopRequired	start_codon	131929293	131929295	0	+	0	transcript_name "STRG.4.2"; gene_name "ENSG00000177192.14"; transcript_type "protein_coding"; tag "NMDFinderB:Start_proximal"; gene_type "protein_coding";
chr12	FirstORF_NoStopRequired	stop_codon	131929407	131929409	0	+	0	transcript_name "STRG.4.2"; gene_name "ENSG00000177192.14"; transcript_type "protein_coding"; tag "NMDFinderB:Start_proximal"; gene_type "protein_coding";
chr12	FirstORF_NoStopRequired	UTR	131929410	131929796	0	+	.	transcript_name "STRG.4.2"; gene_name "ENSG00000177192.14"; transcript_type "protein_coding"; tag "NMDFinderB:Start_proximal"; gene_type "protein_coding";
chr12	FirstORF_NoStopRequired	exon	131929907	131930135	0	+	.	transcript_name "STRG.4.2"; gene_name "ENSG00000177192.14"; transcript_type "protein_coding"; tag "NMDFinderB:Start_proximal"; exon_id "STRG.4.2_Exon002"; gene_type "protein_coding";
chr12	FirstORF_NoStopRequired	UTR	131929907	131930135	0	+	.	transcript_name "STRG.4.2"; gene_name "ENSG00000177192.14"; transcript_type "protein_coding"; tag "NMDFinderB:Start_proximal"; gene_type "protein_coding";
chr12	FirstORF_NoStopRequired	exon	131932175	131932312	0	+	.	transcript_name "STRG.4.2"; gene_name "ENSG00000177192.14"; transcript_type "protein_coding"; tag "NMDFinderB:Start_proximal"; exon_id "STRG.4.2_Exon003"; gene_type "protein_coding";
chr12	FirstORF_NoStopRequired	UTR	131932175	131932312	0	+	.	transcript_name "STRG.4.2"; gene_name "ENSG00000177192.14"; transcript_type "protein_coding"; tag "NMDFinderB:Start_proximal"; gene_type "protein_coding";
chr12	FirstORF_NoStopRequired	exon	131939173	131939275	0	+	.	transcript_name "STRG.4.2"; gene_name "ENSG00000177192.14"; transcript_type "protein_coding"; tag "NMDFinderB:Start_proximal"; exon_id "STRG.4.2_Exon004"; gene_type "protein_coding";
chr12	FirstORF_NoStopRequired	UTR	131939173	131939275	0	+	.	transcript_name "STRG.4.2"; gene_name "ENSG00000177192.14"; transcript_type "protein_coding"; tag "NMDFinderB:Start_proximal"; gene_type "protein_coding";
chr12	FirstORF_NoStopRequired	exon	131941292	131941983	0	+	.	transcript_name "STRG.4.2"; gene_name "ENSG00000177192.14"; transcript_type "protein_coding"; tag "NMDFinderB:Start_proximal"; exon_id "STRG.4.2_Exon005"; gene_type "protein_coding";
chr12	FirstORF_NoStopRequired	UTR	131941292	131941983	0	+	.	transcript_name "STRG.4.2"; gene_name "ENSG00000177192.14"; transcript_type "protein_coding"; tag "NMDFinderB:Start_proximal"; gene_type "protein_coding";
chr12	FirstORF_NoStopRequired	exon	131943539	131945896	0	+	.	transcript_name "STRG.4.2"; gene_name "ENSG00000177192.14"; transcript_type "protein_coding"; tag "NMDFinderB:Start_proximal"; exon_id "STRG.4.2_Exon006"; gene_type "protein_coding";
chr12	FirstORF_NoStopRequired	UTR	131943539	131945896	0	+	.	transcript_name "STRG.4.2"; gene_name "ENSG00000177192.14"; transcript_type "protein_coding"; tag "NMDFinderB:Start_proximal"; gene_type "protein_coding";
chr12	FirstORF_NoStopRequired	transcript	131929276	131943859	0	+	.	transcript_name "STRG.4.3"; gene_name "ENSG00000177192.14"; transcript_type "protein_coding"; tag "NMDFinderB:Start_proximal";transcript_name "STRG.4.3"; gene_name "ENSG00000177192.14"; transcript_type "protein_coding"; tag "NMDFinderB:Start_proximal"; tag "FiveUTR_nEx":"1"; tag "FiveUTR_len":"17"; tag "ThreeUTR_nEx":"5"; tag "ThreeUTR_len":"2017"; tag "ThreeUTR_nEx_AfterFirst50":"5"; tag "CDSLen":"117"; tag "Introns":"chr12:131929512-131931514:+,chr12:131932312-131939172:+,chr12:131939275-131941291:+,chr12:131941983-131943538:+"; gene_type "protein_coding";
chr12	FirstORF_NoStopRequired	exon	131929276	131929512	0	+	.	transcript_name "STRG.4.3"; gene_name "ENSG00000177192.14"; transcript_type "protein_coding"; tag "NMDFinderB:Start_proximal"; exon_id "STRG.4.3_Exon001"; gene_type "protein_coding";
chr12	FirstORF_NoStopRequired	UTR	131929276	131929292	0	+	.	transcript_name "STRG.4.3"; gene_name "ENSG00000177192.14"; transcript_type "protein_coding"; tag "NMDFinderB:Start_proximal"; gene_type "protein_coding";
chr12	FirstORF_NoStopRequired	CDS	131929293	131929409	0	+	0	transcript_name "STRG.4.3"; gene_name "ENSG00000177192.14"; transcript_type "protein_coding"; tag "NMDFinderB:Start_proximal";transcript_name "STRG.4.3"; gene_name "ENSG00000177192.14"; transcript_type "protein_coding"; tag "NMDFinderB:Start_proximal"; gene_type "protein_coding";
chr12	FirstORF_NoStopRequired	start_codon	131929293	131929295	0	+	0	transcript_name "STRG.4.3"; gene_name "ENSG00000177192.14"; transcript_type "protein_coding"; tag "NMDFinderB:Start_proximal"; gene_type "protein_coding";
chr12	FirstORF_NoStopRequired	stop_codon	131929407	131929409	0	+	0	transcript_name "STRG.4.3"; gene_name "ENSG00000177192.14"; transcript_type "protein_coding"; tag "NMDFinderB:Start_proximal"; gene_type "protein_coding";
chr12	FirstORF_NoStopRequired	UTR	131929410	131929512	0	+	.	transcript_name "STRG.4.3"; gene_name "ENSG00000177192.14"; transcript_type "protein_coding"; tag "NMDFinderB:Start_proximal"; gene_type "protein_coding";
chr12	FirstORF_NoStopRequired	exon	131931515	131932312	0	+	.	transcript_name "STRG.4.3"; gene_name "ENSG00000177192.14"; transcript_type "protein_coding"; tag "NMDFinderB:Start_proximal"; exon_id "STRG.4.3_Exon002"; gene_type "protein_coding";
chr12	FirstORF_NoStopRequired	UTR	131931515	131932312	0	+	.	transcript_name "STRG.4.3"; gene_name "ENSG00000177192.14"; transcript_type "protein_coding"; tag "NMDFinderB:Start_proximal"; gene_type "protein_coding";
chr12	FirstORF_NoStopRequired	exon	131939173	131939275	0	+	.	transcript_name "STRG.4.3"; gene_name "ENSG00000177192.14"; transcript_type "protein_coding"; tag "NMDFinderB:Start_proximal"; exon_id "STRG.4.3_Exon003"; gene_type "protein_coding";
chr12	FirstORF_NoStopRequired	UTR	131939173	131939275	0	+	.	transcript_name "STRG.4.3"; gene_name "ENSG00000177192.14"; transcript_type "protein_coding"; tag "NMDFinderB:Start_proximal"; gene_type "protein_coding";
chr12	FirstORF_NoStopRequired	exon	131941292	131941983	0	+	.	transcript_name "STRG.4.3"; gene_name "ENSG00000177192.14"; transcript_type "protein_coding"; tag "NMDFinderB:Start_proximal"; exon_id "STRG.4.3_Exon004"; gene_type "protein_coding";
chr12	FirstORF_NoStopRequired	UTR	131941292	131941983	0	+	.	transcript_name "STRG.4.3"; gene_name "ENSG00000177192.14"; transcript_type "protein_coding"; tag "NMDFinderB:Start_proximal"; gene_type "protein_coding";
chr12	FirstORF_NoStopRequired	exon	131943539	131943859	0	+	.	transcript_name "STRG.4.3"; gene_name "ENSG00000177192.14"; transcript_type "protein_coding"; tag "NMDFinderB:Start_proximal"; exon_id "STRG.4.3_Exon005"; gene_type "protein_coding";
chr12	FirstORF_NoStopRequired	UTR	131943539	131943859	0	+	.	transcript_name "STRG.4.3"; gene_name "ENSG00000177192.14"; transcript_type "protein_coding"; tag "NMDFinderB:Start_proximal"; gene_type "protein_coding";
chr12	FirstORF_NoStopRequired	transcript	131929282	131939275	0	+	.	transcript_name "STRG.4.4"; gene_name "ENSG00000177192.14"; transcript_type "noncoding"; tag "NMDFinderB:Trigger_NMD";transcript_name "STRG.4.4"; gene_name "ENSG00000177192.14"; transcript_type "noncoding"; tag "NMDFinderB:Trigger_NMD"; tag "FiveUTR_nEx":"1"; tag "FiveUTR_len":"11"; tag "ThreeUTR_nEx":"2"; tag "ThreeUTR_len":"206"; tag "ThreeUTR_nEx_AfterFirst50":"2"; tag "CDSLen":"414"; tag "Introns":"chr12:131929380-131929734:+,chr12:131929796-131929906:+,chr12:131930135-131932174:+,chr12:131932312-131939172:+"; gene_type "protein_coding";
chr12	FirstORF_NoStopRequired	exon	131929282	131929380	0	+	.	transcript_name "STRG.4.4"; gene_name "ENSG00000177192.14"; transcript_type "noncoding"; tag "NMDFinderB:Trigger_NMD"; exon_id "STRG.4.4_Exon001"; gene_type "protein_coding";
chr12	FirstORF_NoStopRequired	UTR	131929282	131929292	0	+	.	transcript_name "STRG.4.4"; gene_name "ENSG00000177192.14"; transcript_type "noncoding"; tag "NMDFinderB:Trigger_NMD"; gene_type "protein_coding";
chr12	FirstORF_NoStopRequired	CDS	131929293	131929380	0	+	0	transcript_name "STRG.4.4"; gene_name "ENSG00000177192.14"; transcript_type "noncoding"; tag "NMDFinderB:Trigger_NMD";transcript_name "STRG.4.4"; gene_name "ENSG00000177192.14"; transcript_type "noncoding"; tag "NMDFinderB:Trigger_NMD"; gene_type "protein_coding";
chr12	FirstORF_NoStopRequired	start_codon	131929293	131929295	0	+	0	transcript_name "STRG.4.4"; gene_name "ENSG00000177192.14"; transcript_type "noncoding"; tag "NMDFinderB:Trigger_NMD"; gene_type "protein_coding";
chr12	FirstORF_NoStopRequired	exon	131929735	131929796	0	+	.	transcript_name "STRG.4.4"; gene_name "ENSG00000177192.14"; transcript_type "noncoding"; tag "NMDFinderB:Trigger_NMD"; exon_id "STRG.4.4_Exon002"; gene_type "protein_coding";
chr12	FirstORF_NoStopRequired	CDS	131929735	131929796	0	+	2	transcript_name "STRG.4.4"; gene_name "ENSG00000177192.14"; transcript_type "noncoding"; tag "NMDFinderB:Trigger_NMD";transcript_name "STRG.4.4"; gene_name "ENSG00000177192.14"; transcript_type "noncoding"; tag "NMDFinderB:Trigger_NMD"; gene_type "protein_coding";
chr12	FirstORF_NoStopRequired	exon	131929907	131930135	0	+	.	transcript_name "STRG.4.4"; gene_name "ENSG00000177192.14"; transcript_type "noncoding"; tag "NMDFinderB:Trigger_NMD"; exon_id "STRG.4.4_Exon003"; gene_type "protein_coding";
chr12	FirstORF_NoStopRequired	CDS	131929907	131930135	0	+	0	transcript_name "STRG.4.4"; gene_name "ENSG00000177192.14"; transcript_type "noncoding"; tag "NMDFinderB:Trigger_NMD";transcript_name "STRG.4.4"; gene_name "ENSG00000177192.14"; transcript_type "noncoding"; tag "NMDFinderB:Trigger_NMD"; gene_type "protein_coding";
chr12	FirstORF_NoStopRequired	exon	131932175	131932312	0	+	.	transcript_name "STRG.4.4"; gene_name "ENSG00000177192.14"; transcript_type "noncoding"; tag "NMDFinderB:Trigger_NMD"; exon_id "STRG.4.4_Exon004"; gene_type "protein_coding";
chr12	FirstORF_NoStopRequired	CDS	131932175	131932209	0	+	2	transcript_name "STRG.4.4"; gene_name "ENSG00000177192.14"; transcript_type "noncoding"; tag "NMDFinderB:Trigger_NMD";transcript_name "STRG.4.4"; gene_name "ENSG00000177192.14"; transcript_type "noncoding"; tag "NMDFinderB:Trigger_NMD"; gene_type "protein_coding";
chr12	FirstORF_NoStopRequired	stop_codon	131932207	131932209	0	+	0	transcript_name "STRG.4.4"; gene_name "ENSG00000177192.14"; transcript_type "noncoding"; tag "NMDFinderB:Trigger_NMD"; gene_type "protein_coding";
chr12	FirstORF_NoStopRequired	UTR	131932210	131932312	0	+	.	transcript_name "STRG.4.4"; gene_name "ENSG00000177192.14"; transcript_type "noncoding"; tag "NMDFinderB:Trigger_NMD"; gene_type "protein_coding";
chr12	FirstORF_NoStopRequired	exon	131939173	131939275	0	+	.	transcript_name "STRG.4.4"; gene_name "ENSG00000177192.14"; transcript_type "noncoding"; tag "NMDFinderB:Trigger_NMD"; exon_id "STRG.4.4_Exon005"; gene_type "protein_coding";
chr12	FirstORF_NoStopRequired	UTR	131939173	131939275	0	+	.	transcript_name "STRG.4.4"; gene_name "ENSG00000177192.14"; transcript_type "noncoding"; tag "NMDFinderB:Trigger_NMD"; gene_type "protein_coding";
chr12	FirstORF_NoStopRequired	transcript	131929304	131932680	0	+	.	transcript_name "STRG.4.5"; gene_name "ENSG00000177192.14"; transcript_type "protein_coding"; tag "NMDFinderB:Last_exon";transcript_name "STRG.4.5"; gene_name "ENSG00000177192.14"; transcript_type "protein_coding"; tag "NMDFinderB:Last_exon"; tag "FiveUTR_nEx":"2"; tag "FiveUTR_len":"111"; tag "ThreeUTR_nEx":"1"; tag "ThreeUTR_len":"245"; tag "ThreeUTR_nEx_AfterFirst50":"1"; tag "CDSLen":"480"; tag "Introns":"chr12:131929404-131929906:+,chr12:131930135-131932174:+"; gene_type "protein_coding";
chr12	FirstORF_NoStopRequired	exon	131929304	131929404	0	+	.	transcript_name "STRG.4.5"; gene_name "ENSG00000177192.14"; transcript_type "protein_coding"; tag "NMDFinderB:Last_exon"; exon_id "STRG.4.5_Exon001"; gene_type "protein_coding";
chr12	FirstORF_NoStopRequired	UTR	131929304	131929404	0	+	.	transcript_name "STRG.4.5"; gene_name "ENSG00000177192.14"; transcript_type "protein_coding"; tag "NMDFinderB:Last_exon"; gene_type "protein_coding";
chr12	FirstORF_NoStopRequired	exon	131929907	131930135	0	+	.	transcript_name "STRG.4.5"; gene_name "ENSG00000177192.14"; transcript_type "protein_coding"; tag "NMDFinderB:Last_exon"; exon_id "STRG.4.5_Exon002"; gene_type "protein_coding";
chr12	FirstORF_NoStopRequired	UTR	131929907	131929916	0	+	.	transcript_name "STRG.4.5"; gene_name "ENSG00000177192.14"; transcript_type "protein_coding"; tag "NMDFinderB:Last_exon"; gene_type "protein_coding";
chr12	FirstORF_NoStopRequired	CDS	131929917	131930135	0	+	0	transcript_name "STRG.4.5"; gene_name "ENSG00000177192.14"; transcript_type "protein_coding"; tag "NMDFinderB:Last_exon";transcript_name "STRG.4.5"; gene_name "ENSG00000177192.14"; transcript_type "protein_coding"; tag "NMDFinderB:Last_exon"; gene_type "protein_coding";
chr12	FirstORF_NoStopRequired	start_codon	131929917	131929919	0	+	0	transcript_name "STRG.4.5"; gene_name "ENSG00000177192.14"; transcript_type "protein_coding"; tag "NMDFinderB:Last_exon"; gene_type "protein_coding";
chr12	FirstORF_NoStopRequired	exon	131932175	131932680	0	+	.	transcript_name "STRG.4.5"; gene_name "ENSG00000177192.14"; transcript_type "protein_coding"; tag "NMDFinderB:Last_exon"; exon_id "STRG.4.5_Exon003"; gene_type "protein_coding";
chr12	FirstORF_NoStopRequired	CDS	131932175	131932435	0	+	0	transcript_name "STRG.4.5"; gene_name "ENSG00000177192.14"; transcript_type "protein_coding"; tag "NMDFinderB:Last_exon";transcript_name "STRG.4.5"; gene_name "ENSG00000177192.14"; transcript_type "protein_coding"; tag "NMDFinderB:Last_exon"; gene_type "protein_coding";
chr12	FirstORF_NoStopRequired	stop_codon	131932433	131932435	0	+	0	transcript_name "STRG.4.5"; gene_name "ENSG00000177192.14"; transcript_type "protein_coding"; tag "NMDFinderB:Last_exon"; gene_type "protein_coding";
chr12	FirstORF_NoStopRequired	UTR	131932436	131932680	0	+	.	transcript_name "STRG.4.5"; gene_name "ENSG00000177192.14"; transcript_type "protein_coding"; tag "NMDFinderB:Last_exon"; gene_type "protein_coding";
chr12	FirstORF_NoStopRequired	transcript	131929307	131932833	0	+	.	transcript_name "STRG.4.6"; gene_name "ENSG00000177192.14"; transcript_type "noncoding"; tag "NMDFinderB:No_stop";transcript_name "STRG.4.6"; gene_name "ENSG00000177192.14"; transcript_type "noncoding"; tag "NMDFinderB:No_stop"; tag "FiveUTR_nEx":"0"; tag "FiveUTR_len":"0"; tag "ThreeUTR_nEx":"0"; tag "ThreeUTR_len":"0"; tag "ThreeUTR_nEx_AfterFirst50":"0"; tag "CDSLen":"0"; tag "Introns":"chr12:131929404-131929906:+,chr12:131930135-131932174:+,chr12:131932312-131932721:+"; gene_type "protein_coding";
chr12	FirstORF_NoStopRequired	exon	131929307	131929404	0	+	.	transcript_name "STRG.4.6"; gene_name "ENSG00000177192.14"; transcript_type "noncoding"; tag "NMDFinderB:No_stop"; exon_id "STRG.4.6_Exon001"; gene_type "protein_coding";
chr12	FirstORF_NoStopRequired	exon	131929907	131930135	0	+	.	transcript_name "STRG.4.6"; gene_name "ENSG00000177192.14"; transcript_type "noncoding"; tag "NMDFinderB:No_stop"; exon_id "STRG.4.6_Exon002"; gene_type "protein_coding";
chr12	FirstORF_NoStopRequired	exon	131932175	131932312	0	+	.	transcript_name "STRG.4.6"; gene_name "ENSG00000177192.14"; transcript_type "noncoding"; tag "NMDFinderB:No_stop"; exon_id "STRG.4.6_Exon003"; gene_type "protein_coding";
chr12	FirstORF_NoStopRequired	exon	131932722	131932833	0	+	.	transcript_name "STRG.4.6"; gene_name "ENSG00000177192.14"; transcript_type "noncoding"; tag "NMDFinderB:No_stop"; exon_id "STRG.4.6_Exon004"; gene_type "protein_coding";
chr12	FirstORF_NoStopRequired	transcript	131929589	131932833	0	+	.	transcript_name "STRG.4.7"; gene_name "ENSG00000177192.14"; transcript_type "noncoding"; tag "NMDFinderB:No_stop";transcript_name "STRG.4.7"; gene_name "ENSG00000177192.14"; transcript_type "noncoding"; tag "NMDFinderB:No_stop"; tag "FiveUTR_nEx":"0"; tag "FiveUTR_len":"0"; tag "ThreeUTR_nEx":"0"; tag "ThreeUTR_len":"0"; tag "ThreeUTR_nEx_AfterFirst50":"0"; tag "CDSLen":"0"; tag "Introns":"chr12:131929796-131929906:+,chr12:131930135-131932174:+,chr12:131932312-131932721:+"; gene_type "protein_coding";
chr12	FirstORF_NoStopRequired	exon	131929589	131929796	0	+	.	transcript_name "STRG.4.7"; gene_name "ENSG00000177192.14"; transcript_type "noncoding"; tag "NMDFinderB:No_stop"; exon_id "STRG.4.7_Exon001"; gene_type "protein_coding";
chr12	FirstORF_NoStopRequired	exon	131929907	131930135	0	+	.	transcript_name "STRG.4.7"; gene_name "ENSG00000177192.14"; transcript_type "noncoding"; tag "NMDFinderB:No_stop"; exon_id "STRG.4.7_Exon002"; gene_type "protein_coding";
chr12	FirstORF_NoStopRequired	exon	131932175	131932312	0	+	.	transcript_name "STRG.4.7"; gene_name "ENSG00000177192.14"; transcript_type "noncoding"; tag "NMDFinderB:No_stop"; exon_id "STRG.4.7_Exon003"; gene_type "protein_coding";
chr12	FirstORF_NoStopRequired	exon	131932722	131932833	0	+	.	transcript_name "STRG.4.7"; gene_name "ENSG00000177192.14"; transcript_type "noncoding"; tag "NMDFinderB:No_stop"; exon_id "STRG.4.7_Exon004"; gene_type "protein_coding";
chr12	FirstORF_NoStopRequired	transcript	131929755	131943857	0	+	.	transcript_name "STRG.4.8"; gene_name "ENSG00000177192.14"; transcript_type "protein_coding"; tag "NMDFinderB:Last_exon";transcript_name "STRG.4.8"; gene_name "ENSG00000177192.14"; transcript_type "protein_coding"; tag "NMDFinderB:Last_exon"; tag "FiveUTR_nEx":"1"; tag "FiveUTR_len":"45"; tag "ThreeUTR_nEx":"1"; tag "ThreeUTR_len":"271"; tag "ThreeUTR_nEx_AfterFirst50":"1"; tag "CDSLen":"522"; tag "Introns":"chr12:131930135-131932174:+,chr12:131932312-131943538:+"; gene_type "protein_coding";
chr12	FirstORF_NoStopRequired	exon	131929755	131930135	0	+	.	transcript_name "STRG.4.8"; gene_name "ENSG00000177192.14"; transcript_type "protein_coding"; tag "NMDFinderB:Last_exon"; exon_id "STRG.4.8_Exon001"; gene_type "protein_coding";
chr12	FirstORF_NoStopRequired	UTR	131929755	131929799	0	+	.	transcript_name "STRG.4.8"; gene_name "ENSG00000177192.14"; transcript_type "protein_coding"; tag "NMDFinderB:Last_exon"; gene_type "protein_coding";
chr12	FirstORF_NoStopRequired	CDS	131929800	131930135	0	+	0	transcript_name "STRG.4.8"; gene_name "ENSG00000177192.14"; transcript_type "protein_coding"; tag "NMDFinderB:Last_exon";transcript_name "STRG.4.8"; gene_name "ENSG00000177192.14"; transcript_type "protein_coding"; tag "NMDFinderB:Last_exon"; gene_type "protein_coding";
chr12	FirstORF_NoStopRequired	start_codon	131929800	131929802	0	+	0	transcript_name "STRG.4.8"; gene_name "ENSG00000177192.14"; transcript_type "protein_coding"; tag "NMDFinderB:Last_exon"; gene_type "protein_coding";
chr12	FirstORF_NoStopRequired	exon	131932175	131932312	0	+	.	transcript_name "STRG.4.8"; gene_name "ENSG00000177192.14"; transcript_type "protein_coding"; tag "NMDFinderB:Last_exon"; exon_id "STRG.4.8_Exon002"; gene_type "protein_coding";
chr12	FirstORF_NoStopRequired	CDS	131932175	131932312	0	+	0	transcript_name "STRG.4.8"; gene_name "ENSG00000177192.14"; transcript_type "protein_coding"; tag "NMDFinderB:Last_exon";transcript_name "STRG.4.8"; gene_name "ENSG00000177192.14"; transcript_type "protein_coding"; tag "NMDFinderB:Last_exon"; gene_type "protein_coding";
chr12	FirstORF_NoStopRequired	exon	131943539	131943857	0	+	.	transcript_name "STRG.4.8"; gene_name "ENSG00000177192.14"; transcript_type "protein_coding"; tag "NMDFinderB:Last_exon"; exon_id "STRG.4.8_Exon003"; gene_type "protein_coding";
chr12	FirstORF_NoStopRequired	CDS	131943539	131943586	0	+	0	transcript_name "STRG.4.8"; gene_name "ENSG00000177192.14"; transcript_type "protein_coding"; tag "NMDFinderB:Last_exon";transcript_name "STRG.4.8"; gene_name "ENSG00000177192.14"; transcript_type "protein_coding"; tag "NMDFinderB:Last_exon"; gene_type "protein_coding";
chr12	FirstORF_NoStopRequired	stop_codon	131943584	131943586	0	+	0	transcript_name "STRG.4.8"; gene_name "ENSG00000177192.14"; transcript_type "protein_coding"; tag "NMDFinderB:Last_exon"; gene_type "protein_coding";
chr12	FirstORF_NoStopRequired	UTR	131943587	131943857	0	+	.	transcript_name "STRG.4.8"; gene_name "ENSG00000177192.14"; transcript_type "protein_coding"; tag "NMDFinderB:Last_exon"; gene_type "protein_coding";
chr12	FirstORF_NoStopRequired	transcript	131929765	131941422	0	+	.	transcript_name "STRG.4.9"; gene_name "ENSG00000177192.14"; transcript_type "noncoding"; tag "NMDFinderB:No_stop";transcript_name "STRG.4.9"; gene_name "ENSG00000177192.14"; transcript_type "noncoding"; tag "NMDFinderB:No_stop"; tag "FiveUTR_nEx":"0"; tag "FiveUTR_len":"0"; tag "ThreeUTR_nEx":"0"; tag "ThreeUTR_len":"0"; tag "ThreeUTR_nEx_AfterFirst50":"0"; tag "CDSLen":"0"; tag "Introns":"chr12:131929796-131929906:+,chr12:131930135-131932174:+,chr12:131932312-131932721:+,chr12:131932833-131941291:+"; gene_type "protein_coding";
chr12	FirstORF_NoStopRequired	exon	131929765	131929796	0	+	.	transcript_name "STRG.4.9"; gene_name "ENSG00000177192.14"; transcript_type "noncoding"; tag "NMDFinderB:No_stop"; exon_id "STRG.4.9_Exon001"; gene_type "protein_coding";
chr12	FirstORF_NoStopRequired	exon	131929907	131930135	0	+	.	transcript_name "STRG.4.9"; gene_name "ENSG00000177192.14"; transcript_type "noncoding"; tag "NMDFinderB:No_stop"; exon_id "STRG.4.9_Exon002"; gene_type "protein_coding";
chr12	FirstORF_NoStopRequired	exon	131932175	131932312	0	+	.	transcript_name "STRG.4.9"; gene_name "ENSG00000177192.14"; transcript_type "noncoding"; tag "NMDFinderB:No_stop"; exon_id "STRG.4.9_Exon003"; gene_type "protein_coding";
chr12	FirstORF_NoStopRequired	exon	131932722	131932833	0	+	.	transcript_name "STRG.4.9"; gene_name "ENSG00000177192.14"; transcript_type "noncoding"; tag "NMDFinderB:No_stop"; exon_id "STRG.4.9_Exon004"; gene_type "protein_coding";
chr12	FirstORF_NoStopRequired	exon	131941292	131941422	0	+	.	transcript_name "STRG.4.9"; gene_name "ENSG00000177192.14"; transcript_type "noncoding"; tag "NMDFinderB:No_stop"; exon_id "STRG.4.9_Exon005"; gene_type "protein_coding";
chr12	FirstORF_NoStopRequired	transcript	131940927	131943859	0	+	.	transcript_name "STRG.4.10"; gene_name "ENSG00000177192.14"; transcript_type "protein_coding"; tag "NMDFinderB:Long_exon";transcript_name "STRG.4.10"; gene_name "ENSG00000177192.14"; transcript_type "protein_coding"; tag "NMDFinderB:Long_exon"; tag "FiveUTR_nEx":"1"; tag "FiveUTR_len":"237"; tag "ThreeUTR_nEx":"2"; tag "ThreeUTR_len":"985"; tag "ThreeUTR_nEx_AfterFirst50":"2"; tag "CDSLen":"156"; tag "Introns":"chr12:131941983-131943538:+"; gene_type "protein_coding";
chr12	FirstORF_NoStopRequired	exon	131940927	131941983	0	+	.	transcript_name "STRG.4.10"; gene_name "ENSG00000177192.14"; transcript_type "protein_coding"; tag "NMDFinderB:Long_exon"; exon_id "STRG.4.10_Exon001"; gene_type "protein_coding";
chr12	FirstORF_NoStopRequired	UTR	131940927	131941163	0	+	.	transcript_name "STRG.4.10"; gene_name "ENSG00000177192.14"; transcript_type "protein_coding"; tag "NMDFinderB:Long_exon"; gene_type "protein_coding";
chr12	FirstORF_NoStopRequired	CDS	131941164	131941319	0	+	0	transcript_name "STRG.4.10"; gene_name "ENSG00000177192.14"; transcript_type "protein_coding"; tag "NMDFinderB:Long_exon";transcript_name "STRG.4.10"; gene_name "ENSG00000177192.14"; transcript_type "protein_coding"; tag "NMDFinderB:Long_exon"; gene_type "protein_coding";
chr12	FirstORF_NoStopRequired	start_codon	131941164	131941166	0	+	0	transcript_name "STRG.4.10"; gene_name "ENSG00000177192.14"; transcript_type "protein_coding"; tag "NMDFinderB:Long_exon"; gene_type "protein_coding";
chr12	FirstORF_NoStopRequired	stop_codon	131941317	131941319	0	+	0	transcript_name "STRG.4.10"; gene_name "ENSG00000177192.14"; transcript_type "protein_coding"; tag "NMDFinderB:Long_exon"; gene_type "protein_coding";
chr12	FirstORF_NoStopRequired	UTR	131941320	131941983	0	+	.	transcript_name "STRG.4.10"; gene_name "ENSG00000177192.14"; transcript_type "protein_coding"; tag "NMDFinderB:Long_exon"; gene_type "protein_coding";
chr12	FirstORF_NoStopRequired	exon	131943539	131943859	0	+	.	transcript_name "STRG.4.10"; gene_name "ENSG00000177192.14"; transcript_type "protein_coding"; tag "NMDFinderB:Long_exon"; exon_id "STRG.4.10_Exon002"; gene_type "protein_coding";
chr12	FirstORF_NoStopRequired	UTR	131943539	131943859	0	+	.	transcript_name "STRG.4.10"; gene_name "ENSG00000177192.14"; transcript_type "protein_coding"; tag "NMDFinderB:Long_exon"; gene_type "protein_coding";
chr12	input_gtf	gene	131934642	131934928	.	+	.	gene_name "ENSG00000273568.1"; gene_type "noncoding";
chr12	FirstORF_NoStopRequired	transcript	131934642	131934928	0	+	.	transcript_name "STRG.5.1"; gene_name "ENSG00000273568.1"; transcript_type "noncoding"; tag "NMDFinderB:No_CDS";transcript_name "STRG.5.1"; gene_name "ENSG00000273568.1"; transcript_type "noncoding"; tag "NMDFinderB:No_CDS"; tag "FiveUTR_nEx":"0"; tag "FiveUTR_len":"0"; tag "ThreeUTR_nEx":"0"; tag "ThreeUTR_len":"0"; tag "ThreeUTR_nEx_AfterFirst50":"0"; tag "CDSLen":"0"; tag "Introns":""; gene_type "noncoding";
chr12	FirstORF_NoStopRequired	exon	131934642	131934928	0	+	.	transcript_name "STRG.5.1"; gene_name "ENSG00000273568.1"; transcript_type "noncoding"; tag "NMDFinderB:No_CDS"; exon_id "STRG.5.1_Exon001"; gene_type "noncoding";
```

Also note that the source column in the gtf get's overwritten with this script, according to the options used with this script.

## Conlusion
This script is a simple way to convert a gtf file from the format used by the gencode project to the format used by the leafcutter project. It is not a complete solution, but it should be enough to get you started. It also might be handy for annotating existing gtf files with NMDFinderB my python implementation of the NMDFinderB algorithm. If you have any questions or suggestions, please let me know.

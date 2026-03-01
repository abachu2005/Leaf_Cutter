#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
#
#
######################################################################
# @author      : bjf79 (bjf79@midway2-login1.rcc.local)
# @file        : bedparse_translate_transcripts
# @created     : Friday Jun 21, 2024 12:26:19 CDT
#
# @description : 
######################################################################

import sys
import bedparse
import argparse
import gzip
import re
import pysam
import subprocess
from io import StringIO
from collections import defaultdict
import shutil
import pandas as pd
import csv
import logging
import tempfile

def reorder_gtf(gtf_stringio, output_gtf, mode='w'):
    """Reorder GTF lines by gene and transcript, ensuring correct hierarchical sorting,
    and write to output file."""
    # Read the GTF data into a pandas DataFrame
    columns = ['seqname', 'source', 'feature', 'start', 'end', 'score', 'strand', 'frame', 'attribute']
    df = pd.read_csv(gtf_stringio, sep='\t', names=columns, comment='#')

    # Extract gene_id and transcript_id from the attribute column
    df['gene_name'] = df['attribute'].str.extract(r'gene_name "([^"]+)"')
    df['transcript_name'] = df['attribute'].str.extract(r'transcript_name "([^"]+)"')

    # Determine if the line is a gene line
    df['is_gene'] = df['transcript_name'].isnull()

    # Create a column for the start position of the parent gene
    df['start_parent_gene'] = df.groupby('gene_name')['start'].transform('min')

    # Create a column for the start position of the parent transcript
    df['start_parent_transcript'] = df.groupby('transcript_name')['start'].transform('min')

    # Fill NaNs in start_parent_transcript with start_parent_gene for gene lines
    df['start_parent_transcript'] = df['start_parent_transcript'].fillna(df['start_parent_gene'])

    df = df.copy()
    df.loc[:, 'seqname'] = df['seqname'].astype(str)

    # Sort by seqname, start_parent_gene, gene_id, is_gene, start_parent_transcript, transcript_id, start
    df_sorted = df.sort_values(by=['seqname', 'start_parent_gene', 'gene_name','strand', 'is_gene', 'start_parent_transcript', 'transcript_name', 'start'], ascending=[True, True, True, True, False, True, True, True]).reset_index(drop=True)

    # Drop the helper columns before writing to the output
    df_sorted = df_sorted.drop(columns=['gene_name', 'transcript_name', 'is_gene', 'start_parent_gene', 'start_parent_transcript'])

    # Write the sorted DataFrame to the output GTF file
    df_sorted.to_csv(output_gtf, sep='\t', header=False, index=False, mode=mode, quoting=csv.QUOTE_NONE)
# reorder_gtf("scratch/Chicken_ensemblv84.Reannotated.gtf", "scratch/Chicken_ensemblv84.Reannotated.sorted.gtf", mode='w')
# reorder_gtf("GenomeFiles/Chicken_ensemblv84/Reannotated.B.gtf", "scratch/Chicken_ensemblv84.Reannotated.sorted.gtf")

def run_bedparse_gtf2bed(gtf_file, *args, n=None):
    """
    Wrapper around command line `bedparse gtf2bed`.
    Parameters:
    gtf_file (str): Path to the GTF file.
    *args: Additional arguments for the bedparse command.
    n (int, optional): Number of lines from the GTF file to process. If None, the entire file is processed.
    Returns:
    str: The stdout output from the bedparse command, or None if an error occurred.
    """
    # create temp file of first n lines
    if n is not None:
        with tempfile.NamedTemporaryFile(delete=False, mode='w') as temp_gtf:
            with open(gtf_file, 'r') as f:
                for i, line in enumerate(f):
                    if i >= n:
                        break
                    temp_gtf.write(line)
            gtf_file = temp_gtf.name
    # Construct the bedparse gtf2bed command
    command = ['bedparse', 'gtf2bed', gtf_file]
    command.extend(args)
    # Run the command and capture the output
    result = subprocess.run(command, capture_output=True, text=True)
    # Check for errors
    if result.returncode != 0:
        print(f"Error running bedparse gtf2bed: {result.stderr}")
        return None
    return result.stdout

def count_bars_until_n_position(sequence, n):
    bar_count = 0
    non_bar_count = 0
    for i,char in enumerate(sequence):
        if non_bar_count == n:
            break
        if char == "|":
            bar_count += 1
        else:
            non_bar_count += 1
    return bar_count

def AddORF_Marks(sequence, StartMarkBasePosition=None, StopMarkBasePosition=None):
    """
    returns sequence with '^' at StartmarkBasePosition and '*' at StopMarkBasePosition. '|' strings are ignored in base position. 
    """
    StartMark = '^'
    StopMark = '*'
    if StartMarkBasePosition == None:
        StartMark = ''
        StartMarkBasePosition = 0
    if StopMarkBasePosition == None:
        StopMark = ''
        StopMarkBasePosition = 0
    OffsetToStartPosition = count_bars_until_n_position(sequence, StartMarkBasePosition)
    OffsetToStopPosition = count_bars_until_n_position(sequence, StopMarkBasePosition)
    return sequence[0:StartMarkBasePosition + OffsetToStartPosition] + StartMark + sequence[(StartMarkBasePosition + OffsetToStartPosition):(StopMarkBasePosition+OffsetToStopPosition)] + StopMark + sequence[(StopMarkBasePosition+OffsetToStopPosition):]
AddORF_Marks("||ATG|G|ATAGG", 1, 5)

def extract_sequence(self, fasta_obj, AddMarksForORF=False):
    """
    ...method to be used on bedparse.bedline object. will monkey patch it into bedparse.bedline class
    will also add '|' between blocks to mark exon junctions
    """
    # Extract information from the BED entry
    chrom = self.chr
    start = self.start
    end = self.end
    strand = self.strand if hasattr(self, 'strand') else '+'
    block_sizes = list(map(int, self.exLengths.split(','))) if hasattr(self, 'exLengths') else [self.end - self.start]
    block_starts = list(map(int, self.exStarts.split(','))) if hasattr(self, 'exStarts') else [0]

    sequence = ''
    # Retrieve the sequence for each block and concatenate them
    try:
        if block_sizes and block_starts:
            for block_start, block_size in zip(block_starts, block_sizes):
                block_start_genomic = start + block_start
                block_end_genomic = block_start_genomic + block_size
                block_seq = fasta_obj.fetch(chrom, block_start_genomic, block_end_genomic)
                sequence += "|" + block_seq
            sequence = sequence.rstrip('|').lstrip('|')
        else:
            sequence = fasta_obj.fetch(chrom, start, end)
    except KeyError:
        # if contig not present in fasta
        sequence = ""
    # Reverse complement if on the negative strand
    if strand == '-':
        sequence = reverse_complement(sequence)
        # Add ORF markers if requested
    if AddMarksForORF and self.cds():
        if self.utr(which=5):
            cds_relative_start = sum([int(i) for i in self.utr(which=5).exLengths.split(',')])
        else:
            cds_relative_start = 0
        cds_relative_end = cds_relative_start + sum([int(i) for i in self.cds().exLengths.split(',')])
        sequence = AddORF_Marks(sequence, cds_relative_start, cds_relative_end)
    return sequence

def reverse_complement(seq):
    # Helper function to reverse complement a DNA sequence
    complement = str.maketrans('ATCGatcg', 'TAGCtagc')
    return seq.translate(complement)[::-1]

# Monkey-patch the BEDLine class
bedparse.bedline.extract_sequence = extract_sequence
bedparse.bedline.reverse_complement = staticmethod(reverse_complement)


def insert_marks_for_longset_ORF(sequence, require_ATG=True, require_STOP=True):
    """
    return sequence with "^" to mark start codon, "*" to mark stop for longest ORF. If not require_ATG, can start translation from beginning, which could be useful if transcript starts are not well defined. If not require_STOP, translation does not need stop codon at end. "|" characters (which I use mark splice junctions) are ignored, allowing codons to cross exon junctions.
    """
    if require_ATG and require_STOP:
        regex = r"(?=(\|?A\|?T\|?G\|?(?:(?!\|?T\|?A\|?A|\|?T\|?A\|?G|\|?T\|?G\|?A)\|?[ACGTN]\|?[ACGTN]\|?[ACGTN])*(?:\|?T\|?A\|?A|\|?T\|?A\|?G|\|?T\|?G\|?A)))"
    elif require_ATG and not require_STOP:
        regex = r"(?=(\|?A\|?T\|?G\|?(?:(?!\|?T\|?A\|?A|\|?T\|?A\|?G|\|?T\|?G\|?A)\|?[ACGTN]\|?[ACGTN]\|?[ACGTN])*(?:\|?T\|?A\|?A|\|?T\|?A\|?G|\|?T\|?G\|?A)*))"
    elif (not require_ATG) and require_STOP:
        regex = r"(?=((?:(?!\|?T\|?A\|?A|\|?T\|?A\|?G|\|?T\|?G\|?A)\|?[ACGTN]\|?[ACGTN]\|?[ACGTN])*(?:\|?T\|?A\|?A|\|?T\|?A\|?G|\|?T\|?G\|?A)))"
    else:
        regex = r"(?=(\|?(?:(?!\|?T\|?A\|?A|\|?T\|?A\|?G|\|?T\|?G\|?A)\|?[ACGTN]\|?[ACGTN]\|?[ACGTN])*(?:\|?T\|?A\|?A|\|?T\|?A\|?G|\|?T\|?G\|?A)*))"
    try:
        longest_orf_match = max(re.findall(regex, sequence, flags=re.IGNORECASE), key = len)
    except ValueError:
        longest_orf_match = None
    if longest_orf_match:
        start_codon_pos = sequence.find(longest_orf_match)
        stop_codon_pos = start_codon_pos + len(longest_orf_match)
        return sequence[0:start_codon_pos] + "^" + longest_orf_match + "*" + sequence[stop_codon_pos:]
    else:
        return sequence
    
def insert_marks_for_first_ORF(sequence, require_STOP=True, min_ORF_len=0):
    """
    return sequence with "^" to mark start codon, "*" to mark stop for longest ORF. If not require_STOP, translation does not need stop codon at end. "|" characters (which I use mark splice junctions) are ignored, allowing codons to cross exon junctions. min_ORF_len in codons after start codon.
    """
    if require_STOP:
        regex = r"^.*?(\|?A\|?T\|?G\|?(?:(?!\|?T\|?A\|?A|\|?T\|?A\|?G|\|?T\|?G\|?A)\|?[ACGTN]\|?[ACGTN]\|?[ACGTN]){" + str(min_ORF_len) + r",}((?:\|?T\|?A\|?A|\|?T\|?A\|?G|\|?T\|?G\|?A))).*$"
    else:
        regex = r"^.*?(\|?A\|?T\|?G\|?(?:(?!\|?T\|?A\|?A|\|?T\|?A\|?G|\|?T\|?G\|?A)\|?[ACGTN]\|?[ACGTN]\|?[ACGTN]){" + str(min_ORF_len) + r",}((?:\|?T\|?A\|?A|\|?T\|?A\|?G|\|?T\|?G\|?A))?).*$"
    first_ORF_match = re.match(regex, sequence, flags=re.IGNORECASE)
    # import pdb; pdb.set_trace()
    if first_ORF_match:
        start_codon_pos = first_ORF_match.start(1)
        if first_ORF_match.group(2):
            stop_codon_pos = first_ORF_match.end(1)
            return sequence[0:start_codon_pos] + "^" + first_ORF_match.group(1) + "*" + sequence[stop_codon_pos:]
        else:
            return sequence[0:start_codon_pos] + "^" + sequence[start_codon_pos:]
    else:
        return sequence
    
def insert_marks_for_defined_ORF(sequence, start_codon_pos=None):
    """
    return sequence with "^" to mark start codon at provided string position, "*" to mark stop for longest ORF. If no stop codon is found, then no * will be inserted.
    """
    if start_codon_pos != None and start_codon_pos < len(sequence):
        # orf_match = re.search(r"^\|?(A\|?T\|?G\|?(?:\|?[AGCTN]\|?[AGCTN]\|?[ACGTN])*?)\|?(?:T\|?A\|?A|T\|?A\|?G|T\|?G\|?A)", sequence[start_codon_pos:], flags=re.IGNORECASE)
        # orf_match_nostop = re.search(r"^\|?(A\|?T\|?G\|?(?:\|?[AGCTN]\|?[AGCTN]\|?[ACGTN])+)", sequence[start_codon_pos:], flags=re.IGNORECASE)
        orf_match = re.search(r"^\|?((?:\|?[AGCTN]\|?[AGCTN]\|?[ACGTN])*?)\|?(?:T\|?A\|?A|T\|?A\|?G|T\|?G\|?A)", sequence[start_codon_pos:], flags=re.IGNORECASE)
        orf_match_nostop = re.search(r"^\|?((?:\|?[AGCTN]\|?[AGCTN]\|?[ACGTN])+)", sequence[start_codon_pos:], flags=re.IGNORECASE)
        if orf_match:
            # orf match with start and stop codon
            return sequence[:start_codon_pos] + "^" + orf_match.group(0) + "*" + sequence[orf_match.span(0)[1] + start_codon_pos:]
        elif orf_match_nostop:
            # orf match with no stop
            return sequence[:start_codon_pos] + "^" + sequence[start_codon_pos:]
        else:
            return sequence
    else:
        return sequence
# insert_marks_for_defined_ORF("GGGATGAAAGGGAAA|GGG|T|AA|GGGAAA", 3)
# insert_marks_for_defined_ORF("GGGATGAAAGGGAAA|GGG|TT|AA|GGGAAA", 3)

def Is_bedline_complete(bedline):
    """
    after run_bedparse_gtf2bed, if a transcript feature is read but only some of the child exons, then transcript feature spans longer than all the child exons and the bedline object is buggy.
    """
    return int(bedline.exStarts.split(',')[-1] ) + int(bedline.exLengths.split(',')[-1] ) + bedline.start == bedline.end


def get_NMD_detective_B_classification(sequence):
    """
    sequence should be marked with '^' for start, '*' for stop, and '|' for splice juncs
    """
    CDS = re.search(r"\^(\w+)\**", sequence.replace("|", ""))
    InternalStopExon = re.search(r"(^|\|)([\^ACGTNacgtn]*\*[ACGTNacgtn]*)\|", sequence)
    if "^" not in sequence or CDS == None:
        return "No CDS"
    elif re.search(r"\^[\w|]+$", sequence):
        return "No stop"
    elif re.search(r"(^|\|)[\^ACGTNacgtn]*\*[ACGTNacgtn]*$", sequence):
        return "Last exon"
    elif len(CDS.group(1)) <= 125:
        return "Start proximal"
    elif len(InternalStopExon.group(2)) >= 407:
        return "Long exon"
    elif re.search(r"\*[ACGTNacgtn]{0,50}\|[ACGTNacgtn]+$", sequence):
        return "50 nt rule"
    else:
        return "Trigger NMD"
# get_NMD_detective_B_classification("ACGTACG|CACGT")
# get_NMD_detective_B_classification("A^ATGACG|CACGT")
# get_NMD_detective_B_classification("A^CGTACG|CAC*GT")
# get_NMD_detective_B_classification("A^CGTA*CG|CACGT")
# get_NMD_detective_B_classification("ACG^TACG|" + "A" * 407 +"A*A|CACGT")
# get_NMD_detective_B_classification("ACG^TA" + "A" * 100 + "CG|" + "A" * 40 +"A*A|CACGT")
# get_NMD_detective_B_classification("ACG^TA" + "A" * 150 + "CG|A*" + "A"*60 + "|CACGT")

def get_NMD_detective_B_classification_number(NMD_detective_B_classification):
    OrdinalDict = {"Last exon":1,"Start proximal":2,"50 nt rule":3, "Long exon":4,"Trigger NMD":5, "No stop":6, "No CDS":7}
    try: 
        return OrdinalDict[NMD_detective_B_classification]
    except KeyError:
        return 8

def get_NMD_detective_B_classification_color(NMD_detective_B_classification):
    OrdinalDict = {"Last exon":"#08519c","Start proximal":"#6baed6","50 nt rule":"#c6dbef", "Long exon":"#fcbba1","Trigger NMD":"#de2d26", "No stop":"#a50f15", "No CDS":"#969696"}
    try: 
        return OrdinalDict[NMD_detective_B_classification]
    except KeyError:
        return "#252525"

def calculate_frames(bedline):
    """Calculate the frame for each CDS block."""
    CDS_bed12 = bedline.cds()
    if CDS_bed12:
        frames = []
        current_frame = 0
        bed6_blocks = list(CDS_bed12.bed12tobed6())
        if bedline.strand == '+':
            for cds in bed6_blocks:
                frames.append(current_frame)
                block_length = cds.end - cds.start
                current_frame = (current_frame + block_length) % 3
        elif bedline.strand == '-':
            for cds in reversed(bed6_blocks):
                frames.insert(0, current_frame)
                block_length = cds.end - cds.start
                current_frame = (current_frame + block_length) % 3
        # Adjusting the frames gets the right answer below. haven't figured out why. but i'm sure it makes the function work.
        frames = [2 if frame == 1 else 1 if frame == 2 else frame for frame in frames]
        return frames
    
def extract_codon(bedline, codon_type='start'):
    """
    Extract the start or stop codon from a BED12 object.
    
    Args:
        bedline (bedparse.bedline): A BED12 object.
        codon_type (str): Either 'start' or 'stop' to specify which codon to extract.
    
    Returns:
        bedparse.bedline: A BED12 object representing the codon.
    """
    if bedline.cds() is None:
        raise ValueError("The BED line does not contain any CDS region.")

    CDS_bed12 = bedline.cds()
    bed6_blocks = list(CDS_bed12.bed12tobed6())
    
    if codon_type == 'start':
        codon_blocks = []
        codon_length = 3
        if bedline.strand == '+':
            for block in bed6_blocks:
                block_length = block.end - block.start
                if block_length >= codon_length:
                    codon_blocks.append((block.start, block.start + codon_length))
                    break
                else:
                    codon_blocks.append((block.start, block.end))
                    codon_length -= block_length
        elif bedline.strand == '-':
            for block in reversed(bed6_blocks):
                block_length = block.end - block.start
                if block_length >= codon_length:
                    codon_blocks.append((block.end - codon_length, block.end))
                    break
                else:
                    codon_blocks.append((block.start, block.end))
                    codon_length -= block_length
            codon_blocks = list(reversed(codon_blocks))

    elif codon_type == 'stop':
        codon_blocks = []
        codon_length = 3
        if bedline.strand == '+':
            for block in reversed(bed6_blocks):
                block_length = block.end - block.start
                if block_length >= codon_length:
                    codon_blocks.append((block.end - codon_length, block.end))
                    break
                else:
                    codon_blocks.append((block.start, block.end))
                    codon_length -= block_length
            codon_blocks = list(reversed(codon_blocks))
        elif bedline.strand == '-':
            for block in bed6_blocks:
                block_length = block.end - block.start
                if block_length >= codon_length:
                    codon_blocks.append((block.start, block.start + codon_length))
                    break
                else:
                    codon_blocks.append((block.start, block.end))
                    codon_length -= block_length

    else:
        raise ValueError("codon_type must be either 'start' or 'stop'.")

    block_sizes = ",".join([str(e - s) for s, e in codon_blocks])
    block_starts = ",".join([str(s - codon_blocks[0][0]) for s, _ in codon_blocks])

    return bedparse.bedline([
        bedline.chr,
        codon_blocks[0][0],
        codon_blocks[-1][1],
        bedline.name,
        bedline.score,
        bedline.strand,
        codon_blocks[0][0],
        codon_blocks[-1][1],
        bedline.color,
        len(codon_blocks),
        block_sizes,
        block_starts
    ])

def gtf_formatted_bedline_tx(bedline, source='.', attributes_str=''):
    """
    input bedparse.bedline transcript object, return 9 field gtf line string, where the source and attributes fields are provided as optional argument, since those information cannot be inferred from the bedline object 
    """
    gtf_fields = [bedline.chr, source, 'transcript', str(bedline.start + 1), str(bedline.end), str(bedline.score), bedline.strand, '.', attributes_str]
    return '\t'.join(gtf_fields) + '\n'

def gtf_formatted_bedline_exons(bedline, source='.', attributes_str=''):
    """
    input bedparse.bedline transcript object, return 9 field gtf line strings, where the source and attributes fields are provided as optional argument, since those information cannot be inferred from the bedline object 
    """
    string_to_return = ""
    for exon in bedline.bed12tobed6(appendExN=True):
        gtf_fields = [exon.chr, source, 'exon', str(exon.start + 1), str(exon.end), str(exon.score), exon.strand, '.', attributes_str + f' exon_id "{exon.name}";']
        string_to_return += '\t'.join(gtf_fields) + '\n'
    return string_to_return

def gtf_formatted_bedline_cds(bedline, source='.', attributes_str=''):
    """
    input bedparse.bedline transcript object, return 9 field gtf line strings, where the source and attributes fields are provided as optional argument, since those information cannot be inferred from the bedline object 
    """
    string_to_return = ""
    if bedline.cds():
        for cds, frame in zip(bedline.cds().bed12tobed6(), calculate_frames(bedline)):
            gtf_fields = [cds.chr, source, 'CDS', str(cds.start + 1), str(cds.end), str(cds.score), cds.strand, str(frame), attributes_str]
            string_to_return += '\t'.join(gtf_fields) + attributes_str + '\n'
    return string_to_return



def gtf_formatted_bedline_utr_start_stop(bedline, source='.', attributes_str='', filter_NF_in_attributes_str=True):
    """
    input bedparse.bedline transcript object, return 9 field gtf line strings, where the source and attributes fields are provided as optional argument, since those information cannot be inferred from the bedline object. Note that start and stop have frame attribute that may be non-zero if ATG cross exon boundary. In Gencode (and maybe ensembl... have to check), when a start codon traverses an exon junction, exists as two lines, one for each piece of the codon. If the attributes string contains 'NF' (as in 'mRNA_end_NF'), don't write out start or stop codon.
    """
    string_to_return = ""
    if filter_NF_in_attributes_str and '_NF' in attributes_str:
        return string_to_return
    bedline_cds = bedline.cds()
    if bedline_cds:
        start = extract_codon(bedline_cds, codon_type='start')
        for cds, frame in zip(start.bed12tobed6(), calculate_frames(start)):
            gtf_fields = [cds.chr, source, 'start_codon', str(cds.start + 1), str(cds.end), str(cds.score), cds.strand, str(frame), attributes_str]
            string_to_return += '\t'.join(gtf_fields) + '\n'
        stop = extract_codon(bedline_cds, codon_type='stop')
        for cds, frame in zip(stop.bed12tobed6(), calculate_frames(stop)):
            gtf_fields = [cds.chr, source, 'stop_codon', str(cds.start + 1), str(cds.end), str(cds.score), cds.strand, str(frame), attributes_str]
            string_to_return += '\t'.join(gtf_fields) + '\n'
        UTR_fiveprime = bedline.utr(which=5)
        if UTR_fiveprime:
            for exon in UTR_fiveprime.bed12tobed6():
                gtf_fields = [exon.chr, source, 'UTR', str(exon.start + 1), str(exon.end), str(exon.score), exon.strand, '.', attributes_str]
                string_to_return += '\t'.join(gtf_fields) + '\n'
        UTR_threeprime = bedline.utr(which=3)
        if UTR_threeprime:
            for exon in UTR_threeprime.bed12tobed6():
                gtf_fields = [exon.chr, source, 'UTR', str(exon.start + 1), str(exon.end), str(exon.score), exon.strand, '.', attributes_str]
                string_to_return += '\t'.join(gtf_fields) + '\n'
    return string_to_return

def bed12_formatted_bedline(bedline, attributes_str='', color=''):
    bed12_fields = [bedline.chr, bedline.start, bedline.end, bedline.name, bedline.score, bedline.strand, bedline.cdsStart, bedline.cdsEnd, color, bedline.nEx, bedline.exLengths, bedline.exStarts]
    return '\t'.join([str(i) for i in bed12_fields]) + attributes_str + '\n'
    
def get_transcript_length(bedline):
    return sum([int(i) for i in bedline.exLengths.split(',')])

def get_tx_stats(bedline, fasta_obj):
    """
    Return dict of potential useful stats/attributes about the transcript
    """
    dict_out = {"FiveUTR_nEx":0, "FiveUTR_len":0, "ThreeUTR_nEx":0, "ThreeUTR_len":0, "ThreeUTR_nEx_AfterFirst50":0, "CDSLen":0, "Introns":""}
    FiveUTR = bedline.utr(which=5)
    ThreeUTR = bedline.utr(which=3)
    CDS = bedline.cds()
    Introns = bedline.introns()
    if CDS:
        dict_out["CDSLen"] = get_transcript_length(CDS)
    if FiveUTR:
        dict_out["FiveUTR_nEx"] = FiveUTR.nEx
        dict_out["FiveUTR_len"] = get_transcript_length(FiveUTR)
    if ThreeUTR:
        dict_out["ThreeUTR_nEx"] = ThreeUTR.nEx
        dict_out["ThreeUTR_len"] = get_transcript_length(ThreeUTR)
        dict_out["ThreeUTR_nEx_AfterFirst50"] = ThreeUTR.nEx - ThreeUTR.extract_sequence(fasta_obj)[0:50].count("|")
    if Introns:
        dict_out["Introns"]=','.join([f'{i.chr}:{i.start}-{i.end}:{i.strand}' for i in Introns.bed12tobed6()])
    return dict_out

def get_thickStart_thickStop_from_marked_seq(bedline, ORF_marked_sequence):
    CDS = re.search(r"\^\w+\*", ORF_marked_sequence.replace("|", ""))
    if CDS:
        TranscriptLength = sum([int(i) for i in bedline.exLengths.split(',')])
        relativeStart = CDS.span()[0]
        relativeStop = CDS.span()[1] - 2
        FivePrimeEdge = get_absolute_pos(bedline, relativeStart)
        ThreePrimeEdge = get_absolute_pos(bedline, relativeStop)
        return tuple(sorted([FivePrimeEdge, ThreePrimeEdge]))
    return bedline.cdsStart, bedline.cdsEnd

def get_absolute_pos(bedline, coord):
    """
    Similar to bedline.tx2genome(coord, stranded=True), but I think that function is buggy and doesn't work for getting last base in tx
    """
    cumulative_block_size = 0
    exLengths = [int(i) for i in bedline.exLengths.split(',')]
    exStarts = [int(i) for i in bedline.exStarts.split(',')]
    if bedline.strand == '+':
        for i, block_size in enumerate(exLengths):
            remainder = coord - cumulative_block_size
            cumulative_block_size += block_size
            if cumulative_block_size >= coord:
                absolute_pos = bedline.start + remainder + exStarts[i]
                break
    elif bedline.strand == '-':
        for i, block_size in enumerate(list(reversed(exLengths))):
            remainder = coord - cumulative_block_size
            cumulative_block_size += block_size
            if cumulative_block_size >= coord:
                absolute_pos = bedline.start + list(reversed(exStarts))[i] + block_size - remainder
                break
    return absolute_pos

def add_gene_type_to_gtf(gtf_io, gene_type_dict):
    """
    Add gene_type attribute to a GTF file stored in a StringIO object.
    Parameters:
    gtf_io (StringIO): StringIO object containing the GTF file data.
    gene_type_dict (dict): Dictionary mapping gene_name to gene_type.
    Returns:
    StringIO: A new StringIO object with the updated GTF data.
    """
    # Create a new StringIO object to store the modified GTF content
    updated_gtf_io = StringIO()
    # Seek to the beginning of the input StringIO object
    gtf_io.seek(0)
    # Read and process each line
    for line in gtf_io:
        # Skip comment lines
        if line.startswith("#"):
            updated_gtf_io.write(line)
            continue
        # Parse the gene_name attribute using a regex
        match = re.search(r'gene_name\s+"([^"]+)"', line)
        if match:
            gene_name = match.group(1)
            gene_type = gene_type_dict.get(gene_name, "unknown_gene_type")  # Default to "unknown_gene_type" if not found
            # Add the gene_type attribute to the line
            if 'gene_type' not in line:
                line = line.rstrip() + f' gene_type "{gene_type}";\n'
        # Write the modified (or unmodified) line to the new StringIO object
        updated_gtf_io.write(line)
    # Reset the cursor of the new StringIO object to the beginning
    updated_gtf_io.seek(0)
    return updated_gtf_io

def parse_args(args=None):
    parser = argparse.ArgumentParser(description="Helper script to reformat GTF file for compatibility with SpliceJunctionClassifier.py. More specifically, for a GTF to be compatible with that script, each gene feature and transcript feature must have a 'gene_type' and 'gene_name' attributes where 'gene_type' attribute value is 'protein_coding' for protein coding genes, and 'gene_name' attribute value is unique for each gene. Each child transcript feature must also have 'transcript_type' and 'transcript_name' attributes, where 'transcript_type' is 'protein_coding' for protein_coding (productive) transcript isoforms. Also, each protein_coding isoform must have a child 'exon' 'CDS', 'start_codon' and 'stop_codon' features (which also have the gene_type, gne_name, transcript_type, and transcript_name attributes of their parent features). As in Gencode v43 GTFs, transcripts that are not 'protein_coding' may also have 'CDS', 'start_codon' and 'start_codon' child features but if the transcript_type attribute is not 'protein_coding', these will not determine inclusion into the set of productive start/stop codons by SpliceJunctionClassifier.py. Furthermore, this script adds NMDetectiveB attributes to the transcript (see options for this feature). This is useful in cases when a GTF may be missing 'transcript_type' attribute, and if it is even missing 'CDS' features, this script can add the 'CDS', 'start_codon', and 'stop_codon' features after attempting to translate the transcript sequence and assign values to the 'transcript_type' attribute (eg, 'protein_coding' if NMDetectiveB result is 'Last exon' and 'noncoding if NMDetectiveB result is 'Trigger NMD'). This script also optionally can output bed12 file for each transcript. I have tested this on some GTF files from Gencode, Ensembl, and UCSC. Depending on the exact nature (eg attribute names) in the input GTF, you may have to alter some options to make the output suitable for SpliceJunctionClassifier.py.")
    parser.add_argument('-i', dest='gtf_in', required=True, help='Input GTF file. May alternatively use bed12 file to define input transcripts (see "-input_type" argument)')
    parser.add_argument('-o', dest='gtf_out', required=True, help='Output GTF file')
    parser.add_argument('-input_type', dest='input_type', choices=['gtf', 'bed12'], default='gtf', help='"-i" input file that contains transcript structures can be either gtf format, or bed12 format. If bed12 format, transcript_name, gene_name, transcript_type, and gene_type attributes must be added as columns 13, 14, 15, and 16, respectively. default: "%(default)s"')
    parser.add_argument('-fa', dest='fasta_in', required=True, help='Input FASTA file')
    parser.add_argument('-bed12_out', dest='bed12_out', help='Optional bed12 out. One line per transcript. Attributes as extra columns')
    parser.add_argument('-n', dest='n_lines', help='Number of lines to read in gtf. Useful for quick debugging, but reading only n lines may cause buggy behavior if a transcript feature line is processed but not all of its child (eg exon, or CDS) features', type=int)
    parser.add_argument('-infer_transcript_type_approach', dest='infer_transcript_type_approach', choices=['A', 'B', 'C'], default='A', help='Approach to determine the transcript_type attribute value in output. (A) Use existing value. (B) Value is "protein_coding" if a the transcript contains child features of type "CDS". Value is "noncoding" otherwise. (C) Translate the transcript and use NMDetectiveB on the ORF. See -translation_approach and -NMDetectiveB_coding_threshold options to specify how this is done. default: "%(default)s"')
    parser.add_argument('-infer_gene_type_approach', dest='infer_gene_type_approach', choices=['A', 'B'], default='A', help='Approach to determine the gene_type attribute value in output. (A) Use existing value. (B) Value is "protein_coding" if gene has any child transcripts have attribute transcript_type "protein_coding". Value is "noncoding" otherwise. default: "%(default)s"')
    parser.add_argument('-transcript_type_attribute_name', dest='transcript_type_attribute_name', default='transcript_type', help='Name of the attribute in the input gtf file that defines the transcript_type attribute in the output file. Only relevant if -infer_transcript_type_approach==A. default: "%(default)s"')
    parser.add_argument('-gene_type_attribute_name', dest='gene_type_attribute_name', default='gene_type', help='Name of the attribute in the input gtf file that defines the gene_type attribute in the output file. Only relevant if -infer_gene_type_approach==A. default: "%(default)s"')
    parser.add_argument('-transcript_name_attribute_name', dest='transcript_name_attribute_name', default='transcript_name', help='Name of the attribute in the input gtf file that defines the transcript_name attribute in the output file. default: "%(default)s"')
    parser.add_argument('-gene_name_attribute_name', dest='gene_name_attribute_name', default='gene_name', help='Name of the attribute in the input gtf file that defines the gene_name attribute in the output file. default: "%(default)s"')
    parser.add_argument('-extra_attributes', dest='extra_attributes', default='transcript_support_level,tag,ccds_id',
                        help='Extra transcript attributes (comma delimited quoted string). default: "%(default)s"')
    parser.add_argument('-NMDetectiveB_coding_threshold', dest='NMDetectiveB_coding_threshold', type=int, choices=[1,2,3,4,5,6,7], default=5, help='NMDetectiveB classifies each transcript into 7 ordinal categories, from the most coding potential to the least coding potential: (1) Last exon (2) Start proximal (3) 50nt rule (4) Long exon (5) Trigger NMD (6) No stop (7) No CDS. Transcripts with classified as this NMDetective value or greater will be assigned transcript_type attribute value of "noncoding", while others will be value of "protein_coding". default: "%(default)s"')
    parser.add_argument('-translation_approach', dest='translation_approach', choices=['A', 'B', 'C', 'D', 'E'], default='B', help='Approach to use NMDetective to annotate CDS in output for genes where gene_biotype/gene_type == "protein_coding", some of which may not have annotated CDS in input (eg transcript_type=="processed_transcript"). Possible approaches: (A) using annotated ORF if ORF is present in input with 5UTR and 3UTR. If 3UTR is present and 5UTR is absent (suggesting stop codon is annotated but start codon may be outside of the transcript bounds), search for longest ORF within transcript bounds where a start codon is not required at the beginning of ORF. Similarly, if 5UTR is present but 3UTR is absent, search for longest ORF without requiring stop codon. If neither UTR is present, or if no CDS is annotated in input, search for longest ORF, not requiring start or stop within transcript bounds. I think this approach might be useful to correctly identify the ORF, even if transcript bounds are not accurate, but it has the downside that true "processed_transcripts" with a internal TSS that eliminates the correct start codon, may be erronesously be classified as "Last exon" (ie productive) transcripts by NMDFinderB. (B) Use only annotated CDS. In effect, output gtf is the same except start_codon and stop_codon features are added even if not present in input. This would be useful for dealing with "processed_transcripts" properly by NMDFinder, but I havent checked whether CDS annotations in poorly annotated species (eg lamprey, chicken, etc) are reasonable, which could be a problem for Yangs script. (C) use annotated CDS if present, and use first ATG if no CDS present (minimum ORF length of 30 codons, not including start or stop). (D) Same as (C) but no minimum ORF length. (E) Use first ORF with minimum length > 42, regardless of annotation')
    parser.add_argument('-v', '--verbose', action='store_true', help='Increase verbosity')
    return parser.parse_args(args)


def setup_logging(verbose):
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format='%(asctime)s - %(levelname)s - %(message)s')

def main(args=None):
    args = parse_args(args)
    setup_logging(args.verbose)

    # Example usage of the parsed arguments
    logging.info("Input GTF file: %s", args.gtf_in)
    logging.info("Output GTF file: %s", args.gtf_out)
    logging.info("Input FASTA file: %s", args.fasta_in)
    if args.extra_attributes:
        extra_attributes = args.extra_attributes.split(',')
        logging.info("Extra attributes: %s", extra_attributes)
    else:
        logging.info("No extra attributes provided.")
    
    # Open the FASTA file using pysam
    fasta_obj = pysam.FastaFile(args.fasta_in)

    required_attributes = ','.join([args.transcript_name_attribute_name, args.gene_name_attribute_name, args.transcript_type_attribute_name, args.gene_type_attribute_name])
    attributes_to_extract = ','.join([required_attributes, args.extra_attributes])

    # gtf2bed
    logging.info(f'Reading in transcripts and converting transcripts to bedparse.bedline objects...')
    if args.input_type == "gtf":
        bed12 = run_bedparse_gtf2bed(args.gtf_in, '--extraFields', attributes_to_extract, n=args.n_lines)
    elif args.input_type == "bed12":
        with open(args.gtf_in, 'r') as f:
            bed12 = f.read()
    NumTrancsripts = len(bed12.splitlines())
    logging.info(f'Read in {NumTrancsripts} from input.')

    logging.info(f'Processing transcripts.')
    gtf_stringio = StringIO()
    # Initialize an empty dictionary to keep coordinates of each transcript for each gene. Will need to write out gene coordinates as most extensive span of child transcripts on each chromosome:strand pair, because some genes appear on more than one chromosome, (eg X and Y in Gencode gtf), or even on the same chromosome but different strands (eg some TRNAA gene appears on chr6 + strand and chr6 - strand in human RefSeq annotations from UCSC)
    gene_coords_dict = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda : defaultdict(set))))
    # gene_coords_dict[gene_name][transcript.chr][transcript.strand]['end'] = SetOfTranscriptEnds
    gene_types_dict = defaultdict(lambda: defaultdict((set)))
    bed_out_fh = None
    if args.bed12_out:
        bed_out_fh = open(args.bed12_out, 'w')
    for i,l in enumerate(bed12.splitlines()):
        if i >= 0:
            if i % 1000 == 0: logging.debug(f'processed {i} trancsripts for output...')
            lsplit = l.split('\t')
            transcript_name, gene_name, transcript_type, gene_type = lsplit[12:16]
            extra_attribute_values = lsplit[16:]
            transcript = bedparse.bedline(lsplit[0:12])
            if not Is_bedline_complete(transcript):
                continue
            source = "input_gtf"
            transcript_attributes = f'transcript_name "{transcript_name}"; gene_name "{gene_name}";'
            transcript_out = transcript
            NMDFinderB = "NA"
            #Determine NMDetectiveB classification
            if args.translation_approach == 'A':
                if transcript.cds() and transcript.utr(which=5) and transcript.utr(which=3):
                    source = "input_gtf"
                    sequence = transcript.extract_sequence(fasta_obj, AddMarksForORF=True)
                elif transcript.cds() and transcript.utr(which=5) and not transcript.utr(which=3):
                    source = "LongestORF_NoStopRequired"
                    sequence = insert_marks_for_longset_ORF(transcript.extract_sequence(fasta_obj), require_STOP = False)
                    thickStart, thickStop = get_thickStart_thickStop_from_marked_seq(transcript, sequence)
                    transcript_out = bedparse.bedline([transcript.chr, transcript.start, transcript.end, transcript.name, transcript.score, transcript.strand, thickStart, thickStop, transcript.color, transcript.nEx, transcript.exLengths, transcript.exStarts])
                elif transcript.cds() and not transcript.utr(which=5) and transcript.utr(which=3):
                    source = "LongestORF_NoStartRequired"
                    sequence = insert_marks_for_longset_ORF(transcript.extract_sequence(fasta_obj), require_STOP = False)
                    thickStart, thickStop = get_thickStart_thickStop_from_marked_seq(transcript, sequence)
                    transcript_out = bedparse.bedline([transcript.chr, transcript.start, transcript.end, transcript.name, transcript.score, transcript.strand, thickStart, thickStop, transcript.color, transcript.nEx, transcript.exLengths, transcript.exStarts])
                else:
                    source = "LongestORF_NeitherRequired"
                    sequence = insert_marks_for_longset_ORF(transcript.extract_sequence(fasta_obj), require_ATG=False, require_STOP = False)
                    thickStart, thickStop = get_thickStart_thickStop_from_marked_seq(transcript, sequence)
                    transcript_out = bedparse.bedline([transcript.chr, transcript.start, transcript.end, transcript.name, transcript.score, transcript.strand, thickStart, thickStop, transcript.color, transcript.nEx, transcript.exLengths, transcript.exStarts])
                NMDFinderB = get_NMD_detective_B_classification(sequence)
            elif args.translation_approach == 'B':
                sequence = transcript.extract_sequence(fasta_obj, AddMarksForORF=True)
                NMDFinderB = get_NMD_detective_B_classification(sequence)
            elif args.translation_approach == 'C':
                if transcript.cds():
                    sequence = transcript.extract_sequence(fasta_obj, AddMarksForORF=True)
                    source = "input_gtf"
                else:
                    source = "FirstORF_NoStopRequired"
                    sequence = insert_marks_for_first_ORF(transcript.extract_sequence(fasta_obj), require_STOP = False, min_ORF_len = 30)
                    thickStart, thickStop = get_thickStart_thickStop_from_marked_seq(transcript, sequence)
                    transcript_out = bedparse.bedline([transcript.chr, transcript.start, transcript.end, transcript.name, transcript.score, transcript.strand, thickStart, thickStop, transcript.color, transcript.nEx, transcript.exLengths, transcript.exStarts])
                NMDFinderB = get_NMD_detective_B_classification(sequence)
            elif args.translation_approach == 'D':
                if transcript.cds():
                    sequence = transcript.extract_sequence(fasta_obj, AddMarksForORF=True)
                    source = "input_gtf"
                else:
                    source = "FirstORF_NoStopRequired"
                    sequence = insert_marks_for_first_ORF(transcript.extract_sequence(fasta_obj), require_STOP = False, min_ORF_len = 0)
                    thickStart, thickStop = get_thickStart_thickStop_from_marked_seq(transcript, sequence)
                    transcript_out = bedparse.bedline([transcript.chr, transcript.start, transcript.end, transcript.name, transcript.score, transcript.strand, thickStart, thickStop, transcript.color, transcript.nEx, transcript.exLengths, transcript.exStarts])
                NMDFinderB = get_NMD_detective_B_classification(sequence)
            elif args.translation_approach == 'E':
                source = "FirstORF_NoStopRequired"
                sequence = insert_marks_for_first_ORF(transcript.extract_sequence(fasta_obj), require_STOP = False, min_ORF_len = 42)
                thickStart, thickStop = get_thickStart_thickStop_from_marked_seq(transcript, sequence)
                transcript_out = bedparse.bedline([transcript.chr, transcript.start, transcript.end, transcript.name, transcript.score, transcript.strand, thickStart, thickStop, transcript.color, transcript.nEx, transcript.exLengths, transcript.exStarts])
                NMDFinderB = get_NMD_detective_B_classification(sequence)
            NMDFinderB_NoWhitespace = NMDFinderB.replace(' ', '_')
            NMDFinderB_number = get_NMD_detective_B_classification_number(NMDFinderB)
            # Determine transcript_type
            if args.infer_transcript_type_approach == 'A':
                transcript_type_out = transcript_type
            elif args.infer_transcript_type_approach == 'B':
                transcript_type_out = "protein_coding" if transcript.cds() else "noncoding"
            elif  args.infer_transcript_type_approach == 'C':
                transcript_type_out = "protein_coding" if NMDFinderB_number < args.NMDetectiveB_coding_threshold else "noncoding"
            extra_calculated_transcript_attributes = get_tx_stats(transcript_out, fasta_obj)
            transcript_attributes += f' transcript_type "{transcript_type_out}"; tag "NMDFinderB:{NMDFinderB_NoWhitespace}";'
            transcript_attributes_longer = transcript_attributes + ' ' + '; '.join([f'tag "{k}":"{v}"' for k,v in extra_calculated_transcript_attributes.items()]) + ';'
            _ = gtf_stringio.write(gtf_formatted_bedline_tx(transcript_out, source=source, attributes_str=transcript_attributes + transcript_attributes_longer))
            _ = gtf_stringio.write(gtf_formatted_bedline_exons(transcript_out, source=source, attributes_str=transcript_attributes))
            _ = gtf_stringio.write(gtf_formatted_bedline_cds(transcript_out, source=source, attributes_str=transcript_attributes))
            _ = gtf_stringio.write(gtf_formatted_bedline_utr_start_stop(transcript_out, source=source, attributes_str=transcript_attributes))
            if bed_out_fh is not None:
                _ = bed_out_fh.write(bed12_formatted_bedline(transcript, color=get_NMD_detective_B_classification_color(NMDFinderB), attributes_str='\t' + '\t'.join([gene_name, transcript_name, gene_type, transcript_type_out] + [NMDFinderB_NoWhitespace] + [str(i) for i in extra_calculated_transcript_attributes.values()] + extra_attribute_values)))
            # gene_dict contains gene level information needed to properly write out parent (gene-level) lines based on child (transcript-level) lines
            gene_coords_dict[gene_name][transcript.chr][transcript.strand]['start'].add(transcript.start)
            gene_coords_dict[gene_name][transcript.chr][transcript.strand]['end'].add(transcript.end)
            gene_types_dict[gene_name]['gene_types_in_input'].add(gene_type)
            gene_types_dict[gene_name]['trancscript_types'].add(transcript_type_out)
    if bed_out_fh is not None: bed_out_fh.close()

    logging.info('Writing gene (parent) feature coordiantes and based on child (transcript) feature coordinates')
    for gene, chrom_dict in gene_coords_dict.items():
        for chrom, strand_dict in chrom_dict.items():
            for StrandIteration, (strand, info_dict) in enumerate(strand_dict.items()):
                min_start = min(info_dict['start'])
                max_stop = max(info_dict['end'])
                _ = gtf_stringio.write(f'{chrom}\tinput_gtf\tgene\t{min_start+1}\t{max_stop}\t.\t{strand}\t.\tgene_name "{gene}";\n')
            if StrandIteration > 0:
                logging.warning(f"Transcripts for gene {gene} on {chrom} are on different strands. Writing {gene} gene feature on {chrom} for more than one strand")

    logging.info('Writing gene_type attributes')
    gene_types_dict_final = dict()
    for gene, info_dict in gene_types_dict.items():
        if args.infer_gene_type_approach == "A":
            if len(info_dict['gene_types_in_input']) != 1: raise ValueError(f"{gene} does not have a single gene_type in input")
            gene_types_dict_final[gene] = list(info_dict['gene_types_in_input'])[0]
        elif args.infer_gene_type_approach == "B":
            gene_types_dict_final[gene] = "protein_coding" if "protein_coding" in info_dict['trancscript_types'] else "noncoding"
    # Write out gene_type attributes
    gtf_stringio_updated = add_gene_type_to_gtf(gtf_stringio, gene_types_dict_final)

    logging.info('sorting and writing out gtf')
    # output filehandle
    with open(args.gtf_out, 'w') as output_fh:
        # transfer commented headers from original gtf
        with open(args.gtf_in, 'r') as input_fh:
            for l in input_fh:
                if l.startswith('#'):
                    _ = output_fh.write(l)
                else:
                    break
            _ = output_fh.write(f"#! args: {args}\n")
        reorder_gtf(gtf_stringio_updated, output_fh, mode='a')

if __name__ == "__main__":
    if hasattr(sys, 'ps1'):
        # main("-i /project2/yangili1/bjf79/ReferenceGenomes/Mouse_UCSC.mm39_GencodeComprehensive46/Reference.gtf -o scratch/Mouse_UCSC.mm39_GencodeComprehensive46.gtf -fa /project2/yangili1/bjf79/ReferenceGenomes/Mouse_UCSC.mm39_GencodeComprehensive46/Reference.GencodePrimary.fa -v -infer_gene_type_approach A -infer_transcript_type_approach A -transcript_name_attribute_name transcript_id -gene_name_attribute_name gene_id -n 10000 -bed12_out scratch/Mouse_UCSC.10K.bed".split(' '))
        # main("-i scratch/Mouse_UCSC.10K.bed -input_type bed12 -o scratch/Mouse_UCSC.mm39_GencodeComprehensive46.gtf -fa /project2/yangili1/bjf79/ReferenceGenomes/Mouse_UCSC.mm39_GencodeComprehensive46/Reference.GencodePrimary.fa -v -infer_gene_type_approach A -infer_transcript_type_approach A -transcript_name_attribute_name transcript_id -gene_name_attribute_name gene_id -n 10000 -bed12_out scratch/Mouse_UCSC.10K.Redone.bed".split(' '))
        # main("-i Maz -input_type bed12 -o scratch/Mouse_UCSC.mm39_GencodeComprehensive46.gtf -fa /project2/yangili1/bjf79/ReferenceGenomes/Mouse_UCSC.mm39_GencodeComprehensive46/Reference.GencodePrimary.fa -v -infer_gene_type_approach A -infer_transcript_type_approach A -transcript_name_attribute_name transcript_id -gene_name_attribute_name gene_id -n 10000 -bed12_out scratch/Mouse_UCSC.10K.Redone.bed".split(' '))
        # main("-i scratch/TRNAA.gtf -o scratch/TRNAA_reformated.gtf -fa /project2/yangili1/bjf79/ReferenceGenomes/Human_UCSC.hg38_GencodeComprehensive46/Reference.fa -v -infer_gene_type_approach B -infer_transcript_type_approach B -transcript_name_attribute_name transcript_id -gene_name_attribute_name gene_id".split(' '))
        main("-i /project2/yangili1/bjf79/ReferenceGenomes/Human_UCSC.hg38_GencodeComprehensive46/Reference.gtf -fa /project2/yangili1/bjf79/ReferenceGenomes/Human_UCSC.hg38_GencodeComprehensive46/Reference.fa -o scratch/test.gtf -bed12_out scratch/test.bed -n 10000 -v -infer_gene_type_approach B -infer_transcript_type_approach B -transcript_name_attribute_name transcript_id -gene_name_attribute_name gene_id".split(' '))

    else:
        main()

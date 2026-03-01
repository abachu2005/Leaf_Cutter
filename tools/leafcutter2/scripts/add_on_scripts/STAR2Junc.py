import sys
import gzip

arguments = sys.argv
input_file = arguments[1]
output = arguments[2]

with open(input_file, 'r') as fh:
    with gzip.open(output, 'wb') as out_fh:
        for line in fh:
            row = line.rstrip().split('\t')
            chrom = row[0]
            start = str(int(row[1])-1) # Coordinates are 1-based
            end = row[2]
            counts = row[6]
            if row[3] == '1':
                strand = '+'
            else:
                strand = '-'
                
            out_line = ('\t'.join([chrom, start, end, '.', counts, strand]) + '\n').encode()
            
            out_fh.write(out_line)
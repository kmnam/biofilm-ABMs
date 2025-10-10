"""
Helper script for setting up continuations of simulations on an HPC cluster. 

Authors:
    Kee-Myoung Nam

Last updated:
    10/10/2025
"""

import sys
import os
import glob
import re
import json

######################################################################
if __name__ == '__main__':
    # Collect input directories and next run index ...
    #
    # If there is a trailing '#' in the signature, whatever precedes the 
    # '#' (which may include wildcards) must be a number representing the 
    # replicate number (random seed)
    signature = sys.argv[1]
    signature_prefix = signature.rstrip('#')
    signature_modified = (
        signature.replace('#', '*') if signature.endswith('#') else signature
    )
    indirs = [
        d for d in glob.glob(signature_modified)
        if re.match(r'{}[0-9]+'.format(signature_prefix), d) is not None
    ]
    idx = int(sys.argv[2])
    runname = 'run{}'.format(idx)
    cmdfilename = sys.argv[3]

    with open(cmdfilename, 'w') as cmdf:
        # In each directory ...
        for indir in indirs:
            print('Parsing: {} ...'.format(indir))

            # Get all simulation files in the directory 
            filenames = [
                os.path.join(indir, f) for f in os.listdir(indir)
                if os.path.isfile(os.path.join(indir, f)) and
                f.endswith('.txt') and not f.endswith('lineage.txt')
            ]

            # If there is no file that ends with '_final.txt' ... 
            if not any(f.endswith('_final.txt') for f in filenames):
                # Get the timepoint associated with each file 
                filenames_with_times = []
                for filename in filenames:
                    with open(filename) as f:
                        for line in f:
                            if line.startswith('# t_curr ='):
                                time = float(line.split()[-1])
                                break
                    filenames_with_times.append((filename, time))

                # Sort the files by timepoint, and extract the last file 
                filenames_with_times.sort(key=lambda item: item[1])
                final_filename = filenames_with_times[-1][0]
                print('- Final file: {}'.format(final_filename))
               
                # Set up a continuation of this simulation
                #
                # First make a new directory 
                if idx == 1:
                    outdir = os.path.join(indir, runname)
                else:
                    outdir = os.path.join(os.path.dirname(indir.rstrip('/')), runname)
                try:
                    os.mkdir(outdir)
                except FileExistsError:
                    pass

                # Read in the .json file for the simulation, whose path can be 
                # inferred from the input directory name
                split = os.path.basename(final_filename).split('_')
                prefix = '_'.join(split[:-2])
                seed = int(split[-2])
                if idx == 1:
                    json_filename = os.path.join('json', prefix + '.json')
                else:
                    json_filename = os.path.join(indir, prefix + '.json')
                with open(json_filename) as f:
                    json_data = json.load(f)

                # Add in the final filename to the .json file as the initial 
                # frame for the continuation 
                json_data['init_filename'] = os.path.abspath(final_filename)

                # Write the updated .json file to the output directory
                with open(os.path.join(outdir, prefix + '.json'), 'w') as f:
                    json.dump(json_data, f, indent=4)

                # Stitch together the command for the continuation 
                cmd = './bin/abmConstAdhesion {} {} {}'.format(
                    os.path.abspath(os.path.join(outdir, prefix + '.json')), 
                    os.path.abspath(os.path.join(outdir, '{}_{}'.format(prefix, seed))), 
                    seed
                )
                print('- Writing command to file')
                cmdf.write(cmd + '\n')


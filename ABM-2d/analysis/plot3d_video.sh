#!/bin/bash

#######################################################################
DIR=$1
NFRAMES=$2
NMAX=$3

# Get list of files to plot 
FILES=`python3 plot3d_filenames.py ${DIR} ${NFRAMES} -1 ${NMAX}`

# Get final file 
i=0
for FILE in $FILES
do
    if [[ $i -eq $((NFRAMES-1)) ]]
    then
	FINAL=$FILE
    fi
    i=$((i+1))
done

# Plot final file with the given axes limits
PREFIX=${FINAL%.*}
PREFIX=${PREFIX%_*}
OUTFILE="${PREFIX}_final_frame.jpg"
POSITION=`python3 plot3d_frame.py ${FINAL} ${OUTFILE} | tail -n 1`

# Plot each file with the given axes limits 
for FILE in $FILES
do
    OUTFILE="${FILE%.*}_frame.jpg"
    python3 plot3d_frame.py ${FILE} ${OUTFILE} ${POSITION}
done

# Stitch together the files 
OUTFILE="${PREFIX}.avi"
python3 stitch_frames.py frame $FILES $OUTFILE


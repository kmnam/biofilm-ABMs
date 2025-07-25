#!/bin/bash

#######################################################################
DIR=$1
NCELLS=$2
NFRAMES=$3
SEED=$4
TMIN=$5

# Get list of files to plot 
FILES=`python3 generate_trajectories.py ${DIR} ${NCELLS} ${NFRAMES} ${SEED} ${TMIN}`

# Get prefix of boundary files
BASENAME=`basename $DIR` 
BOUNDARY_PREFIX="${DIR}/boundaries/${BASENAME}"

# For each file ... 
for FILE in $FILES
do
    # Get the number of frames 
    NFRAMES_IN_FILE=`cat $FILE | grep -v "#" | wc -l`

    # Plot the final frame in that trajectory
    OUTFILES=()
    PREFIX=${FILE%.*}
    OUTFILE="${PREFIX}_final_frame.jpg"
    POSITION=`python3 plot3d_trajectory_frame.py ${FILE} ${OUTFILE} ${BOUNDARY_PREFIX} -1 | tail -n 1`
    echo $OUTFILE
    echo $POSITION

    # Plot each preceding frame with the given axes limits
    for i in $(seq 0 $((NFRAMES_IN_FILE-1)))
    do
	echo $i
	echo $OUTFILE
	OUTFILE="${PREFIX}_idx${i}_frame.jpg"
	OUTFILES+=($OUTFILE)
	python3 plot3d_trajectory_frame.py ${FILE} ${OUTFILE} ${BOUNDARY_PREFIX} $i ${POSITION}
    done

    # Collect the final frame filename 
    OUTFILE="${PREFIX}_final_frame.jpg"
    OUTFILES+=($OUTFILE)

    # Stitch together the files 
    python3 stitch_frames.py None ${OUTFILES[@]} "${PREFIX}.avi"
done 

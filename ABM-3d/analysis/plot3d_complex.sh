#!/bin/bash

#######################################################################
DIR=$1
NFRAMES=$2
NMAX=$3
BASAL=$4
MAXDIM=$5

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
echo $FINAL

# Fix the axes limits by inferring them from the final file 
#
# Note that this must (for some reason) be done for the 3-D videos, but 
# not the 2-D videos 
XMIN=`python3 -c "import numpy as np; print(np.loadtxt('${FINAL}', delimiter='\t', comments='#')[:, 1].min())"`
XMAX=`python3 -c "import numpy as np; print(np.loadtxt('${FINAL}', delimiter='\t', comments='#')[:, 1].max())"`
YMIN=`python3 -c "import numpy as np; print(np.loadtxt('${FINAL}', delimiter='\t', comments='#')[:, 2].min())"`
YMAX=`python3 -c "import numpy as np; print(np.loadtxt('${FINAL}', delimiter='\t', comments='#')[:, 2].max())"`
ZMIN=`python3 -c "import numpy as np; print(np.loadtxt('${FINAL}', delimiter='\t', comments='#')[:, 3].min())"`
ZMAX=`python3 -c "import numpy as np; print(np.loadtxt('${FINAL}', delimiter='\t', comments='#')[:, 3].max())"`

# Plot final file with the given axes limits
PREFIX=${FINAL%.*}
PREFIX=${PREFIX%_*}
BASEPREFIX=`basename $PREFIX`
GRAPHFILE="${DIR}/graphs/${BASEPREFIX}_final_graph.txt"
OUTFILE="${DIR}/graphs/${BASEPREFIX}_final_complex_dim${MAXDIM}.jpg"
OUTFILES=()
if [[ $BASAL -eq 1 ]]
then
    POSITION=`python3 plot3d_complex_frame.py ${GRAPHFILE} ${FINAL} ${OUTFILE} ${MAXDIM} ${XMIN} ${XMAX} ${YMIN} ${YMAX} ${ZMIN} ${ZMAX} --basal | tail -n 1`
else 
    POSITION=`python3 plot3d_complex_frame.py ${GRAPHFILE} ${FINAL} ${OUTFILE} ${MAXDIM} ${XMIN} ${XMAX} ${YMIN} ${YMAX} ${ZMIN} ${ZMAX} | tail -n 1`
fi

# Plot each file with the given axes limits 
for FILE in $FILES
do
    BASENAME=`basename $FILE`
    BASENAME=${BASENAME%.*}
    GRAPHFILE="${DIR}/graphs/${BASENAME}_graph.txt"
    OUTFILE="${DIR}/graphs/${BASENAME}_complex_dim${MAXDIM}.jpg"
    OUTFILES+=($OUTFILE)
    python3 plot3d_complex_frame.py ${GRAPHFILE} ${FILE} ${OUTFILE} ${MAXDIM} ${XMIN} ${XMAX} ${YMIN} ${YMAX} ${ZMIN} ${ZMAX} ${POSITION}
done

# Collect the final frame filename 
OUTFILE="${DIR}/graphs/${BASEPREFIX}_final_complex_dim${MAXDIM}.jpg"
OUTFILES+=($OUTFILE)

# Stitch together the files 
python3 stitch_frames.py None ${OUTFILES[@]} "${PREFIX}_complex_dim${MAXDIM}.avi"


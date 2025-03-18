#!/bin/bash

#######################################################################
DIR=$1
NFRAMES=$2

# Get list of files to plot 
FILES=`python3 plot3d_filenames.py "${DIR}/*" ${NFRAMES}`

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

# Get axes limits from cells in the final file
XMIN=`python3 -c "import numpy as np; print(np.loadtxt('${FINAL}', delimiter='\t', comments='#')[:, 1].min())"`
XMAX=`python3 -c "import numpy as np; print(np.loadtxt('${FINAL}', delimiter='\t', comments='#')[:, 1].max())"`
YMIN=`python3 -c "import numpy as np; print(np.loadtxt('${FINAL}', delimiter='\t', comments='#')[:, 2].min())"`
YMAX=`python3 -c "import numpy as np; print(np.loadtxt('${FINAL}', delimiter='\t', comments='#')[:, 2].max())"`
ZMIN=`python3 -c "import numpy as np; print(np.loadtxt('${FINAL}', delimiter='\t', comments='#')[:, 3].min())"`
ZMAX=`python3 -c "import numpy as np; print(np.loadtxt('${FINAL}', delimiter='\t', comments='#')[:, 3].max())"`

# Plot final file with the given axes limits
OUTFILE="${FINAL%.*}_frame.jpg"
POSITION=`python3 plot3d_frame.py ${FINAL} ${OUTFILE} ${XMIN} ${XMAX} ${YMIN} ${YMAX} ${ZMIN} ${ZMAX} | tail -n 1`

# Plot each file with the given axes limits 
for FILE in $FILES
do
    OUTFILE="${FILE%.*}_frame.jpg"
    python3 plot3d_frame.py ${FILE} ${OUTFILE} ${XMIN} ${XMAX} ${YMIN} ${YMAX} ${ZMIN} ${ZMAX} ${POSITION}
done

# Stitch together the files 
python3 stitch.py $FILES 


#!/bin/bash
#$ -S /bin/bash
#$ -V
#$ -cwd
#$ -M j.davidgriffiths@gmail.com
#$ -m be

use freesurfer
#use fsl
use workbench

use ics  # should help with an intel mkl fatal irrer

export PATH="/home/hpc3230/Software/anaconda2/bin:$PATH"

source activate tvb # dipy_release

python jg_04_visualize_distance.py






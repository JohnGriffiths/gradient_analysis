#!/bin/bash
#$ -S /bin/bash
#$ -V
#$ -cwd
#$ -M j.davidgriffiths@gmail.com
#$ -m be

use freesurfer
#use fsl
use workbench

export PATH="/home/hpc3230/Software/anaconda2/bin:$PATH"

source activate tvb # dipy_release

python jg_02_embed_connectomes.py






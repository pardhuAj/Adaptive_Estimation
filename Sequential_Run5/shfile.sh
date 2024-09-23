#!/bin/bash
#
#PBS -N Sequential_55
#PBS -l select=1:ncpus=8:mem=23gb:interconnect=1g,walltime=50:00:00
#PBS -j oe



module add matlab/2023b

cd $PBS_O_WORKDIR
matlab -nodisplay -nosplash < EstDRLMain.m > RLEST5.txt

#!/bin/bash
#
#PBS -N Sequential_1
#PBS -l select=1:ncpus=28:mem=125gb:interconnect=fdr,walltime=50:00:00
#PBS -j oe

module add matlab/2023b

cd $PBS_O_WORKDIR
matlab -nodisplay -nosplash < EstDRLMain.m > RLEST0.txt

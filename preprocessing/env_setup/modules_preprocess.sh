#!/usr/bin/env bash

# __author__ = Michael Langguth
# __date__  = '2020_08_01'

# This script loads the required modules for the preprocessing of IFS HRES data in scope of the
# downscaling application in scope of the MAELSTROM project on Juwels and HDF-ML.
# Note that some other packages have to be installed into a venv (see create_env.sh and requirements_preprocess.txt).

SCR_NAME_MOD="modules_preprocess.sh"
HOST_NAME=`hostname`

echo "%${SCR_NAME_MOD}: Start loading modules on ${HOST_NAME} required for preprocessing IFS HRES data."

ml purge
ml use $OTHERSTAGES
ml Stages/2020

ml GCC/9.3.0
ml GCCcore/.9.3.0
ml ParaStationMPI/5.4.7-1
ml CDO/1.9.8
ml NCO/4.9.5
ml mpi4py/3.0.3-Python-3.8.5
ml SciPy-Stack/2020-Python-3.8.5
ml TensorFlow/2.3.1-Python-3.8.5

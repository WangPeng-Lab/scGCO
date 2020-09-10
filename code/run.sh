#!/bin/bash
set -ex
export MPLBACKEND=Agg 


# Reproduce main figure

for f in ./analysis/*.ipynb
do
    jupyter nbconvert \
        --ExecutePreprocessor.allow_errors=True \
        --ExecutePreprocessor.timeout=-1 \
        --FileWriter.build_directory=../../../results \
        --execute "$f";
done


#Analysis MouseOB

for f in ./analysis/MouseOB/*.ipynb
do
    jupyter nbconvert \
        --ExecutePreprocessor.allow_errors=True \
        --ExecutePreprocessor.timeout=-1 \
        --FileWriter.build_directory=../../../results \
        --execute "$f";
done


# Analysis Breast Cancer

for f in ./analysis/Breast_Cancer/*.ipynb
do
    jupyter nbconvert \
        --ExecutePreprocessor.allow_errors=True \
        --ExecutePreprocessor.timeout=-1 \
        --FileWriter.build_directory=../../../results \
        --execute "$f";
done


# Analysis MERFISH

for f in ./analysis/MERFISH/*.ipynb
do
    jupyter nbconvert \
        --ExecutePreprocessor.allow_errors=True \
        --ExecutePreprocessor.timeout=-1 \
        --FileWriter.build_directory=../../../results \
        --execute "$f";
done


# Benchmark of simulated data

##simulate_script will waster lots of time, so can select to run them.

for f in ./Simulation/*.ipynb
do
    jupyter nbconvert \
        --ExecutePreprocessor.allow_errors=True \
        --ExecutePreprocessor.timeout=-1 \
        --FileWriter.build_directory=../../../results \
        --execute "$f";
done

cd /
mkdir -p ./results/figures && mv ./results/*.pdf ./results/figures
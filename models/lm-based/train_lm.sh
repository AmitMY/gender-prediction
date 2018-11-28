#!/bin/bash
MODELDIR=$1
if [ -z $MTPATH ]
then
    echo "MT Path not set up"
    echo "Please export MTPATH"
    exit 1
fi

if [ $# -lt 1 ]
then
    echo "Two few input arguments"
    exit 1
fi
INFILE=$1

ORDER=3

echo " ----- STEP 1 ----- "
# Preprocess: tokenize, DO NOT lower-case
sh preprocess.sh $INFILE $INFILE.lc-tok $MTPATH

echo " ----- STEP 2 ----- "
# Build Language Model using KENLM #
$MTPATH/mosesdecoder/bin/lmplz -o $ORDER --prune 0 1 -S 50% -T /tmp < $INFILE.lc-tok > $INFILE.lm

echo " ----- STEP 3 ----- "
# Create binary LM #
$MTPATH/mosesdecoder/bin/build_binary $INFILE.lm $INFILE.blm

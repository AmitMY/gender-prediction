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

ORDER=6

echo " ----- STEP 1 ----- "
# Build Language Model using KENLM #
$MTPATH/mosesdecoder/bin/lmplz -o $ORDER -S 50% -T /tmp < $INFILE > $INFILE.lm

echo " ----- STEP 2 ----- "
# Create binary LM #
$MTPATH/mosesdecoder/bin/build_binary $INFILE.lm $INFILE.blm

#!/bin/bash
MODELDIR=$1
BASEPATH=$( dirname "$0" )
MTPATH=$BASEPATH/external
if [ $# -lt 1 ]
then
    echo "Too few input arguments"
    exit 1
fi
INFILE=$1

ORDER=$2

echo $INFILE
echo $ORDER

echo " ----- STEP 1 ----- "
# Preprocess: tokenize, DO NOT lower-case
sh $BASEPATH/preprocess.sh $INFILE $INFILE.lc-tok $MTPATH
#cp $INFILE $INFILE.lc-tok

echo " ----- STEP 2 ----- "
# Build Language Model using KENLM #
$MTPATH/lmplz -o $ORDER --prune 0 1 --discount_fallback 3 -S 20% -T /tmp < $INFILE.lc-tok > $INFILE.lm

echo " ----- STEP 3 ----- "
# Create binary LM #
$MTPATH/build_binary $INFILE.lm $INFILE.blm

#!/bin/bash
INFILE=$1
OUTFILE=$2

# PATH is kind of hardcoded

SRCLANG=nl

BASEPATH=$( dirname "$0" )
TOKENIZER=$BASEPATH/external

# remove @username
#perl -ne 's/[^ ]+@[^ ]+ / /g; print;' $INFILE > $INFILE.pp1
# remove #-tags
#perl -ne 's/[^ ]+#[^ ]+ / /g; print;' $INFILE.pp1 > $INFILE.pp2
# remove ' `

$TOKENIZER/tokenizer.perl -l $SRCLANG < $INFILE > $OUTFILE #| $TOKENIZER/lowercase.perl > $OUTFILE

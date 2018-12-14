#!/bin/bash
# PATH is kind of hardcoded
SENT=$1
SRCLANG=nl

BASEPATH=$( dirname "$0" )
TOKENIZER=$BASEPATH/external

# remove @username
#perl -ne 's/[^ ]+@[^ ]+ / /g; print;' $INFILE > $INFILE.pp1
# remove #-tags
#perl -ne 's/[^ ]+#[^ ]+ / /g; print;' $INFILE.pp1 > $INFILE.pp2
# remove ' `

echo $SENT | $TOKENIZER/tokenizer.perl -l $SRCLANG 2> /dev/null

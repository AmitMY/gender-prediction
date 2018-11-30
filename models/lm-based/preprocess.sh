#!/bin/bash
INFILE=$1
OUTFILE=$2

# PATH is kind of hardcoded

SRCLANG=nl

TOKENIZER=external

# remove @username
#perl -pe 's/(^| )@[^ ]+( |$)/ /g;' $INFILE > $INFILE.pp1
# remove #-tags
#perl -pe 's/(^| )#[^ ]+( |$)/ /g;' $INFILE.pp1 > $INFILE.pp2
# remove ' `

$TOKENIZER/tokenizer.perl -l $SRCLANG < $INFILE > $OUTFILE #| $TOKENIZER/lowercase.perl > $OUTFILE

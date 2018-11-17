#!/usr/bin/env bash

# I recommend using a new environment
# conda create -n gender python=3

# BERT
cd models
git clone https://github.com/google-research/bert.git
cd bert
wget http://storage.googleapis.com/bert_models/2018_11_03/multilingual_L-12_H-768_A-12.zip
unzip multilingual_L-12_H-768_A-12.zip

# Pytorch-Text-Classification
cd ..
git clone https://github.com/prakashpandey9/Text-Classification-Pytorch.git

# Spacy Dutch
conda install spacy
python -m spacy download nl_core_news_sm
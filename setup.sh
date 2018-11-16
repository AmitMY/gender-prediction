#!/usr/bin/env bash

# BERT
cd models
git clone https://github.com/google-research/bert.git
cd bert
wget http://storage.googleapis.com/bert_models/2018_11_03/multilingual_L-12_H-768_A-12.zip
unzip multilingual_L-12_H-768_A-12.zip
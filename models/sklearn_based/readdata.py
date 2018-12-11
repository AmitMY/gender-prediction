import xml.etree.ElementTree as ET
import sys
import argparse
import re
from sklearn.feature_extraction.text import CountVectorizer


def readData(training):

  ##### BUILD TRAINING SET ###################################
  # Load training text and training labels
  # (make sure to convert labels to integers (0 or 1, not '0' or '1')
  #  so that we can enforce the condition that label data is binary)

  with open(training, 'r') as f:
    xml = f.read()
    xml = re.sub('&','&amp;',xml)           # Delete pattern abc
    xml = re.sub("'",'&apos;',xml)           # Delete pattern abc
    xml = re.sub("<3",'hartje',xml)

    tree = ET.fromstring("<data>" + xml + "</data>")

    count=0

    all_labels=[]
    pos=[]
    all_text=[]

    for child in tree:
      if child.attrib['gender'] == "M":
        label=0
      elif child.attrib['gender']=="F":
        label=1
      all_labels.append(label)

      child.text=child.text.rstrip("\n").lstrip("\n")
      all_text.append(child.text)

      count+=1
    print(str(count))
  return all_labels, all_text




from functools import lru_cache
from itertools import chain
import random
from utils.file_system import listdir, savetodir, save_pair_todir
from regex import findall

import argparse

import os
import spacy
script_dir = os.path.dirname(__file__)

nlp = spacy.load("nl_core_news_sm")

from spacy.lang.nl import Dutch
tokenizer = Dutch().Defaults.create_tokenizer(nlp)

@lru_cache(maxsize=None)
def cached_tokenizer(text):
    return " ".join(map(str, tokenizer(text)))

class Data:
    def __init__(self, name, source, specific=None, tokenize=True):
        if isinstance(source, str):
            files = listdir(os.path.join(script_dir, source))
            self.categories = {f.split("_")[1].split(".")[0].lower(): self.parse_file(f, tokenize) for f in files}
        else:
            self.categories = source

        self.categories = {w: c for w, c in self.categories.items() if not specific or w in specific}

        print(name, "found", {w: len(c) for w, c in self.categories.items()})

    def parse_file(self, f, tokenize=False):
        raw = open(f, "r", encoding="utf-8").read()

        matches = findall('<doc id="(\d*?)" genre="(.*?)" gender="(M|F)">\n([\s\S]*?)\n<\/doc>', raw)

        data = list(map(lambda m: {"id": m[0], "genre": m[1], "gender": m[2],
                                   "text": m[3] if not tokenize else cached_tokenizer(m[3])}, matches))
        random.Random(1234).shuffle(data) # Same shuffle seed
        return data

    def split(self, percent=0.9):
        data1 = Data("train", {w: c[:int(len(c) * percent)] for w, c in self.categories.items()})
        data2 = Data("dev", {w: c[int(len(c) * percent):] for w, c in self.categories.items()})

        return data1, data2

    def export(self):
        #pairs = [(w + ": " + c["text"], 0 if c["gender"] == "M" else 1)
        pairs = [(c["text"], 0 if c["gender"] == "M" else 1)
                 for w, co in self.categories.items() for c in co]

        return list(zip(*pairs))  # List of texts, list of tags

def main():
    data_choices = {'a':'All',
                    'y':'YouTube',
                    't':'Twitter',
                    'n':'News'}

    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-d', '--data-dir', required=False, default='train', help='the folder containing a data set (to be split).')
    parser.add_argument('-o', '--output-dir', required=False, default='train', help='the folder where splits will be saved.')
    parser.add_argument('-t', '--data-type', required=False, default='a', choices=data_choices.keys(), help='the type of data [a = All, y = YouTube, t = Twitter, n = News].')

    args = parser.parse_args()

    data = Data(data_choices[args.data_type], args.data_dir, tokenize=True)
    train, dev = data.split()

    # TODO: make a class for this
    texts = {}
    categories = {}

    texts['train'], categories['train'] = train.export()
    texts['dev'], categories['dev'] = dev.export()

    for data_split in texts:
        print(str(len(texts['train'])))
        print(str(len(categories['train'])))
        save_pair_todir(args.output_dir, texts[data_split], categories[data_split], data_split + '.dat', data_split + '.lbl')


if __name__ == "__main__":
    main()

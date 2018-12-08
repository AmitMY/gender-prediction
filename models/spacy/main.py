#!/usr/bin/env python
# coding: utf8
"""Train a convolutional neural network text classifier on the
IMDB dataset, using the TextCategorizer component. The dataset will be loaded
automatically via Thinc's built-in dataset loader. The model is added to
spacy.pipeline, and predictions are available via `doc.cats`. For more details,
see the documentation:
* Training: https://spacy.io/usage/training
Compatible with: spaCy v2.0.0+
"""
from __future__ import unicode_literals, print_function

import spacy
from spacy.util import minibatch, compounding
from tqdm import tqdm

from data.reader import Data


class ModelRunner:
    def __init__(self, model, train, dev, opt={}):
        self.nlp = spacy.load(model)

        self.train_set = train
        self.dev_set = dev
        self.opt = opt

    def load(self, path):
        self.nlp = spacy.load(path)

    def save(self, path):
        self.nlp.to_disk(path)

    def cat(self, test):
        (texts, _) = test.export()

        docs = (self.nlp(text) for text in texts)

        for doc in self.nlp.get_pipe("textcat").pipe(docs):
            print(doc.cats["g"])
            print(doc)

    def train(self):
        # add the text classifier to the pipeline if it doesn't exist
        # nlp.create_pipe works for built-ins that are registered with spaCy
        if 'textcat' not in self.nlp.pipe_names:
            textcat = self.nlp.create_pipe('textcat')
            self.nlp.add_pipe(textcat, last=True)
        # otherwise, get it, so we can add labels to it
        else:
            textcat = self.nlp.get_pipe('textcat')

        # add label to text classifier
        textcat.add_label('g')

        # load the dataset
        print("Exporting data to correct format")
        (train_texts, train_cats) = self.train_set.export(lowercase=self.opt["lowercase"], prefix=self.opt["prefix"])
        (dev_texts, dev_cats) = self.dev_set.export(lowercase=self.opt["lowercase"], prefix=self.opt["prefix"])

        train_cats = [{"g": c} for c in train_cats]
        dev_cats = [{"g": c} for c in dev_cats]

        n_texts = len(train_texts)

        print("Using {} examples ({} training, {} evaluation)".format(n_texts, len(train_texts), len(dev_texts)))
        train_data = list(zip(train_texts, [{'cats': cats} for cats in train_cats]))

        # get names of other pipes to disable them during training
        other_pipes = [pipe for pipe in self.nlp.pipe_names if pipe != 'textcat']
        with self.nlp.disable_pipes(*other_pipes):  # only train textcat
            optimizer = self.nlp.begin_training()
            print("Training the model...")
            print('{:^5}\t{:^5}\t{:^5}\t{:^5}\t{:^5}'.format('LOSS', 'A', 'P', 'R', 'F'))
            while True:
                losses = {}
                # batch up the examples using spaCy's minibatch
                batches = minibatch(train_data, size=compounding(4., 32., 1.001))
                for batch in tqdm(list(batches)):
                    texts, annotations = zip(*batch)
                    self.nlp.update(texts, annotations, sgd=optimizer, drop=0.2,
                                    losses=losses)
                with textcat.model.use_params(optimizer.averages):
                    # evaluate on the dev data split off in load_data()
                    scores = evaluate(self.nlp, textcat, dev_texts, dev_cats)

                    yield scores['acc'] * 100
                print('{0:.3f}\t{1:.3f}\t{1:.3f}\t{2:.3f}\t{3:.3f}'  # print a simple table
                      .format(losses['textcat'], scores['acc'], scores['textcat_p'],
                              scores['textcat_r'], scores['textcat_f']))


def evaluate(nlp, textcat, texts, cats):
    docs = (nlp(text) for text in texts)
    tp = 1e-8  # True positives
    fp = 1e-8  # False positives
    fn = 1e-8  # False negatives
    tn = 1e-8  # True negatives

    correct = 0

    for gold, doc in zip(cats, textcat.pipe(docs)):
        for label, score in doc.cats.items():
            if label not in gold:
                continue

            if round(score) == gold[label]:
                correct += 1

            if score >= 0.5 and gold[label] >= 0.5:
                tp += 1.
            elif score >= 0.5 and gold[label] < 0.5:
                fp += 1.
            elif score < 0.5 and gold[label] < 0.5:
                tn += 1
            elif score < 0.5 and gold[label] >= 0.5:
                fn += 1

    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f_score = 2 * (precision * recall) / (precision + recall)
    return {'textcat_p': precision, 'textcat_r': recall, 'textcat_f': f_score, 'acc': float(correct) / len(cats)}


if __name__ == '__main__':
    train, dev = Data("All", "train", tokenize=False).split()
    test = Data("All", "test", tokenize=False)
    # train = Data("Train", "train", ["twitter", "youtube"])
    # dev = Data("Dev", "train", ["news"])

    print("\n")

    inst = ModelRunner(model="nl_core_news_sm", train=train, dev=dev)
    inst.load("../checkpoints/Spacy/")
    inst.cat(test)

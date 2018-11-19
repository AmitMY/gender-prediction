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
import plac
import random
from pathlib import Path

import spacy
from spacy.util import minibatch, compounding
from tqdm import tqdm

from data.reader import Data


def main(model, train, dev, output_dir=None, n_iter=20):
    nlp = spacy.load(model)  # load existing spaCy model
    print("Loaded model '%s'" % model)

    # add the text classifier to the pipeline if it doesn't exist
    # nlp.create_pipe works for built-ins that are registered with spaCy
    if 'textcat' not in nlp.pipe_names:
        textcat = nlp.create_pipe('textcat')
        nlp.add_pipe(textcat, last=True)
    # otherwise, get it, so we can add labels to it
    else:
        textcat = nlp.get_pipe('textcat')

    # add label to text classifier
    textcat.add_label('g')

    # load the dataset
    print("Exporting data to correct format")
    (train_texts, train_cats) = train.export()
    (dev_texts, dev_cats) = dev.export()

    train_cats = [{"g": c} for c in train_cats]
    dev_cats = [{"g": c} for c in dev_cats]

    n_texts = len(train_texts)

    print("Using {} examples ({} training, {} evaluation)".format(n_texts, len(train_texts), len(dev_texts)))
    train_data = list(zip(train_texts, [{'cats': cats} for cats in train_cats]))

    # get names of other pipes to disable them during training
    other_pipes = [pipe for pipe in nlp.pipe_names if pipe != 'textcat']
    with nlp.disable_pipes(*other_pipes):  # only train textcat
        optimizer = nlp.begin_training()
        print("Training the model...")
        print('{:^5}\t{:^5}\t{:^5}\t{:^5}\t{:^5}'.format('LOSS', 'A', 'P', 'R', 'F'))
        for i in range(n_iter):
            losses = {}
            # batch up the examples using spaCy's minibatch
            batches = minibatch(train_data, size=compounding(4., 32., 1.001))
            for batch in tqdm(list(batches)):
                texts, annotations = zip(*batch)
                nlp.update(texts, annotations, sgd=optimizer, drop=0.2,
                           losses=losses)
            with textcat.model.use_params(optimizer.averages):
                # evaluate on the dev data split off in load_data()
                scores = evaluate(nlp.tokenizer, textcat, dev_texts, dev_cats)

                yield scores['acc']
            print('{0:.3f}\t{1:.3f}\t{1:.3f}\t{2:.3f}\t{3:.3f}'  # print a simple table
                  .format(losses['textcat'], scores['acc'], scores['textcat_p'],
                          scores['textcat_r'], scores['textcat_f']))

    # test the trained model
    test_text = "This movie sucked"
    doc = nlp(test_text)
    print(test_text, doc.cats)

    if output_dir is not None:
        output_dir = Path(output_dir)
        if not output_dir.exists():
            output_dir.mkdir()
        nlp.to_disk(output_dir)
        print("Saved model to", output_dir)

        # test the saved model
        print("Loading from", output_dir)
        nlp2 = spacy.load(output_dir)
        doc2 = nlp2(test_text)
        print(test_text, doc2.cats)


def evaluate(tokenizer, textcat, texts, cats):
    docs = (tokenizer(text) for text in texts)
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
    train, dev = Data("All", "train", tokenize=True).split()
    # train = Data("Train", "train", ["twitter", "youtube"])
    # dev = Data("Dev", "train", ["news"])

    print("\n")

    main(model="nl_core_news_sm", train=train, dev=dev, n_iter=50)

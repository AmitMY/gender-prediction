from torchtext import data
from torchtext import datasets

from data.pytorch_dataset import GxG
from data.reader import Data


def load_dataset(train, dev):
    tokenize = lambda x: x.split()
    TEXT = data.Field(sequential=True, tokenize=tokenize, lower=True, include_lengths=True, batch_first=True)
    LABEL = data.LabelField()

    train_data = GxG(list(zip(*train.export())), TEXT, LABEL)
    dev_data = GxG(list(zip(*dev.export())), TEXT, LABEL)
    TEXT.build_vocab(train_data)
    LABEL.build_vocab(train_data)

    print(train_data)

    print("Length of Text Vocabulary: " + str(len(TEXT.vocab)))
    print("Label Length: " + str(len(LABEL.vocab)))

    train_iter = train_data.iters(batch_size=32)
    dev_iter = dev_data.iters(batch_size=32)

    vocab_size = len(TEXT.vocab)

    return TEXT, vocab_size, train_iter, dev_iter


if __name__ == "__main__":
    train, dev = Data("All", "train", tokenize=False).split()

    load_dataset(train, dev)

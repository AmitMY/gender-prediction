import os
import glob
import io

from torchtext import data


class GxG(data.Dataset):
    @staticmethod
    def sort_key(ex):
        return len(ex.text)

    def __init__(self, results, text_field, label_field, **kwargs):
        fields = [('text', text_field), ('label', label_field)]
        examples = [data.Example.fromlist([text, label], fields) for (text, label) in results]

        super(GxG, self).__init__(examples, fields, **kwargs)

    def iters(self, batch_size=32, device=None):
        """Create iterator objects for splits of the SST dataset.
        Arguments:
            batch_size: Batch_size
            device: Device to create batches on. Use - 1 for CPU and None for
                the currently active GPU device.
            root: The root directory that the dataset's zip archive will be
                expanded into; therefore the directory in whose trees
                subdirectory the data files will be stored.
            vectors: one of the available pretrained vectors or a list with each
                element one of the available pretrained vectors (see Vocab.load_vectors)
            Remaining keyword arguments: Passed to the splits method.
        """
        TEXT = data.Field()
        LABEL = data.Field(sequential=False)

        TEXT.build_vocab(self)
        LABEL.build_vocab(self)

        return data.BucketIterator(dataset=self, batch_size=batch_size, device=device, sort_key=GxG.sort_key)

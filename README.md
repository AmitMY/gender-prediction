# Gender Prediction from Dutch

## Adding a model
After writing some model, please create the following class:

```python
from data.reader import Data

class ModelRunner:
    def __init__(self, model, train, dev, opt={}):
        self.train_set = train
        self.dev_set = dev
        self.opt = opt
        
        # Initialize your model
        # If you have some options for your model, you will get them in opt.
        

    def load(self, path):
        # If possible, load your model from a file
        pass

    def save(self, path):
        # If possible, save your model to a file
        pass
    
    def train(self):
        # Here you should have the main training loop.
        while True:
            # Train an epoch, or something
            result = self.eval(self.dev_set) # A number between 0 and 100
            yield result
        
        # If your model does not include a training loop, and just returns a result:
        result = self.eval(self.dev_set)
        return [result]

# To test if your model runs at all
if __name__ == '__main__':
    train, dev = Data("All", "train", tokenize=False).split() # Tokenize=False is just faster, but less accurate

    inst = ModelRunner(model="some name or parameter", train=train, dev=dev, opt={})
    first_epoch = next(iter(inst.train())) # Returns first result
    inst.save("checkpoint") # Make sure this doesn't error
    inst.load("checkpoint") # Make sure this doesn't error
```

Once it is implemented, 

### Experiments
- Model.X means a new seed for the model
- Model+ means it includes frozen fasttext embeddings


#UNIGRAMS

|TRAIN|DEV|SVM|LOG|RF|
| --- | - | - | - | -|
|Twitter 90%|Twitter 10%|58%|58%|60%|
|YouTube 90%|YouTube 10%|57%|59%|58%|
|News 90%|News 10%|59%|57%|56%|
| -------| ------ | - | - | - |
|Twitter, News|YouTube|53%|53%|51%|
|YouTube, News|News|53%|53%|52%|
|Twitter, Youtube|Twitter|52%|54%|51%|


#POS

|TRAIN|DEV|SVM|LOG|RF|
| --- | - | - | - | -|
|Twitter 90%|Twitter 10%|44%|43%|45%|
|YouTube 90%|YouTube 10%|58%|58%|55%|
|News 90%|News 10%|18%|18%|16%|
| -------| ------ | - | - | - |
|Twitter, News|YouTube|50%|R|R|
|YouTube, News|Twitter|51%|R|R|
|Twitter, Youtube|News|50%|R|R|

#UNIGRAMS + POS

|TRAIN |DEV|SVM|LOG|RF|
| ---- | - | - | - | -|
|Twitter 90%|Twitter 10%|59%|58%|51%|
|YouTube 90%|YouTube 10%|56%|61%|56%|
|News 90%|News 10%|59%|58%|53%|
| -------| ------ | - | - | - |
|Twitter, News|YouTube|52%|52%|RUN|
|YouTube, News|Twitter|52%|52%|RUN|
|Twitter, Youtube|News|RUN|52%|RUN|



| Model | Twitter 90% Twitter 10% | YouTube 90% YouTube 10% | YouTube, News Twitter | Twitter, News YouTube | Twitter, YouTube News | News 90% News 10% |
| ----- | ----------------------- | ----------------------- | --------------------- | --------------------- | --------------------- | ----------------- |
| CNN.n+.0 | 62.46 | 59.70 | 53.55 | 52.91 | 51.79 | 59.50 |
| CNN.n.0 | 60.57 | 58.87 | 53.72 | 52.78 | 50.74 | 54.67 |
| CNN.y+.0 | 63.79 | 59.45 | 54.04 | 53.69 | 52.03 | 61.17 |
| CNN.y.0 | 59.02 | 57.43 | 51.36 | 53.82 | 51.90 | 56.83 |
| KENLM.n.0.0 | 62.42 | 57.88 | 53.32 | 48.70 | 52.13 | 61.96 |
| KENLM.n.1.0 | 62.42 | 57.69 | 53.11 | 48.70 | 52.21 | 60.33 |
| KENLM.n.1.1 | 54.61 | 58.76 | 53.41 | 49.06 | 52.21 | 60.87 |
| KENLM.n.2.0 | 61.98 | 58.72 | 53.19 | 49.00 | 52.05 | 64.13 |
| KENLM.n.2.1 | 61.98 | 58.72 | 53.19 | 49.00 | 52.05 | 64.13 |
| KENLM.n.3.0 | 62.34 | 58.49 | 53.11 | 49.40 | 52.21 | 61.96 |
| KENLM.n.3.1 | 62.34 | 58.49 | 53.11 | 49.40 | 52.21 | 61.96 |
| KENLM.n.4.0 | 62.42 | 57.69 | 53.27 | 49.36 | 52.21 | 61.41 |
| KENLM.n.4.1 | 62.42 | 57.69 | 53.27 | 49.36 | 52.21 | 61.41 |
| KENLM.n.5.0 | 62.42 | 57.50 | 53.32 | 49.38 | 52.13 | 60.60 |
| KENLM.n.5.1 | 62.42 | 57.50 | 53.32 | 49.38 | 52.13 | 60.60 |
| KENLM.n.6.0 | 62.42 | 57.61 | 53.32 | 49.38 | 52.13 | 60.87 |
| KENLM.n.6.1 | 62.42 | 57.61 | 53.32 | 49.38 | 52.13 | 60.87 |
| KENLM.n.7.0 | 62.42 | 57.69 | 53.32 | 48.70 | 50.68 | 61.41 |
| KENLM.n.7.1 | 62.42 | 57.88 | 53.27 | 49.38 | 52.13 | 60.87 |
| KENLM.n.8.0 | 62.42 | 57.88 | 53.41 | 49.40 | 52.21 | 61.96 |
| KENLM.n.8.1 | 54.61 | 57.27 | 53.11 | 49.36 | 50.66 | 60.33 |
| KENLM.n.9.0 | 62.42 | 57.88 | 53.41 | 48.70 | 52.21 | 61.96 |
| KENLM.n.9.1 | 54.61 | 57.69 | 53.27 | 49.06 | 52.13 | 61.68 |
| KENLM.y.0.0 | 62.51 | 58.38 | 53.58 | 49.72 | 51.58 | 62.77 |
| KENLM.y.1.0 | 62.51 | 58.38 | 53.64 | 49.19 | 51.45 | 60.87 |
| KENLM.y.1.1 | 59.32 | 51.61 | 53.52 | 49.14 | 51.45 | 52.45 |
| KENLM.y.2.0 | 62.67 | 60.10 | 53.65 | 49.08 | 51.61 | 64.95 |
| KENLM.y.2.1 | 62.67 | 60.10 | 53.65 | 49.08 | 51.61 | 64.95 |
| KENLM.y.3.0 | 61.82 | 59.30 | 53.64 | 49.69 | 51.50 | 62.77 |
| KENLM.y.3.1 | 61.82 | 59.30 | 53.64 | 49.69 | 51.50 | 62.77 |
| KENLM.y.4.0 | 62.46 | 59.26 | 53.63 | 49.72 | 51.56 | 63.04 |
| KENLM.y.4.1 | 62.46 | 59.26 | 53.63 | 49.72 | 51.56 | 63.04 |
| KENLM.y.5.0 | 62.51 | 58.95 | 53.52 | 49.76 | 51.45 | 62.50 |
| KENLM.y.5.1 | 62.51 | 58.95 | 53.52 | 49.76 | 51.45 | 62.50 |
| KENLM.y.6.0 | 62.42 | 59.03 | 53.60 | 49.77 | 51.45 | 62.77 |
| KENLM.y.6.1 | 62.42 | 59.03 | 53.60 | 49.77 | 51.45 | 62.77 |
| KENLM.y.7.0 | 59.61 | 58.38 | 53.65 | 49.08 | 51.61 | 60.87 |
| KENLM.y.7.1 | 59.61 | 57.92 | 53.60 | 49.72 | 51.53 | 62.77 |
| KENLM.y.8.0 | 59.73 | 59.30 | 53.68 | 49.19 | 51.56 | 60.87 |
| KENLM.y.8.1 | 62.51 | 57.80 | 53.52 | 49.69 | 51.53 | 60.87 |
| KENLM.y.9.0 | 59.61 | 59.30 | 53.65 | 49.08 | 51.45 | 60.87 |
| KENLM.y.9.1 | 59.32 | 58.95 | 53.68 | 49.19 | 51.45 | 60.87 |
| LSTM.n+.0 | 63.16 | 51.64 | 50.01 | 49.63 | 53.38 | 43.00 |
| LSTM.n.0 | 59.89 | 58.98 | 52.85 | 51.56 | 52.55 | 45.17 |
| LSTM.y+.0 | 62.75 | 51.68 | 50.27 | 49.79 | 53.62 | 45.17 |
| LSTM.y.0 | 56.43 | 58.26 | 51.00 | 53.49 | 51.33 | 43.00 |
| LSTMAttention.n+.0 | 63.10 | 59.62 | __**56.48**__ | 53.24 | 53.16 | 55.33 |
| LSTMAttention.n.0 | 61.68 | 58.09 | 52.97 | 54.83 | 51.41 | 55.33 |
| LSTMAttention.y+.0 | 63.43 | 57.77 | 55.87 | 53.65 | 53.50 | 58.00 |
| LSTMAttention.y.0 | 60.21 | 58.38 | 52.84 | 53.90 | 51.17 | 55.33 |
| RCNN.n+.0 | 62.98 | 58.83 | 54.88 | 54.07 | 52.83 | 60.67 |
| RCNN.n.0 | 61.89 | 59.43 | 53.72 | 54.10 | 53.19 | 58.00 |
| RCNN.y+.0 | 61.63 | 59.04 | 55.30 | 54.63 | 53.64 | 58.83 |
| RCNN.y.0 | 61.32 | 59.19 | 54.69 | 52.21 | __**54.69**__ | 58.33 |
| RNN.n+.0 | 55.62 | 54.00 | 53.39 | 52.26 | 52.64 | 45.67 |
| RNN.n.0 | 56.43 | 57.53 | 50.56 | 53.03 | 52.97 | 51.67 |
| RNN.y+.0 | 56.62 | 53.23 | 52.99 | 53.36 | 50.40 | 47.67 |
| RNN.y.0 | 60.24 | 58.13 | 49.56 | 54.56 | 52.31 | 46.67 |
| SKLearn-knn.n.0 | 52.35 | 54.17 | 50.07 | 50.60 | 52.18 | 53.80 |
| SKLearn-knn.y.0 | 51.60 | 53.83 | 50.71 | 51.04 | 50.93 | 50.00 |
| SKLearn-log.n.0 | 62.70 | 58.98 | 53.36 | 53.14 | 52.89 | 61.96 |
| SKLearn-log.y.0 | 63.00 | 59.25 | 53.93 | 53.21 | 53.60 | 65.22 |
| SKLearn-nb.n.0 | 64.50 | 58.44 | 54.43 | 51.24 | 52.46 | 62.50 |
| SKLearn-nb.y.0 | 64.60 | 59.93 | 54.67 | 51.40 | 51.53 | 64.13 |
| SKLearn-rf.n.0 | 57.15 | 55.80 | 50.97 | 51.70 | 50.60 | 52.72 |
| SKLearn-rf.y.0 | 57.80 | 54.58 | 51.83 | 52.78 | 50.11 | 64.67 |
| SKLearn-svm.n.0 | 62.80 | 59.66 | 51.84 | 52.22 | 50.98 | 63.04 |
| SKLearn-svm.y.0 | 61.05 | 58.85 | 51.93 | 52.98 | 52.73 | 65.22 |
| SelfAttention.n+.0 | 62.17 | 58.81 | 55.86 | 54.16 | 53.74 | 62.17 |
| SelfAttention.n.0 | 61.81 | 57.83 | 53.68 | 53.82 | 51.76 | 51.17 |
| SelfAttention.y+.0 | 63.29 | 57.94 | 55.36 | 54.57 | 52.10 | 61.67 |
| SelfAttention.y.0 | 61.03 | 57.38 | 53.75 | __**56.16**__ | 50.33 | 50.17 |
| Spacy.n.0 | 66.65 | __**62.92**__ | 55.79 | 54.22 | 53.60 | __**72.28**__ |
| Spacy.y.0 | __**66.90**__ | __**62.92**__ | 55.54 | 54.52 | 53.77 | 70.11 |
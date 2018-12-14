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
        
    def eval_one(self, text):
        # Optional. returns a single number
        pass
        
    def eval_all(self, texts):
        # Optional. returns an array of number (ordered in texts order)
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
- Model-c means it is converted to clusters


| Model | Twitter, YouTube News | Twitter 90% Twitter 10% | News 90% News 10% | YouTube, News Twitter | Twitter, News YouTube | YouTube 90% YouTube 10% | Twitter 90%, Twisty Twitter 10% | Twitter, News, External YouTube | Twitter, YouTube, External News | YouTube, News, CSI Twitter | All 90% All 10% |
| ----- | --------------------- | ----------------------- | ----------------- | --------------------- | --------------------- | ----------------------- | ------------------------------- | ------------------------------- | ------------------------------- | -------------------------- | --------------- |
| CNN+.0 | 52.84 | 63.03 | 57.00 | 55.49 | 53.34 | 59.96 | ? | ? | ? | ? | ? |
| CNN+.1 | 51.98 | 62.79 | 58.50 | 54.90 | 53.14 | 60.04 | ? | ? | ? | ? | ? |
| CNN-c.0 | 50.78 | 55.41 | 56.33 | 52.65 | 51.56 | 54.15 | ? | ? | ? | ? | ? |
| CNN-c.1 | 50.50 | 57.90 | 54.33 | 52.87 | 51.46 | 54.62 | ? | ? | ? | ? | ? |
| CNN.0 | 49.38 | 62.22 | 53.50 | 54.24 | 53.76 | 58.55 | ? | ? | ? | ? | ? |
| CNN.1 | 49.91 | 60.76 | 55.50 | 53.73 | __**55.58**__ | 57.57 | ? | ? | ? | ? | ? |
| KENLM.3.0 | 51.50 | 61.82 | 62.77 | 53.64 | 49.69 | 59.30 | 80.28 | 53.99 | 54.39 | 52.20 | ? |
| KENLM.3.1 | 51.50 | 61.82 | 62.77 | 53.64 | 49.69 | 59.30 | 80.28 | 53.99 | 54.39 | ? | ? |
| KENLM.4.0 | 51.56 | 62.46 | 63.04 | 53.63 | 49.72 | 59.26 | 80.20 | ? | 54.04 | 52.11 | ? |
| KENLM.4.1 | 51.56 | 62.46 | 63.04 | 53.63 | 49.72 | 59.26 | 80.20 | 53.95 | 54.04 | 52.11 | ? |
| KENLM.5.0 | 51.45 | 62.51 | 62.50 | 53.52 | 49.76 | 58.95 | 80.22 | 53.92 | 54.26 | ? | ? |
| KENLM.5.1 | 51.45 | 62.51 | 62.50 | 53.52 | 49.76 | 58.95 | 80.22 | 53.92 | 54.26 | ? | ? |
| KENLM.6.0 | 51.45 | 62.42 | 62.77 | 53.60 | 49.77 | 59.03 | 80.23 | ? | ? | ? | ? |
| KENLM.6.1 | 51.45 | 62.42 | 62.77 | 53.60 | 49.77 | 59.03 | 80.23 | ? | ? | ? | ? |
| LSTM+.0 | 49.28 | 63.02 | 46.50 | 49.82 | 49.61 | 51.51 | 71.77 | 53.56 | 54.83 | 49.90 | ? |
| LSTM+.1 | 52.83 | 62.92 | 46.17 | 49.74 | 49.69 | 51.34 | 71.79 | 53.81 | 53.98 | 49.77 | ? |
| LSTM-c.0 | 49.84 | 61.25 | 46.50 | 49.85 | 51.36 | 58.81 | 66.51 | 51.82 | 53.66 | 51.56 | ? |
| LSTM-c.1 | 51.98 | 59.87 | 47.00 | 49.87 | 51.77 | 56.60 | 66.16 | 53.38 | 51.78 | 52.12 | ? |
| LSTM.0 | 51.34 | 62.51 | 46.17 | 52.73 | 52.55 | 56.53 | 79.90 | 54.46 | ? | 52.09 | ? |
| LSTM.1 | 52.71 | 59.75 | 46.17 | 53.63 | 54.83 | 59.00 | 80.13 | 51.83 | 53.00 | 49.62 | ? |
| LSTMAttention+.0 | __**54.83**__ | 63.65 | 55.83 | 55.71 | 53.34 | 58.66 | 72.22 | 53.20 | 53.62 | 54.35 | ? |
| LSTMAttention+.1 | 53.41 | 63.08 | 56.83 | 55.73 | 53.31 | 59.62 | 72.06 | 53.79 | 53.12 | 54.10 | ? |
| LSTMAttention-c.0 | 51.33 | 61.56 | 56.00 | 53.52 | 52.61 | 58.23 | 67.40 | 52.06 | 52.12 | 53.76 | ? |
| LSTMAttention-c.1 | 52.41 | 61.29 | 57.33 | 53.26 | 52.99 | 59.04 | 67.41 | 52.44 | 53.62 | 52.95 | ? |
| LSTMAttention.0 | 51.78 | 60.98 | 55.33 | 53.08 | 53.66 | 58.94 | 80.17 | 52.56 | ? | 51.84 | ? |
| LSTMAttention.1 | 51.38 | 62.19 | 46.17 | 53.31 | 51.80 | 59.06 | 80.91 | 53.62 | 52.91 | 53.37 | ? |
| RCNN+.0 | 51.83 | 62.84 | 58.33 | 54.55 | 53.97 | 59.28 | 72.47 | 54.11 | 53.09 | 55.29 | ? |
| RCNN+.1 | 54.36 | 62.60 | 60.50 | 54.65 | 53.55 | 58.94 | 72.67 | 54.47 | 53.47 | 55.31 | ? |
| RCNN-c.0 | 50.81 | 60.27 | 55.33 | 52.84 | 53.34 | 58.70 | 67.81 | 52.72 | ? | 53.31 | ? |
| RCNN-c.1 | 51.66 | 60.24 | 58.50 | 52.59 | 52.82 | 58.81 | 67.14 | 51.95 | 53.48 | 53.25 | ? |
| RCNN.0 | 51.95 | 62.83 | 56.83 | 53.26 | 53.95 | 59.77 | 80.04 | 51.69 | ? | 53.40 | ? |
| RCNN.1 | 52.76 | 62.79 | 58.17 | 54.28 | 55.00 | 59.64 | 80.54 | 52.44 | 51.50 | 52.11 | ? |
| RNN+.0 | 50.53 | 53.22 | 48.17 | 50.70 | 50.40 | 52.57 | 64.77 | 52.05 | ? | 51.22 | ? |
| RNN+.1 | 50.48 | 61.05 | 48.67 | 51.26 | 50.91 | 53.34 | 63.38 | 51.86 | 51.07 | 53.17 | ? |
| RNN-c.0 | 50.66 | 58.75 | 49.17 | 52.65 | 52.17 | 55.83 | 65.16 | 53.20 | 51.60 | 50.89 | ? |
| RNN-c.1 | 52.17 | 59.27 | 48.17 | 52.70 | 51.89 | 53.83 | 65.33 | 53.41 | ? | 51.43 | ? |
| RNN.0 | 52.69 | 57.13 | 48.50 | 52.23 | 53.66 | 57.66 | 80.29 | 54.74 | 53.33 | 52.11 | ? |
| RNN.1 | 53.53 | 57.10 | 45.50 | 50.06 | 55.29 | 56.87 | 80.55 | 55.00 | 51.09 | 52.89 | ? |
| SKLearn-knn-c.0 | 51.91 | 53.50 | 57.61 | 51.40 | 50.48 | 51.93 | 59.42 | 50.66 | 49.02 | 51.11 | 57.53 |
| SKLearn-knn-c.1 | 51.91 | 53.50 | 57.61 | 51.40 | 50.48 | 51.93 | 59.42 | 50.66 | 49.02 | 51.11 | 57.53 |
| SKLearn-knn.0 | 50.93 | 51.60 | 50.00 | 50.71 | 51.04 | 53.83 | 62.00 | 50.33 | 50.16 | 50.36 | 60.19 |
| SKLearn-knn.1 | 50.93 | 51.60 | 50.00 | 50.71 | 51.04 | 53.83 | 62.00 | 50.33 | 50.16 | 50.36 | 60.19 |
| SKLearn-log-c.0 | 52.62 | 62.10 | 66.85 | 54.18 | 53.49 | 58.85 | 66.07 | 52.52 | 52.95 | 54.67 | 65.03 |
| SKLearn-log-c.1 | 52.62 | 62.10 | 66.85 | 54.18 | 53.49 | 58.85 | 66.07 | 52.52 | 52.95 | 54.67 | 65.03 |
| SKLearn-log.0 | 53.60 | 62.95 | 64.67 | 53.92 | 53.23 | 59.25 | 77.29 | 52.17 | 52.02 | 53.93 | 74.82 |
| SKLearn-log.1 | 53.60 | 62.95 | 64.67 | 53.92 | 53.23 | 59.25 | 77.29 | 52.17 | 52.02 | 53.93 | 74.82 |
| SKLearn-nb-c.0 | 51.69 | 63.45 | 60.87 | 55.40 | 52.54 | 56.95 | 65.76 | 54.33 | 51.75 | 49.60 | 63.24 |
| SKLearn-nb-c.1 | 51.69 | 63.45 | 60.87 | 55.40 | 52.54 | 56.95 | 65.76 | 54.33 | 51.75 | 49.60 | 63.24 |
| SKLearn-nb.0 | 51.53 | 64.60 | 64.13 | 54.67 | 51.40 | 59.93 | 78.62 | 52.77 | 51.36 | 49.98 | __**77.81**__ |
| SKLearn-nb.1 | 51.53 | 64.60 | 64.13 | 54.67 | 51.40 | 59.93 | 78.62 | 52.77 | 51.36 | 49.98 | __**77.81**__ |
| SKLearn-rf-c.0 | 50.55 | 57.40 | 63.04 | 52.04 | 51.47 | 54.03 | ? | 52.40 | 49.40 | 51.42 | 62.44 |
| SKLearn-rf-c.1 | 50.55 | 56.25 | 58.70 | 52.15 | 50.90 | 57.83 | ? | 52.61 | 49.78 | 51.32 | 61.66 |
| SKLearn-rf.0 | 52.29 | 57.30 | 59.78 | 51.31 | 52.28 | 56.14 | ? | 51.34 | 52.07 | 51.51 | 70.25 |
| SKLearn-rf.1 | 50.55 | 58.10 | 59.78 | 51.55 | 52.60 | 54.58 | ? | ? | 51.36 | 51.00 | 69.66 |
| SKLearn-svm-c.0 | 52.67 | 60.25 | 62.50 | 53.57 | 53.48 | 58.44 | 66.00 | 53.12 | 52.62 | 53.84 | 65.21 |
| SKLearn-svm-c.1 | 52.67 | 60.25 | 62.50 | 53.57 | 53.48 | 58.51 | 66.00 | 53.16 | 52.57 | 53.77 | 65.21 |
| SKLearn-svm.0 | 52.73 | 61.05 | 65.22 | 51.93 | 52.98 | 58.85 | 80.51 | 52.18 | 51.97 | 52.20 | 77.25 |
| SKLearn-svm.1 | 52.73 | 61.05 | 65.22 | 51.94 | 52.98 | 58.85 | 80.51 | 52.18 | 51.97 | 52.20 | 77.25 |
| SelfAttention+.0 | 53.10 | 63.92 | 59.50 | 54.72 | 53.70 | 58.00 | 72.30 | 53.99 | 53.84 | 54.39 | ? |
| SelfAttention+.1 | 52.74 | 62.60 | 59.50 | 54.73 | 53.56 | 58.87 | 72.01 | 54.07 | 52.91 | 54.50 | ? |
| SelfAttention-c.0 | 51.76 | 61.59 | 55.50 | 53.97 | 52.64 | 58.17 | 66.46 | 52.06 | 53.52 | 53.15 | ? |
| SelfAttention-c.1 | 52.02 | 59.16 | 59.00 | 53.46 | 53.17 | 57.21 | 66.67 | 53.39 | 53.17 | 53.44 | ? |
| SelfAttention.0 | 53.76 | 61.33 | 54.83 | 54.59 | 53.19 | 56.64 | 78.59 | __**56.15**__ | 53.90 | 53.25 | ? |
| SelfAttention.1 | 52.24 | 62.70 | 53.17 | 53.55 | 53.96 | 58.34 | 80.98 | 53.96 | 51.66 | 52.66 | ? |
| Spacy-c.0 | 54.42 | 63.05 | 69.57 | 54.48 | 53.51 | 59.46 | 67.64 | 52.67 | ? | 54.71 | ? |
| Spacy-c.1 | 54.04 | 62.10 | 68.48 | 54.87 | 53.45 | 59.86 | 67.34 | 52.46 | 54.31 | 54.76 | ? |
| Spacy.0 | 53.49 | 65.65 | 69.02 | __**56.15**__ | 54.29 | __**63.66**__ | 81.05 | 52.95 | __**54.97**__ | __**55.65**__ | ? |
| Spacy.1 | 54.15 | __**65.70**__ | __**70.65**__ | 55.73 | 54.42 | 63.46 | 81.23 | ? | ? | 55.00 | ? |
| Spacy.2 | ? | ? | ? | ? | ? | ? | __**81.40**__ | ? | ? | ? | ? |
| Spacy.3 | ? | ? | ? | ? | ? | ? | 80.90 | ? | ? | ? | ? |
| Spacy.4 | ? | ? | ? | ? | ? | ? | 81.21 | ? | ? | ? | ? |


| Model | Twitter 90% Twitter 10% | YouTube 90% YouTube 10% | YouTube, News Twitter | Twitter, News YouTube | Twitter, YouTube News | News 90% News 10% | Twitter 90%, Twisty Twitter 10% | YouTube, News, CSI Twitter | Twitter, News, External YouTube | Twitter, YouTube, External News |
| ----- | ----------------------- | ----------------------- | --------------------- | --------------------- | --------------------- | ----------------- | ------------------------------- | -------------------------- | ------------------------------- | ------------------------------- |
| CNN.n+.0 | 62.46 | 59.70 | 53.55 | 52.91 | 51.79 | 59.50 | 71.26 | 52.38 | 52.20 | 54.16 |
| CNN.n+.1 | 63.51 | 60.23 | 55.20 | 53.21 | 52.52 | 60.00 | 71.14 | 52.68 | 52.18 | 53.83 |
| CNN.n+.2 | 63.73 | 59.51 | 54.57 | 53.46 | 52.62 | 59.50 | 70.69 | 52.73 | 52.65 | 53.12 |
| CNN.n.0 | 60.57 | 58.87 | 53.72 | 52.78 | 50.74 | 54.67 | 77.59 | 53.08 | 56.05 | 51.33 |
| CNN.n.1 | 60.57 | 58.72 | 50.99 | 55.19 | 52.84 | 54.17 | 77.96 | 53.32 | 50.13 | 52.47 |
| CNN.n.2 | 59.16 | 59.53 | 52.62 | 54.31 | 52.47 | 58.33 | 78.07 | 53.11 | 50.05 | 53.74 |
| CNN.y+.0 | 63.79 | 59.45 | 54.04 | 53.69 | 52.03 | 61.17 | 71.29 | 52.56 | 51.95 | 53.48 |
| CNN.y+.1 | 63.63 | 59.91 | 53.89 | 53.06 | 52.26 | 61.17 | 70.64 | 53.47 | 52.76 | 53.60 |
| CNN.y+.2 | 63.38 | 60.06 | 54.31 | 53.29 | 52.17 | 60.00 | 70.68 | 52.07 | 53.23 | 54.07 |
| CNN.y.0 | 59.02 | 57.43 | 51.36 | 53.82 | 51.90 | 56.83 | 78.55 | 53.14 | 50.51 | 52.93 |
| CNN.y.1 | 60.32 | 58.19 | 52.93 | __**56.73**__ | 51.64 | 57.50 | 77.21 | 52.84 | 50.36 | 51.19 |
| CNN.y.2 | 60.21 | 59.11 | 52.76 | 55.35 | 51.67 | 53.83 | 78.41 | 53.93 | 55.96 | 50.69 |
| KENLM.n.3.0 | 62.34 | 58.49 | 53.11 | 49.40 | 52.21 | 61.96 | 78.48 | 52.22 | 53.95 | 52.65 |
| KENLM.n.3.1 | 62.34 | 58.49 | 53.11 | 49.40 | 52.21 | 61.96 | 78.48 | 52.22 | 53.95 | 52.65 |
| KENLM.n.3.2 | 62.34 | 58.49 | 53.11 | 49.40 | 52.21 | 61.96 | 78.48 | 52.22 | 53.95 | 52.65 |
| KENLM.n.4.0 | 62.42 | 57.69 | 53.27 | 49.36 | 52.21 | 61.41 | 78.38 | 52.15 | 53.80 | 52.59 |
| KENLM.n.4.1 | 62.42 | 57.69 | 53.27 | 49.36 | 52.21 | 61.41 | 78.38 | 52.15 | 53.80 | 52.59 |
| KENLM.n.4.2 | 62.42 | 57.69 | 53.27 | 49.36 | 52.21 | 61.41 | 78.38 | 52.15 | 53.80 | 52.59 |
| KENLM.n.5.0 | 62.42 | 57.50 | 53.32 | 49.38 | 52.13 | 60.60 | 78.44 | __**57.81**__ | 49.98 | 50.76 |
| KENLM.n.5.1 | 62.42 | 57.50 | 53.32 | 49.38 | 52.13 | 60.60 | 78.44 | __**57.81**__ | 53.92 | 52.73 |
| KENLM.n.5.2 | 62.42 | 57.50 | 53.32 | 49.38 | 52.13 | 60.60 | 78.44 | 51.41 | 52.15 | 50.63 |
| KENLM.n.6.0 | 62.42 | 57.61 | 53.32 | 49.38 | 52.13 | 60.87 | 78.51 | __**57.81**__ | 53.73 | 52.73 |
| KENLM.n.6.1 | 62.42 | 57.61 | 53.32 | 49.38 | 52.13 | 60.87 | 78.51 | 52.17 | 49.97 | 50.76 |
| KENLM.n.6.2 | 62.42 | 57.61 | 53.32 | 49.38 | 52.13 | 60.87 | 78.51 | 52.17 | 53.91 | 52.67 |
| KENLM.y.3.0 | 61.82 | 59.30 | 53.64 | 49.69 | 51.50 | 62.77 | 78.78 | 52.20 | 53.92 | 53.66 |
| KENLM.y.3.1 | 61.82 | 59.30 | 53.64 | 49.69 | 51.50 | 62.77 | 78.78 | 52.20 | 53.92 | 53.66 |
| KENLM.y.3.2 | 61.82 | 59.30 | 53.64 | 49.69 | 51.50 | 62.77 | 78.78 | 52.20 | 53.92 | 53.66 |
| KENLM.y.4.0 | 62.46 | 59.26 | 53.63 | 49.72 | 51.56 | 63.04 | 78.97 | 52.11 | 53.92 | 54.23 |
| KENLM.y.4.1 | 62.46 | 59.26 | 53.63 | 49.72 | 51.56 | 63.04 | 78.97 | 52.11 | 53.92 | 54.23 |
| KENLM.y.4.2 | 62.46 | 59.26 | 53.63 | 49.72 | 51.56 | 63.04 | 78.97 | 52.11 | 53.92 | 54.23 |
| KENLM.y.5.0 | 62.51 | 58.95 | 53.52 | 49.76 | 51.45 | 62.50 | 78.86 | 52.15 | 49.34 | 53.68 |
| KENLM.y.5.1 | 62.51 | 58.95 | 53.52 | 49.76 | 51.45 | 62.50 | 78.86 | 51.31 | 54.96 | 53.68 |
| KENLM.y.5.2 | 62.51 | 58.95 | 53.52 | 49.76 | 51.45 | 62.50 | 78.86 | 51.18 | 54.96 | 53.68 |
| KENLM.y.6.0 | 62.42 | 59.03 | 53.60 | 49.77 | 51.45 | 62.77 | 78.97 | 55.68 | 54.95 | 52.13 |
| KENLM.y.6.1 | 62.42 | 59.03 | 53.60 | 49.77 | 51.45 | 62.77 | 78.97 | 52.15 | 54.98 | 52.10 |
| KENLM.y.6.2 | 62.42 | 59.03 | 53.60 | 49.77 | 51.45 | 62.77 | 78.97 | 52.15 | 54.98 | 53.66 |
| LSTM.n+.0 | 63.16 | 51.64 | 50.01 | 49.63 | 53.38 | 43.00 | 71.83 | 49.68 | 49.51 | 52.60 |
| LSTM.n+.1 | 62.94 | 51.83 | 50.00 | 49.73 | 49.12 | 45.00 | 70.79 | 49.74 | 53.49 | 53.26 |
| LSTM.n+.2 | 58.30 | 51.57 | 50.27 | 49.62 | 49.00 | 45.50 | 71.79 | 49.83 | 52.24 | 54.64 |
| LSTM.n.0 | 59.89 | 58.98 | 52.85 | 51.56 | 52.55 | 45.17 | 79.29 | 50.87 | 53.92 | 52.38 |
| LSTM.n.1 | 60.21 | 57.49 | 52.13 | 52.14 | 51.95 | 45.50 | 79.81 | 52.75 | 52.80 | 52.41 |
| LSTM.n.2 | 60.11 | 58.60 | 49.59 | 53.84 | 50.74 | 45.50 | 79.03 | 50.94 | 56.26 | 53.07 |
| LSTM.y+.0 | 62.75 | 51.68 | 50.27 | 49.79 | 53.62 | 45.17 | 71.89 | 49.95 | 53.09 | 48.86 |
| LSTM.y+.1 | 50.48 | 56.70 | 50.03 | 49.62 | 49.12 | 45.00 | 72.05 | 54.18 | 53.18 | 53.16 |
| LSTM.y+.2 | 62.57 | 51.53 | 50.28 | 49.71 | 49.07 | 45.50 | 71.73 | 49.78 | 53.36 | 53.29 |
| LSTM.y.0 | 56.43 | 58.26 | 51.00 | 53.49 | 51.33 | 43.00 | 78.64 | 51.18 | 52.52 | 52.34 |
| LSTM.y.1 | 59.95 | 58.36 | 49.76 | 52.48 | 50.07 | 45.50 | 79.80 | 52.39 | 52.75 | 53.17 |
| LSTM.y.2 | 60.67 | 58.38 | 52.17 | 52.96 | 51.43 | 45.50 | 79.84 | 49.73 | 51.69 | 52.28 |
| LSTMAttention.n+.0 | 63.10 | 59.62 | __**56.48**__ | 53.24 | 53.16 | 55.33 | 71.58 | 54.97 | 53.44 | 53.97 |
| LSTMAttention.n+.1 | 62.94 | 59.00 | 55.70 | 52.75 | 54.17 | 55.33 | 71.41 | 55.12 | 52.49 | 53.22 |
| LSTMAttention.n+.2 | 62.94 | 59.87 | 54.91 | 53.55 | __**54.91**__ | 55.83 | 72.21 | 54.70 | 52.67 | 54.12 |
| LSTMAttention.n.0 | 61.68 | 58.09 | 52.97 | 54.83 | 51.41 | 55.33 | 79.94 | 53.26 | 53.13 | 52.88 |
| LSTMAttention.n.1 | 60.71 | 58.70 | 51.96 | 53.25 | 51.31 | 45.50 | 78.73 | 52.45 | 53.64 | 52.26 |
| LSTMAttention.n.2 | 60.75 | 58.47 | 53.52 | 55.16 | 51.12 | 54.33 | 80.76 | 53.98 | 53.74 | 53.29 |
| LSTMAttention.y+.0 | 63.43 | 57.77 | 55.87 | 53.65 | 53.50 | 58.00 | 71.58 | 55.17 | 52.28 | 54.05 |
| LSTMAttention.y+.1 | 63.56 | 58.55 | 55.13 | 53.33 | 54.22 | 50.00 | 72.23 | 55.14 | 52.61 | 53.95 |
| LSTMAttention.y+.2 | 62.87 | 60.32 | 54.85 | 52.90 | 53.74 | 54.33 | 71.83 | 55.00 | 52.91 | 53.31 |
| LSTMAttention.y.0 | 60.21 | 58.38 | 52.84 | 53.90 | 51.17 | 55.33 | 76.96 | 53.29 | 51.38 | 53.50 |
| LSTMAttention.y.1 | 61.59 | 58.85 | 53.17 | 55.46 | 51.24 | 57.50 | 79.65 | 54.05 | 52.26 | 53.34 |
| LSTMAttention.y.2 | 59.68 | 59.11 | 53.51 | 55.53 | 49.21 | 58.50 | 80.22 | 53.57 | 53.18 | 53.14 |
| RCNN.n+.0 | 62.98 | 58.83 | 54.88 | 54.07 | 52.83 | 60.67 | 72.39 | 55.11 | 53.14 | 53.84 |
| RCNN.n+.1 | 63.41 | 59.32 | 54.86 | 54.12 | 53.79 | 60.17 | 72.29 | 55.07 | 52.80 | 54.43 |
| RCNN.n+.2 | 62.71 | 58.85 | 55.32 | 54.01 | 52.34 | 61.67 | 72.54 | 55.32 | 53.34 | 54.34 |
| RCNN.n.0 | 61.89 | 59.43 | 53.72 | 54.10 | 53.19 | 58.00 | 78.97 | 52.92 | 54.69 | 52.45 |
| RCNN.n.1 | 61.19 | 59.32 | 52.74 | 54.35 | 52.76 | 55.83 | 79.73 | 52.10 | 53.74 | 52.07 |
| RCNN.n.2 | 61.75 | 60.00 | 52.75 | 54.03 | 51.81 | 57.83 | 79.10 | 53.26 | __**56.27**__ | 51.57 |
| RCNN.y+.0 | 61.63 | 59.04 | 55.30 | 54.63 | 53.64 | 58.83 | 72.32 | 54.64 | 53.25 | 53.79 |
| RCNN.y+.1 | 63.49 | 59.15 | 54.83 | 53.63 | 52.45 | 60.67 | 73.15 | 55.02 | 53.64 | 54.17 |
| RCNN.y+.2 | 62.97 | 58.43 | 55.44 | 54.18 | 54.53 | 60.00 | 72.37 | 55.11 | 53.29 | 53.98 |
| RCNN.y.0 | 61.32 | 59.19 | 54.69 | 52.21 | 54.69 | 58.33 | 79.61 | 51.73 | 55.08 | 51.43 |
| RCNN.y.1 | 60.76 | 60.45 | 52.58 | 54.16 | 52.71 | 57.33 | 79.83 | 54.06 | 53.65 | 52.79 |
| RCNN.y.2 | 61.06 | 58.47 | 53.17 | 52.84 | 52.24 | 55.33 | 79.77 | 52.03 | 52.97 | 53.79 |
| RNN.n+.0 | 55.62 | 54.00 | 53.39 | 52.26 | 52.64 | 45.67 | 65.47 | 50.46 | 50.90 | 50.72 |
| RNN.n+.1 | 53.14 | 53.72 | 50.82 | 51.12 | 51.66 | 46.67 | 64.10 | 50.45 | 52.08 | 52.67 |
| RNN.n+.2 | 60.79 | 56.83 | 53.09 | 51.08 | 50.62 | 47.00 | 64.64 | 50.39 | 51.44 | 51.17 |
| RNN.n.0 | 56.43 | 57.53 | 50.56 | 53.03 | 52.97 | 51.67 | 79.51 | 51.14 | 50.50 | 51.67 |
| RNN.n.1 | 57.92 | 56.81 | 53.23 | 54.19 | 52.74 | 48.67 | 79.15 | 52.72 | 52.13 | 52.97 |
| RNN.n.2 | 58.35 | 56.94 | 53.14 | 51.29 | 52.21 | 49.67 | 79.02 | 50.98 | 54.66 | 52.12 |
| RNN.y+.0 | 56.62 | 53.23 | 52.99 | 53.36 | 50.40 | 47.67 | 64.17 | 50.50 | 52.27 | 49.57 |
| RNN.y+.1 | 54.37 | 50.98 | 50.73 | 51.20 | 51.45 | 49.00 | 65.41 | 50.52 | 52.96 | 51.19 |
| RNN.y+.2 | 55.97 | 52.94 | 50.67 | 51.11 | 53.16 | 49.17 | 62.43 | 50.10 | 51.86 | 52.53 |
| RNN.y.0 | 60.24 | 58.13 | 49.56 | 54.56 | 52.31 | 46.67 | 78.38 | 49.86 | 54.41 | 53.71 |
| RNN.y.1 | 58.27 | 57.74 | 52.85 | 52.79 | 51.47 | 49.67 | 80.40 | 53.40 | 54.05 | 53.88 |
| RNN.y.2 | 58.46 | 56.40 | 51.59 | 54.19 | 51.76 | 51.17 | 79.68 | 53.34 | 54.56 | 54.53 |
| SKLearn-knn.n.0 | 52.35 | 54.17 | 50.07 | 50.60 | 52.18 | 53.80 | 62.19 | 50.31 | 50.64 | 50.82 |
| SKLearn-knn.n.1 | 52.35 | 54.17 | 50.07 | 50.60 | 52.18 | 53.80 | 62.19 | 50.31 | 50.64 | 50.82 |
| SKLearn-knn.n.2 | 52.35 | 54.17 | 50.07 | 50.60 | 52.18 | 53.80 | 62.19 | 50.31 | 50.64 | 50.82 |
| SKLearn-knn.y.0 | 51.60 | 53.83 | 50.71 | 51.04 | 50.93 | 50.00 | 62.54 | 50.36 | 49.98 | 49.67 |
| SKLearn-knn.y.1 | 51.60 | 53.83 | 50.71 | 51.04 | 50.93 | 50.00 | 62.54 | 50.36 | 49.98 | 49.67 |
| SKLearn-knn.y.2 | 51.60 | 53.83 | 50.71 | 51.04 | 50.93 | 50.00 | 62.54 | 50.36 | 49.98 | 49.67 |
| SKLearn-log.n.0 | 62.70 | 58.98 | 53.36 | 53.14 | 52.89 | 61.96 | 76.42 | 54.05 | 51.72 | 52.29 |
| SKLearn-log.n.1 | 62.70 | 58.98 | 53.36 | 53.14 | 52.89 | 61.96 | 76.42 | 54.05 | 51.72 | 52.29 |
| SKLearn-log.n.2 | 62.70 | 58.98 | 53.36 | 53.14 | 52.89 | 61.96 | 76.42 | 54.05 | 51.72 | 52.29 |
| SKLearn-log.y.0 | 63.00 | 59.25 | 53.93 | 53.21 | 53.60 | 65.22 | 76.76 | 53.91 | 52.23 | 53.11 |
| SKLearn-log.y.1 | 63.00 | 59.25 | 53.93 | 53.21 | 53.60 | 65.22 | 76.76 | 53.91 | 52.22 | 53.11 |
| SKLearn-log.y.2 | 63.00 | 59.25 | 53.93 | 53.21 | 53.60 | 65.22 | 76.76 | 53.91 | 52.23 | 53.11 |
| SKLearn-nb.n.0 | 64.50 | 58.44 | 54.43 | 51.24 | 52.46 | 62.50 | 77.48 | 49.98 | 53.40 | 50.22 |
| SKLearn-nb.n.1 | 64.50 | 58.44 | 54.43 | 51.24 | 52.46 | 62.50 | 77.48 | 49.98 | 53.40 | 50.22 |
| SKLearn-nb.n.2 | 64.50 | 58.44 | 54.43 | 51.24 | 52.46 | 62.50 | 77.48 | 49.98 | 53.40 | 50.22 |
| SKLearn-nb.y.0 | 64.60 | 59.93 | 54.67 | 51.40 | 51.53 | 64.13 | 77.95 | 49.98 | 53.91 | 50.66 |
| SKLearn-nb.y.1 | 64.60 | 59.93 | 54.67 | 51.40 | 51.53 | 64.13 | 77.95 | 49.98 | 53.91 | 50.66 |
| SKLearn-nb.y.2 | 64.60 | 59.93 | 54.67 | 51.40 | 51.53 | 64.13 | 77.95 | 49.98 | 53.91 | 50.66 |
| SKLearn-rf.n.0 | 57.15 | 55.80 | 50.97 | 51.70 | 50.60 | 52.72 | 71.26 | 51.26 | 50.66 | 49.84 |
| SKLearn-rf.n.1 | 56.60 | 56.54 | 51.82 | 52.29 | 50.27 | 55.98 | 71.34 | 51.13 | 50.62 | 48.80 |
| SKLearn-rf.n.2 | 58.25 | 54.24 | 51.58 | 52.67 | 50.76 | 56.52 | 71.03 | 50.76 | 50.66 | 49.07 |
| SKLearn-rf.y.0 | 57.80 | 54.58 | 51.83 | 52.78 | 50.11 | 64.67 | 71.20 | 50.76 | 50.57 | 50.55 |
| SKLearn-rf.y.1 | 57.75 | 56.20 | 51.63 | 52.46 | 50.11 | 59.78 | 71.05 | 51.06 | 50.87 | 49.84 |
| SKLearn-rf.y.2 | 58.50 | 55.53 | 51.12 | 52.59 | 52.02 | 65.22 | 71.14 | 50.84 | 50.60 | 50.55 |
| SKLearn-svm.n.0 | 62.80 | 59.66 | 51.84 | 52.22 | 50.98 | 63.04 | 79.20 | 52.11 | 51.57 | 52.84 |
| SKLearn-svm.n.1 | 62.80 | 59.66 | 51.84 | 52.22 | 50.98 | 63.04 | 79.20 | 52.11 | 51.57 | 52.84 |
| SKLearn-svm.n.2 | 62.80 | 59.66 | 51.84 | 52.22 | 50.98 | 63.04 | 79.20 | 52.10 | 51.57 | 52.89 |
| SKLearn-svm.y.0 | 61.05 | 58.85 | 51.93 | 52.98 | 52.73 | 65.22 | 80.18 | 52.20 | 51.81 | 54.04 |
| SKLearn-svm.y.1 | 61.05 | 58.85 | 51.93 | 52.98 | 52.73 | 65.22 | 80.18 | 52.20 | 51.81 | 54.09 |
| SKLearn-svm.y.2 | 61.05 | 58.85 | 51.94 | 52.98 | 52.73 | 65.22 | 80.18 | 52.20 | 51.81 | 54.04 |
| SelfAttention.n+.0 | 62.17 | 58.81 | 55.86 | 54.16 | 53.74 | 62.17 | 71.63 | 54.44 | 53.70 | 52.81 |
| SelfAttention.n+.1 | 63.52 | 57.94 | 54.93 | 53.35 | 52.83 | 57.33 | 72.32 | 54.67 | 53.77 | 53.31 |
| SelfAttention.n+.2 | 63.32 | 59.04 | 55.93 | 54.55 | 52.33 | 57.00 | 71.82 | 54.59 | 53.49 | 53.62 |
| SelfAttention.n.0 | 61.81 | 57.83 | 53.68 | 53.82 | 51.76 | 51.17 | 79.29 | 54.13 | 55.42 | 52.33 |
| SelfAttention.n.1 | 60.37 | 59.47 | 53.85 | 56.13 | 51.91 | 56.33 | 79.19 | 52.93 | 54.80 | 51.59 |
| SelfAttention.n.2 | 61.27 | 59.17 | 52.48 | 54.46 | 51.57 | 51.17 | 79.42 | 54.36 | 52.54 | 52.12 |
| SelfAttention.y+.0 | 63.29 | 57.94 | 55.36 | 54.57 | 52.10 | 61.67 | 72.02 | 53.66 | 53.33 | 53.69 |
| SelfAttention.y+.1 | 62.68 | 57.85 | 54.99 | 53.70 | 52.50 | 60.17 | 72.08 | 54.71 | 54.00 | 52.97 |
| SelfAttention.y+.2 | 62.22 | 57.60 | 54.85 | 53.55 | 53.03 | 61.83 | 72.63 | 54.83 | 53.43 | 53.36 |
| SelfAttention.y.0 | 61.03 | 57.38 | 53.75 | 56.16 | 50.33 | 50.17 | 79.41 | 53.73 | 52.99 | 52.91 |
| SelfAttention.y.1 | 60.19 | 59.26 | 53.53 | 55.78 | 53.21 | 56.50 | 78.62 | 52.19 | 55.72 | 51.97 |
| SelfAttention.y.2 | 61.14 | 58.62 | 49.53 | 53.42 | 53.05 | 54.33 | 80.49 | 54.48 | 52.89 | 51.74 |
| Spacy.n.0 | 66.65 | 62.92 | 55.79 | 54.22 | 53.60 | __**72.28**__ | 80.49 | 55.94 | 53.76 | 54.75 |
| Spacy.n.1 | 66.85 | 62.78 | 54.98 | 54.67 | 54.09 | 70.65 | __**81.34**__ | 54.82 | 53.47 | 54.31 |
| Spacy.n.2 | 66.15 | __**63.39**__ | 55.92 | 54.68 | 53.55 | 69.02 | 81.08 | 55.14 | 53.64 | 54.04 |
| Spacy.y.0 | __**66.90**__ | 62.92 | 55.54 | 54.52 | 53.77 | 70.11 | 80.88 | 55.30 | 52.90 | 53.71 |
| Spacy.y.1 | 65.45 | 62.51 | 55.89 | 54.08 | 53.82 | 69.57 | 80.70 | 55.01 | 52.82 | __**55.08**__ |
| Spacy.y.2 | 65.75 | 62.92 | 55.92 | 54.33 | 54.64 | 71.74 | 80.96 | 55.24 | 52.66 | 54.69 |
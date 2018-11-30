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


| Model | Original 90% Original 10% | Twitter 90% Twitter 10% | YouTube 90% YouTube 10% | News 90% News 10% | Twitter, News YouTube | Twitter, YouTube News | YouTube, News Twitter |
| ----- | ------------------------- | ----------------------- | ----------------------- | ----------------- | --------------------- | --------------------- | --------------------- |
| CNN+.0 | 61.62 | 63.33 | 59.74 | 60.67 | 53.09 | 52.53 | 55.56 |
| CNN+.1 | 61.30 | 63.17 | 59.17 | 63.17 | 53.62 | 52.84 | 54.81 |
| CNN+.2 | 61.47 | 63.00 | 60.53 | 61.00 | 53.98 | 53.38 | 55.13 |
| CNN.0 | 57.57 | 61.27 | 58.34 | 56.33 | 54.41 | 51.10 | 52.89 |
| CNN.1 | 58.91 | 60.44 | 59.13 | 58.33 | 53.09 | 51.48 | 53.67 |
| CNN.2 | 59.50 | 61.27 | 58.89 | 51.17 | 51.48 | 49.47 | 52.63 |
| LSTM+.0 | 47.87 | 62.87 | 51.36 | 45.00 | 49.85 | 53.83 | 50.11 |
| LSTM+.1 | 61.28 | 63.98 | 51.77 | 45.00 | 49.58 | 53.07 | 49.91 |
| LSTM+.2 | 50.77 | 63.38 | 51.53 | 46.17 | 49.56 | 49.26 | 49.95 |
| LSTM.0 | 58.83 | 59.81 | 57.85 | 43.50 | 52.29 | 51.93 | 52.13 |
| LSTM.1 | 59.31 | 59.48 | 56.32 | 44.00 | 54.68 | 49.88 | 52.91 |
| LSTM.2 | 59.22 | 59.49 | 57.72 | 44.00 | 54.77 | 51.10 | 51.21 |
| LSTMAttention+.0 | 60.84 | 63.22 | 58.94 | 59.67 | 53.14 | 55.26 | 54.84 |
| LSTMAttention+.1 | 60.20 | 62.75 | 59.36 | 63.33 | 53.57 | __**55.66**__ | 55.89 |
| LSTMAttention+.2 | 60.78 | 62.94 | 59.04 | 60.17 | 53.93 | 54.21 | 55.12 |
| LSTMAttention.0 | 61.52 | 61.40 | 57.62 | 54.67 | 54.17 | 50.45 | 53.24 |
| LSTMAttention.1 | 61.36 | 61.67 | 58.17 | 57.33 | 53.49 | 51.97 | 53.15 |
| LSTMAttention.2 | 60.90 | 59.59 | 58.30 | 44.00 | 54.19 | 51.95 | 52.43 |
| RCNN+.0 | 62.38 | 62.40 | 58.62 | 60.17 | 53.29 | 52.19 | 55.43 |
| RCNN+.1 | 62.46 | 62.49 | 59.13 | 61.67 | 53.04 | 52.50 | 55.52 |
| RCNN+.2 | 61.96 | 62.24 | 58.45 | 60.83 | 53.16 | 53.16 | 55.46 |
| RCNN.0 | 61.21 | 61.65 | 58.98 | 57.33 | 52.89 | 51.36 | 54.40 |
| RCNN.1 | 61.35 | 61.87 | 59.04 | 55.33 | 54.07 | 52.74 | 53.37 |
| RCNN.2 | 61.81 | 62.33 | 58.51 | 57.00 | 53.41 | 53.40 | 53.84 |
| RNN+.0 | 50.94 | 55.27 | 55.09 | 48.67 | 50.22 | 51.31 | 50.33 |
| RNN+.1 | 50.70 | 56.10 | 51.30 | 48.00 | 51.70 | 51.86 | 50.16 |
| RNN+.2 | 51.52 | 56.11 | 52.15 | 48.00 | 50.67 | 49.10 | 50.66 |
| RNN.0 | 60.73 | 57.25 | 55.74 | 49.67 | 55.02 | 51.14 | 54.12 |
| RNN.1 | 56.37 | 59.75 | 56.15 | 50.67 | 54.33 | 51.36 | 51.07 |
| RNN.2 | 60.74 | 58.17 | 56.74 | 50.67 | 53.89 | 52.17 | 53.40 |
| SelfAttention+.0 | 60.76 | 61.86 | 57.81 | 58.67 | 53.67 | 51.81 | 55.51 |
| SelfAttention+.1 | 60.65 | 62.48 | 57.68 | 58.50 | 53.31 | 51.57 | 55.21 |
| SelfAttention+.2 | 61.13 | 62.76 | 58.38 | 60.50 | 53.45 | 51.41 | 55.17 |
| SelfAttention.0 | 58.77 | 61.37 | 57.68 | 54.33 | 53.26 | 53.62 | 53.76 |
| SelfAttention.1 | 60.96 | 61.19 | 59.40 | 52.67 | 55.10 | 54.14 | 54.31 |
| SelfAttention.2 | 61.73 | 63.13 | 59.64 | 53.83 | __**55.87**__ | 52.45 | 53.11 |
| Spacy.0 | 63.82 | 64.80 | 61.97 | 69.02 | 53.55 | 54.04 | __**56.16**__ |
| Spacy.1 | 64.42 | __**65.80**__ | 62.44 | __**72.28**__ | 54.16 | 53.71 | 55.31 |
| Spacy.2 | __**64.66**__ | 64.85 | __**62.92**__ | 65.76 | 53.92 | 54.42 | 55.35 |
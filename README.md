# Gender Prediction from Dutch

## Experiments

| Train | Dev | Spacy | RNN | CNN | RCNN | LSTM | LSTMAttention | SelfAttention |
| ----- | --- | ----- | --- | --- | ---- | ---- | ------------- | ------------- |
| Twitter 90%|Twitter 10% | 65.95 | 58.48 | 61.48 | 62.68 | 60.43 | 62.08 | 59.73 |
| All 90%|All 10% | 64.28 | 58.61 | 58.10 | 61.55 | 59.62 | 60.83 | 61.51 |
| YouTube 90%|YouTube 10% | 62.17 | 56.83 | 59.23 | 58.49 | 58.43 | 58.49 | 59.32 |
| News 90%|News 10% | 66.85 | 51.83 | 56.33 | 57.83 | 47.00 | 62.33 | 52.67 |
| Twitter, News|YouTube | 53.55 | 53.40 | 52.69 | 50.90 | 52.33 | 56.03 | 51.93 |
| Twitter, YouTube|News | 52.84 | 51.83 | 50.84 | 52.84 | 51.47 | 49.28 | 53.17 |
| YouTube, News|Twitter | 55.32 | 52.66 | 52.84 | 53.71 | 51.87 | 50.85 | 53.84 |

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


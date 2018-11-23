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
| YouTube, News|Twitter | 55.32 | ? | ? | ? | ? | ? | ? |
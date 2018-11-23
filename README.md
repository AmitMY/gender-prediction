# Gender Prediction from Dutch

## Experiments

| Train | Dev | Spacy | RNN | CNN | RCNN | LSTM | LSTMAttention | SelfAttention |
| ----- | --- | ----- | --- | --- | ---- | ---- | ------------- | ------------- |
| Twitter 90%|Twitter 10% | 65.95 | 58.48 | 61.48 | 62.68 | 60.43 | 62.08 | 59.73 |
| All 90%|All 10% | 64.28 | 58.61 | 58.10 | ? | 59.62 | ? | 61.51 |
| YouTube 90%|YouTube 10% | 62.17 | ? | 59.23 | ? | 58.43 | ? | 59.32 |
| News 90%|News 10% | 66.85 | ? | 56.33 | ? | 47.00 | ? | 52.67 |
| Twitter, News|YouTube | 53.55 | ? | ? | ? | 52.33 | ? | 51.93 |
| Twitter, YouTube|News | 52.84 | ? | ? | ? | 51.47 | ? | 53.17 |
| YouTube, News|Twitter | 55.32 | ? | ? | ? | ? | ? | ? |
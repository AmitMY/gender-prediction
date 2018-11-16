# Gender Prediction from Dutch

## Experiments
### Spacy TextCategorizer (CNN)

| Train            | Dev         | P     | R     | F     |
|------------------|-------------|-------|-------|-------|
| Twitter 90%      | Twitter 10% | 67%   | 66.3% | 67.3% |
| YouTube 90%      | YouTube 10% | 63.8% | 61.7% | 65.5% |
| News 90%         | News 10%    | 70.7% | 73.3% | 73.3% |
| Twitter, News    | YouTube     | 53.8% | 52.5% | 79.7% |
| YouTube, News    | Twitter     | 55.7% | 57.0% | 46.7% |
| Twitter, Youtube | News        | 52.1% | 56.6% | 18.3% |
| All 90%          | All 10%     | 62.8% | 61.6% | 62.3% |
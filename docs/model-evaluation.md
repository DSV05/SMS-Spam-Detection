# SMS Spam Detection — Model Evaluation

Generated (UTC): 2026-10-15T04:38:29+00:00

## Dataset and split

- Source file: `SMSSpamCollection` (SHA-256 `7d039a24a6083ed9ef0f806ebad56bbb976e3aeb8de05669173bfdc4996c239d`)
- Records: 5574; unique message texts: 5171
- Class counts: {'ham': 4827, 'spam': 747}
- Exact duplicate rows retained: 403
- Missing cells: 0; conflicting-label messages: 0
- Split: 4462 train / 1112 test (target test fraction 20%; actual 19.95%)
- Message-text overlap: 0
- Random seed: 42; stratified by unique message

## Model and evaluation

- Pipeline: TfidfVectorizer(ngram_range=(1, 2)) → MultinomialNB
- scikit-learn: 1.7.2
- Accuracy: 0.9442
- Macro precision / recall / F1: 0.9697 / 0.7947 / 0.8552
- HAM precision / recall / F1: 0.9394 / 1.0000 / 0.9688
- SPAM precision / recall / F1: 1.0000 / 0.5894 / 0.7417

Confusion matrix (rows = actual, columns = predicted; ham, spam):

```text
[961, 0]
[62, 89]
```

## Classification report

| Class | Precision | Recall | F1 | Support |
| --- | ---: | ---: | ---: | ---: |
| HAM | 0.9394 | 1.0000 | 0.9688 | 961 |
| SPAM | 1.0000 | 0.5894 | 0.7417 | 151 |
| macro avg | 0.9697 | 0.7947 | 0.8552 | 1112 |
| weighted avg | 0.9476 | 0.9442 | 0.9379 | 1112 |

Accuracy alone can be misleading because ham examples outnumber spam examples. Spam recall and the confusion matrix show how many spam messages were missed; precision shows the false-alarm trade-off. These metrics describe one held-out split, not guaranteed future performance.

Model probabilities are not calibrated and should not be treated as certainty.

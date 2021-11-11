# SMS Spam
## Classifying SMS text messages into "spam" or "ham" (non-spam)

There is a dataset on Kaggle called the [SMS Spam Collection
dataset](https://www.kaggle.com/uciml/sms-spam-collection-dataset) and it has
several thousand SMS text messages labeled as either "spam", which we all know
the meaning of and "ham", which is another way of saying "not spam". Naturally
I am attempting classification using supervised learning here. Because the
original dataset is only ~13% spam, I used over and under sampling to produce
a nearly perfectly balanced dataset, `sms-balanced.csv`, which is what
`learn.R` uses upon running.

The effort marked my first use of the [tidymodels](https://www.tidymodels.org)
package and other associated packages in R. I feel they're almost but not
quite what R really needs and I had a number of frustrations using it, which
culminated in me throwing out any attempt to tune the final test set model
which was fortunately unneeded to begin with. There is a very solid
possibility that any further attempts I make at text classification will be
done using the Python library [spaCy](https://spacy.io).

Having said all of that, the project is currently the absolute best I've been
able to do among the various machine learning efforts I've had a go at.
Accuracy, sensitivity, specificity and AUC-ROC for the test set are **all**
through the roof at over 0.99 each:

```
[1] "Evaluating performance on test set:"
# A tibble: 4 × 4
  .metric  .estimator .estimate .config
  <chr>    <chr>          <dbl> <chr>
1 accuracy binary         0.991 Preprocessor1_Model1
2 sens     binary         0.991 Preprocessor1_Model1
3 spec     binary         0.991 Preprocessor1_Model1
4 roc_auc  binary         0.998 Preprocessor1_Model1
```

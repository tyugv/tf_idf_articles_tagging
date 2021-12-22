# tf_idf_articles_tagging
Predict article tags with tf_idf and lemming. 

The model was trained on BBC articles dataset: https://www.kaggle.com/c/learn-ai-bbc. 

For lemming was used code from https://www.machinelearningplus.com/nlp/lemmatization-examples-python/.

Install
```
pip install git+https://github.com/tyugv/tf_idf_articles_tagging.git
```
```
from articles_tagging.tf_idf import TfIdf
tf_idf = TfIdf()
tf_idf.tagging('Some long article for test')
```

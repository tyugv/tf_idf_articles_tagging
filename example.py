from sklearn.feature_extraction.text import TfidfVectorizer
from joblib import load
import os
import sys

from preprocessing import preprocess
from training import training


def tagging(text, tf_idf_model, max_value=0.15):
  text = preprocess(text)
  result = tf_idf_model.transform([text]).toarray()[0]
  tags = []
  for value, word in zip(result, tf_idf_model.get_feature_names_out()):
  	if value > max_value:
  		tags.append(word)
  return tags


if __name__ == '__main__':
	if 'tf_idf.joblib' not in os.listdir():
		training()
	tf_idf = load('tf_idf.joblib')

	if len(sys.argv) > 1:
		print(tagging(str(sys.argv[1]), tf_idf))
	else:
		print(tagging('Hello world!', tf_idf))


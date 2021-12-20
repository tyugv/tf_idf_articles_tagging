import pandas as pd
from joblib import dump
from sklearn.feature_extraction.text import TfidfVectorizer

from articles_tagging.preprocessing import preprocess_full_dataset


def training():
	pd_data = pd.read_csv('data/BBC News Train.csv')
	data = preprocess_full_dataset(list(pd_data['Text']))

	tf_idf = TfidfVectorizer().fit(data)
	dump(tf_idf, 'model_weight/tf_idf.joblib')


if __name__ == '__main__':
	training()

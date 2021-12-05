import pandas as pd
from joblib import dump
from sklearn.feature_extraction.text import TfidfVectorizer

from preprocessing import preprocess_full_dataset


def training():
	pd_data = pd.read_csv('BBC News Train.csv')
	data = preprocess_full_dataset(list(pd_data['Text']))

	tf_idf = TfidfVectorizer().fit(data)
	dump(tf_idf, 'tf_idf.joblib')


if __name__ == '__main__':
	training()

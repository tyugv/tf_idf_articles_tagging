import io
import requests
import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from articles_tagging.preprocessing import preprocess, preprocess_full_dataset

WEIGHTS_URL = 'https://raw.github.com/tyugv/tf_idf_articles_tagging/main/articles_tagging/model_weights/tf_idf.joblib'


class TfIdf:
    def __init__(self, pretrained=True):
        if pretrained:
            print('Load weights')
            weights_bytes = requests.get(WEIGHTS_URL).content
            weights = io.BytesIO(weights_bytes)
            self.model = joblib.load(weights)
        else:
            self.model = TfidfVectorizer()

    def training(self, data_path=None):
        if type(data_path) is not str:
            raise Exception('Please add path to data in csv with column named as Text for training')
        pd_data = pd.read_csv(data_path)
        print('Preprocessing dataset')
        data = preprocess_full_dataset(list(pd_data['Text']))
        print('Training')
        self.model.fit(data)

    def tagging(self, text, max_value=0.15):
        text = preprocess(text)
        result = self.model.transform([text]).toarray()[0]
        tags = []
        values = []
        for value, word in zip(result, self.model.get_feature_names_out()):
            if value > max_value:
                tags.append(word)
                values.append(value)
        # sort by tf-idf value
        sorted_tags = [(tags[word_value[0]], word_value[1]) for word_value in sorted(enumerate(values),
                                                                                     key=lambda x:x[1], reverse=True)]
        return sorted_tags

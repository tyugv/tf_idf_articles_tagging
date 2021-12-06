from sklearn.feature_extraction.text import TfidfVectorizer
from joblib import load

from src.preprocessing import preprocess


def tagging(text, tf_idf_model, max_value=0.15):
    text = preprocess(text)
    result = tf_idf_model.transform([text]).toarray()[0]
    tags = []
    for value, word in zip(result, tf_idf_model.get_feature_names_out()):
        if value > max_value:
            tags.append(word)
    return tags

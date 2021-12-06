from sklearn.feature_extraction.text import TfidfVectorizer
from joblib import load

from articles_tagging.preprocessing import preprocess


def tagging(text, max_value=0.15):
    if 'tf_idf.joblib' not in os.listdir('model_weight'):
        training()
    tf_idf_model = load('model_weight/tf_idf.joblib')
    text = preprocess(text)
    result = tf_idf_model.transform([text]).toarray()[0]
    tags = []
    for value, word in zip(result, tf_idf_model.get_feature_names_out()):
        if value > max_value:
            tags.append(word)
    return tags

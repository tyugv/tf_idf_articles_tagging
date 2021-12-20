from sklearn.feature_extraction.text import TfidfVectorizer
from articles_tagging.preprocessing import preprocess


def tagging(text, tf_idf_model, max_value=0.15):
    text = preprocess(text)
    result = tf_idf_model.transform([text]).toarray()[0]
    tags = []
    values = []
    for value, word in zip(result, tf_idf_model.get_feature_names_out()):
        if value > max_value:
            tags.append(word)
            values.append(value)
    sorted_tags = [tags[i[0]] for i in sorted(enumerate(values), key=lambda x:x[1], reverse=True)]
    return sorted_tags

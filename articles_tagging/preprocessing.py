import nltk
from nltk.corpus import stopwords, wordnet
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem import WordNetLemmatizer


# nltk data download if not exist
def download_nltk_library(name, path):
    try:
        nltk.find(path + '/' + name)
    except LookupError:
        nltk.download(name)


download_nltk_library('punkt', 'tokenizers')
download_nltk_library('stopwords', 'corpora')
download_nltk_library('wordnet', 'corpora')
download_nltk_library('averaged_perceptron_tagger', 'taggers')

stopWords = set(stopwords.words('english'))
stopWords.add('mr')
stopWords.add('ms')
punct = "/-'?!.,#$%\'()*+-/:;<=>@[\\]^_`{|}~`" + '""“”’' + '∞θ÷α•à−β∅³π‘₹´°£€\×™√²—–&'
lemmatizer = WordNetLemmatizer()


def get_wordnet_pos(word):
    """Map POS tag to first character lemmatize() accepts"""
    tag = nltk.pos_tag([word])[0][1][0].upper()
    tag_dict = {"J": wordnet.ADJ,
                "N": wordnet.NOUN,
                "V": wordnet.VERB,
                "R": wordnet.ADV}
    return tag_dict.get(tag, wordnet.NOUN)

def clean_special_chars(text):
    return text.lower().translate(str.maketrans('', '', punct))


def delete_stop_words(words):
    clean_words = []
    for word in words:
        if word not in stopWords:
            if len(word) > 0:
                clean_words.append(word)
    return clean_words


def preprocess(article):
    preprocessed_article = ''
    for sentence in sent_tokenize(article):
        sentence = clean_special_chars(sentence)
        tokenized_sentence = delete_stop_words(word_tokenize(sentence))
        lemmatized_sentence = [lemmatizer.lemmatize(word, get_wordnet_pos(word)) for word in tokenized_sentence]
        preprocessed_article += ' '.join(lemmatized_sentence)
        preprocessed_article += ' '
    return preprocessed_article


def preprocess_full_dataset(data):
    return list(map(preprocess, data))

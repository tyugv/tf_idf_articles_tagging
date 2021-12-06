import os
import sys
from joblib import load

from src.tagging import tagging
from src.training import training


if __name__ == '__main__':
    if 'tf_idf.joblib' not in os.listdir('../model_weight'):
        training()
    tf_idf = load('../model_weight/tf_idf.joblib')

    if len(sys.argv) > 1:
        print(tagging(str(sys.argv[1]), tf_idf))
    else:
        print(tagging('Hello world!', tf_idf))

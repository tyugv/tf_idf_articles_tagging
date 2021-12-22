import sys
from articles_tagging.tf_idf import TfIdf


if __name__ == '__main__':
    tf_idf = TfIdf()

    # data_path = ### your data path
    # tf_idf = TfIdf(pretrained=False)
    # tf_idf.training(data_path)

    if len(sys.argv) > 1:
        print(tf_idf.tagging(str(sys.argv[1])))
    else:
        print(tf_idf.tagging('Hello world!'))

from distutils.core import setup

setup(name="articles_tagging",
      version="0.0.1",
      description='file: README.md',
      packages=["articles_tagging"],
      author='Milena',
      url='https://github.com/tyugv/tf_idf_articles_tagging',
      )

_deps = ["joblib==1.1.0",
         "nltk==3.6.5",
         "pandas==1.3.4",
         "scikit-learn==1.0.1"]

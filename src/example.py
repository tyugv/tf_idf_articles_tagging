import os
import sys

from articles_tagging.tagging import tagging
from articles_tagging.training import training


if __name__ == '__main__':
    if len(sys.argv) > 1:
        print(tagging(str(sys.argv[1])))
    else:
        print(tagging('Hello world!'))

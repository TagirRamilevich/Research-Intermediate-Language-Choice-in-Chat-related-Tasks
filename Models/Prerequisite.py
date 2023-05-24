!pip install fasttext
!pip install -U easynmt
!pip install sacremoses
!pip install pandarallel
!pip install npm
!pip install nodeenv
!pip install -q Naked==0.1.31
!npm install --save @iamtraction/google-translate
!npm install -g npm
!pip install utils
!pip install transformers

import ast
import io
import json
import matplotlib.pyplot as plt
import nltk
import numpy as np
import os
import pandas as pd
from pathlib import Path
import re
import requests
import seaborn as sns
from sklearn.metrics import f1_score
from sklearn.model_selection import ParameterGrid, train_test_split
import string
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader
from transformers import (BertTokenizer, BertModel, DataCollatorWithPadding,
                          MT5Model, MT5ForConditionalGeneration, MT5Tokenizer,
                          XLMRobertaModel)
import uuid
from urllib.error import HTTPError
from typing import Union, List, Optional
from tqdm import tqdm
import fasttext
from gensim.parsing.preprocessing import STOPWORDS, remove_stopwords
from google.colab import drive
from Naked.toolshed.shell import muterun_js
import fasttext.util
from easynmt import EasyNMT
from pandarallel import pandarallel
from transformers import BloomModel, BloomTokenizerFast, AutoTokenizer, AutoModel

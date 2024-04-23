import pandas as pd 
import nltk
import string 
from nltk.probability import FreqDist
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

dff=pd.read_csv('paragraphs.txt',sep='\t', header=None)
dff.head()

dff=dff.applymap(lambda x: x.lower() if isinstance(x, str) else x)
dff.head()

nltk.download('stopwords')
from nltk.corpus import stopwords
stop_words=set(stopwords.words('english'))
def remove_stop(x):
    return " ".join([word for word in str(x).split() if word not in stop_words])

dff=dff.applymap(remove_stop)
dff.head()

from nltk.probability import FreqDist 
nltk.download('punkt')
text_data=' '.join(dff.values.flatten())
words= word_tokenize(text_data)
filtered_words=[word for word in words if word.lower() not in stop_words]

word_freq= FreqDist(filtered_words)
for word, freq in word_freq.items():
    print(f"{word}: {freq}")
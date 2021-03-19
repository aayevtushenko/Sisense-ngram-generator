
# N-Gram Generator for Sisense

## Setup

Make sure that the language you will be running this script on is supported by a matching SpaCy pretrained model: https://spacy.io/usage/models


```python
import pandas as pd
import numpy as np
import re

import nltk, re, string, collections
from nltk.util import ngrams # function for making ngrams

from collections import Counter
from collections import defaultdict 

from matplotlib.pyplot import plot

import spacy

!python3 -m spacy download fr_core_news_sm
```

### Load the data into the ```df``` variable.
If using Sisense Custom Code, point the query output to this variable.


```python
df = pd.read_csv('customer_data.csv')
df.head()
```

### Load the SpaCy model for the language of your choice
#### The cell below is formatted for french.

For English you would use ```nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])```

We disable the **parser** and **ner** components of the SpaCy pipeline for speed. We are not using them in this application.


```python
nlp = spacy.load('fr_core_news_sm', disable=['parser', 'ner'])
```

### This function will return the tokens for ngram generation
Add additional logic to customize what tokens are produced for ngram generation.


```python
def clean(document):
    return [token.lemma_ for token in nlp(document) if (token.is_stop == False and token.pos_ == 'NOUN' and token.lemma_ != 'oui')]
```

### Settings

##### ```ngram_depth:``` how many words should be comined in the ngram.
>1: "product"

>2: "product is"

>3: "product is great"

##### ```limit:``` For each ngram above the order of 1, ```limit``` is the number of children ngrams that must exist for the higher order ngram to be added to the list.
This logic cuts down on the amount of processing that is needed for higher order ngrams.

##### ```min_ngram_length:``` Prune ngrams that are below this order. Default is 1 or no pruning.
##### ```one_gram_frequency:``` Prune 1-grams that show up less than this amount of times. Default is 10.

##### ```text_column:``` The column containing text to analyze
##### ```id_column:``` The ID column that will join the output to the full text


```python
ngram_depth = 3
limit = 3
min_ngram_length = 1
one_gram_frequency_min = 10

text_column = 'CustomerText'
id_column = 'CallId'
```

### Execute ngram object generation


```python
counter = Counter()
mapping = defaultdict(set)

for ngram_order in range(1,ngram_depth+1):
    for index, row in df.iterrows(): 
        index = row[id_column]
        tokens = clean(row[text_column])
    
        if len(tokens) >= ngram_order:
            ngram_set = ngrams(tokens, ngram_order)
            if ngram_order == 1:
                for ngram in ngram_set:
                    counter[ngram] += 1
                    mapping[ngram].add(index)
            else:
                for ngram in ngram_set:
                    front_parent = counter.get(ngram[1:], 0)
                    back_parent = counter.get(ngram[:-1], 0)
                    if front_parent >= limit or back_parent >= limit:
                        counter[ngram] += 1
                        mapping[ngram].add(index)
        
if min_ngram_length != 1:
    for key in [key for key in counter if len(key) < min_ngram_length]: 
        del counter[key]
            
if one_gram_frequency_min != 1:
    for key in [key for key in counter if counter[key] < one_gram_frequency_min and len(key) == 1]: 
        del counter[key]
```

### Create dataframes for joining tables and expoding related text IDs into columns


```python
counts_df = pd.DataFrame({'keys':counter.keys()})
print('counts df', counts_df.head())

mapping_df = pd.DataFrame({'keys':mapping.keys(), 'indexes':mapping.values()})
print('mapping df before explode', mapping_df.head())

mapping_df = mapping_df.explode('indexes')
print('mapping df after explode', mapping_df.head())

output_df = pd.merge(counts_df, mapping_df, how='inner', on='keys')

# review final output
print('\nhead\n')
print(output_df.head())
print('\ntail\n')
print(output_df.tail())
```

### Clean final output and add ngram length for dashboard filtering


```python
output_df = output_df[output_df['keys'] != ('nan',)]
output_df['ngram_length'] = output_df.apply(lambda row: len(row['keys']), axis=1)
output_df['keys'] = output_df.apply(lambda row: ' '.join(row['keys']), axis=1)

print(output_df.head())
print(output_df.tail())
```

### Out to CSV
If this is in Sisense Custom Code, this last cell is not necessary


```python
output_df.to_csv('ngrams_no_oui.csv', index=False)
```

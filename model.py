import gensim
import numpy as np
import pandas as pd
import py3hlda
import nltk
from nltk.stem import WordNetLemmatizer
from gensim.utils import lemmatize
from gensim.corpora import Dictionary
from gensim.corpora import MmCorpus
from gensim.models.ldamodel import LdaModel
from gensim.models import hdpmodel
from gensim.parsing.preprocessing import STOPWORDS
from gensim.models.phrases import Phrases, Phraser
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel

import warnings
import pickle

warnings.filterwarnings("ignore")

df = pd.read_csv('healtcare2.csv')
df = df.rename(columns={"Search terms report": "content"})

data = df.content.values.tolist() # data is a list that includes string values in "content" column.


# Tokenization and removing puntuations from values in list.
def sent_to_words(sentences):
    for sentence in sentences:
        yield (gensim.utils.simple_preprocess(str(sentence), deacc=True))  # deacc=True removes punctuations


data_words = list(sent_to_words(data))  # data_words is a list that inclues lists of values with cleaned form.

# Build the bigram and trigram models
bigram = Phrases(data_words, threshold=1) # higher threshold fewer phrases.
trigram = Phrases(bigram[data_words], threshold=1)
# Faster way to get a sentence clubbed as a trigram/bigram
bigram_mod = Phraser(bigram)
trigram_mod = Phraser(trigram)


nltk.download('wordnet')
lemmatizer = WordNetLemmatizer()

def remove_stopwords(texts):
    return [[word for word in simple_preprocess(str(doc)) if word not in nlp.Defaults.stop_words] for doc in texts]

def make_bigrams(texts):
    return [bigram_mod[doc] for doc in texts]

def make_trigrams(texts):
    return [trigram_mod[bigram_mod[doc]] for doc in texts]

def lemmatization(texts):
    texts_out = []
    for sent in texts:
        lemmatized_output = [lemmatizer.lemmatize(w) for w in sent]
        texts_out.append(lemmatized_output)
    return texts_out

# Remove Stop Words
data_words_nostops = remove_stopwords(data_words)

# Form Bigrams
data_words_bigrams = make_bigrams(data_words_nostops)

# Initialize spacy 'en' model, keeping only tagger component (for efficiency)
nlp = spacy.load('en', disable=['parser', 'ner'])

# Do lemmatization keeping only noun, adj, vb, adv
data_lemmatized = lemmatization(data_words_bigrams)

# Create Dictionary
id2word = corpora.Dictionary(data_lemmatized) # her unique kelime ve söz öbeği için id verdi.

# Create Corpus
texts = data_lemmatized

# Term Document Frequency
corpus = [id2word.doc2bow(text) for text in texts] # format = list of (token_id, token_count) tuples

def lda_model():
    lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus,
                                                id2word=id2word,
                                                num_topics=8,
                                                random_state=100,
                                                update_every=1,
                                                chunksize=10,
                                                passes=10,
                                                alpha='auto',
                                                per_word_topics=True)
    pickle.dump(lda_model, open('model.pkl', 'wb'))
    model = pickle.load(open('model.pkl', 'rb'))

def coherence_lda(): # Compute Coherence Score for LDA
    coherence_model_lda = CoherenceModel(model=lda_model, texts=texts, dictionary=id2word, coherence='c_v')
    coherence_lda = coherence_model_lda.get_coherence()

    return coherence_lda

def hiearachical_lda():

    n_samples = 10  # no of iterations for the sampler
    alpha = 10.0  # smoothing over level distributions
    gamma = 1.0  # CRP smoothing parameter; number of imaginary customers at next, as yet unused table
    eta = 0.1  # smoothing over topic-word distributions
    num_levels = 3  # the number of levels in the tree
    display_topics = 3  # the number of iterations between printing a brief summary of the topics so far
    n_words = 5  # the number of most probable words to print for each topic after model estimation
    with_weights = False  # whether to print the words with the weights

    common_dictionary_t = Dictionary(texts)
    voc = sorted(list(common_dictionary_t.values()))
    voc_ind = {k:v for v,k in enumerate(voc)}
    new_corpus = []
    for doc in texts:
        new_doc = []
        for word in doc:
            word_idx = voc_ind[word]
            new_doc.append(word_idx)
        new_corpus.append(new_doc)

    hlda = py3hlda.sampler.HierarchicalLDA(new_corpus, voc, alpha=alpha, gamma=gamma, eta=eta, num_levels=num_levels)

    pickle.dump(hlda, open('model.pkl', 'wb'))
    model = pickle.load(open('model.pkl', 'rb'))


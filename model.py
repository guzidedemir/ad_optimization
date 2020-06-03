import gensim
import numpy as np
import pandas as pd
import py3hlda
from py3hlda.sampler import HierarchicalLDA
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

class Model:

    def __init__(self,df):
        self.df = df
        self.main()

    def sent_to_words(self, sentences): # Tokenization and removing puntuations from values in list.
        for sentence in sentences:
            yield (gensim.utils.simple_preprocess(str(sentence), deacc=True))  # deacc=True removes punctuations

    def main(self):
        self.df = self.df.rename(columns={"Search terms report": "content"})
        self.data = self.df.content.values.tolist() # data is a list that includes string values in "content" column.
        self.data_words = list(self.sent_to_words(self.data))  # self.data_words is a list that inclues lists of values with cleaned form.
        # Build the self.bigram and self.trigram models
        self.bigram = Phrases(self.data_words, threshold=1) # higher threshold fewer phrases.
        self.trigram = Phrases(self.bigram[self.data_words], threshold=1)
        # Faster way to get a sentence clubbed as a self.trigram/self.bigram
        self.bigram_mod = Phraser(self.bigram)
        self.trigram_mod = Phraser(self.trigram)
        nltk.download('wordnet')
        self.lemmatizer = WordNetLemmatizer()
        self.nlp_stopwords = {'somewhere', 'such', 'six', 'go', 'quite', 'someone', 'then', 'please', 'least', 'not', 'being',
                             'sometimes', 'against', 'themselves', 'anywhere', 'amount', 'did', 'him', 'fifty', 'since', 'but', 'under',
                             'neither', 'around', 'full', 'off', 'does', 'done', 'whereafter', 'whom', 'down', 'been', 'perhaps', 'seeming', 'they',
                             'whence', 'here', 'no', 'me', 'how', 'it', 'in', '’m', 'every', 'former', 'much', 'after', 'moreover', 'becoming', 'another',
                             'has', 'per', 'though', "'s", 'if', 'own', 'them', 'therein', 'whenever', 'himself', 'might', 'the', 'a', 'have', 'itself',
                             'onto', 'across', 'yours', 'doing', 'wherever', '’d', 'whereas', 'you', 'i', 'noone', 'must', 'than', 'using', 'something',
                             'indeed', 'say', 'below', 'never', 'this', 'beyond', 'used', 'may', 'any', 'often', 'ever', 'side', 'our', 'n’t', 'us', 'over',
                             'become', 'nowhere', 'name', 'less', 'hence', 'thru', 'thereby', "'m", 'until', 'call', 'everything', 'cannot', 'hereafter', 'anyhow',
                             'forty', 'show', 'that', "'ll", 'together', 'behind', 'whatever', 'toward', 'where', 'whereby', '‘d', '’ve', 'even', 'beside', 'all',
                             '‘s', 'twenty', 'because', 'most', 'unless', 'seem', 'along', 'are', 'none', 'ours', 'through', 'front', 'from', 'n‘t', 'eleven',
                             'yourselves', 're', 'back', 'ten', 'top', "n't", 'thereafter', 'so', 'her', 'now', 'others', 'herself', 'for', 'well', 'without',
                             'when', 'just', 'what', 'their', 'thus', 'more', 'first', 'few', 'already', 'make', 'various', 'yet', "'ve", 'am', 'there', 'whose',
                             'sometime', 'anyway', 'several', 'formerly', 'get', 'next', 'while', 'due', 'ca', 'should', 'sixty', 'very', 'same', 'serious',
                             'hereupon', 'third', 'rather', 'out', '‘m', 'into', 'my', 'will', '‘ll', 'before', 'thence', 'further', 'still', 'also', 'five',
                             'some', 'those', 'almost', 'above', 'nobody', 'nothing', 'of', 'beforehand', 'seems', 'else', 'except', 'do', 'during', 'towards',
                             'therefore', 'is', 'ourselves', 'thereupon', 'elsewhere', 'however', 'fifteen', 'herein', 'he', 'bottom', 'among', 'although', 'and',
                             'myself', 'nine', 'really', 'take', 'she', 'became', '’ll', 'both', 'mostly', 'other', 'twelve', 'between', 'last', 'enough', 'only',
                             'put', 'either', "'re", 'mine', '‘ve', 'everyone', 'nor', 'latter', 'was', 'again', 'could', 'alone', 'up', 'were', 'whither', 'once', 'with', 'latterly', 'eight', 'seemed', 'these', 'hers', 'upon', 'your', 'part', 'hundred', 'to', 'which', 'whole', '’re', 'via', 'one', 'too', 'on', 'whereupon', 'can', 'made', 'namely', 'keep', 'hereby', 'regarding', 'wherein', 'anyone', 'many', "'d", 'whoever', 'always', 'four', 'otherwise', 'an', 'his', 'somehow', 'about', 'throughout', 'nevertheless', 'two', 'everywhere', 'see', 'whether', '’s', 'three', 'give', 'becomes', 'besides', 'empty', 'or', 'afterwards', 'by', 'each', 'move', 'as',
                             'within', 'yourself', 'amongst', 'we', 'be', 'why', 'would', 'who', 'had', 'meanwhile', 'its', 'anything', '‘re', 'at'}
        self.pre_processing()

    def remove_stopwords(self,texts):
        return [[word for word in simple_preprocess(str(doc)) if word not in self.nlp_stopwords] for doc in texts]

    def make_bigrams(self,texts):
        return [self.bigram_mod[doc] for doc in texts]

    def make_trigrams(self,texts):
        return [self.trigram_mod[self.bigram_mod[doc]] for doc in texts]

    def lemmatization(self,texts):
        texts_out = []
        for sent in texts:
            lemmatized_output = [self.lemmatizer.lemmatize(w) for w in sent]
            texts_out.append(lemmatized_output)
        return texts_out

    def pre_processing(self):

        self.data_words_nostops = self.remove_stopwords(self.data_words) # Remove Stop Words
        self.data_words_bigrams = self.make_bigrams(self.data_words_nostops) # Form Bigrams
        #nlp = spacy.load('en', disable=['parser', 'ner']) # Initialize spacy 'en' model, keeping only tagger component (for efficiency)
        self.data_lemmatized = self.lemmatization(self.data_words_bigrams) # Do lemmatization keeping only noun, adj, vb, adv
        self.id2word = corpora.Dictionary(self.data_lemmatized) # her unique kelime ve söz öbeği için id verdi. Create Dictionary
        self.texts = self.data_lemmatized
        self.corpus = [self.id2word.doc2bow(text) for text in self.texts] # format = list of (token_id, token_count) tuples
        self.hiearachical_lda()

    def lda_model(self):
        lda_model = gensim.models.ldamodel.LdaModel(corpus=self.corpus,
                                                    id2word=self.id2word,
                                                    num_topics=8,
                                                    random_state=100,
                                                    update_every=1,
                                                    chunksize=10,
                                                    passes=10,
                                                    alpha='auto',
                                                    per_word_topics=True)

    def coherence_lda(self): # Compute Coherence Score for LDA
        coherence_model_lda = CoherenceModel(model=self.lda_model, texts=self.texts, dictionary=self.id2word, coherence='c_v')
        coherence_lda = coherence_model_lda.get_coherence()

        return coherence_lda

    def hiearachical_lda(self):

        n_samples = 3  # no of iterations for the sampler
        alpha = 10.0  # smoothing over level distributions
        gamma = 1.0  # CRP smoothing parameter; number of imaginary customers at next, as yet unused table
        eta = 0.1  # smoothing over topic-word distributions
        num_levels = 3  # the number of levels in the tree
        display_topics = 2  # the number of iterations between printing a brief summary of the topics so far
        n_words = 3  # the number of most probable words to print for each topic after model estimation
        with_weights = False  # whether to print the words with the weights

        common_dictionary_t = Dictionary(self.texts)
        voc = sorted(list(common_dictionary_t.values()))
        voc_ind = {k:v for v,k in enumerate(voc)}
        new_corpus = []
        for doc in self.texts:
            new_doc = []
            for word in doc:
                word_idx = voc_ind[word]
                new_doc.append(word_idx)
            new_corpus.append(new_doc)

        hlda_model = py3hlda.sampler.HierarchicalLDA(new_corpus, voc, alpha=alpha, gamma=gamma, eta=eta, num_levels=num_levels)

        with open('static/HLDA.txt', 'w') as txt_file:
            hlda_model.estimate(n_samples, display_topics=display_topics, n_words=n_words, with_weights=with_weights, txt=txt_file)

import csv
import os

from gensim import corpora, models
from gensim.test.utils import datapath

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag

import spacy

import string

# The following two lines are used for downloading nltk
# import nltk
# nltk.download()


def retrieve_pos_tag(word):
    treebank_tag = pos_tag([word])[0][1]

    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN


def clean_text(review):
    # Remove all non-ASCII character
    review = ''.join(char for char in review if ord(char) < 128)

    # Split the text into words
    review = word_tokenize(review)

    # Convert to lower-case
    review = [word.lower() for word in review]

    # Remove punctuation
    table = str.maketrans('', '', string.punctuation)
    review = [word.translate(table) for word in review]

    # Filter out all non-charac and all single-charac word
    review = [word for word in review if word.isalpha() and len(word) >= 2]

    # Filter out all stop word
    stop_words = set(stopwords.words('english'))
    review = [word for word in review if word not in stop_words]

    # Lemmatizer twice by Spacy
    lemma_spacy = spacy.load('en_core_web_sm', disable=['parser', 'ner'])
    review = lemma_spacy(" ".join(review))

    review = [x.lemma_ for x in review if x.lemma_ != '-PRON-']

    # Lemmatizer by wordnet
    lemma_wordnet = WordNetLemmatizer()
    review = [lemma_wordnet.lemmatize(word, retrieve_pos_tag(word)) for word in review]

    #print(" ".join(review))

    return " ".join(review)


def lda_train(train_txt, num_topic):
    print("Start Training LDA model...")
    dict_w = corpora.Dictionary(train_txt)

    corpus = [dict_w.doc2bow(d) for d in train_txt]

    lda_model = models.LdaModel(corpus,
                                num_topics=num_topic,
                                id2word=dict_w,
                                alpha='auto',
                                eta='auto',
                                passes=20,
                                chunksize=200,
                                minimum_probability=0)

    print("------------------------------------")

    return lda_model


def places_reviews(file_name):
    res = {}
    r_count = 1

    with open(file_name, "r", encoding="utf8") as handle:
        rfile = csv.reader(handle)

        cols = next(rfile)

        r_col, t_col, p_col = cols.index("review"), cols.index("title"), cols.index("place")

        for each_row in rfile:
            cleaned_review = clean_text(each_row[r_col] + '. ' + each_row[t_col])
            res[each_row[p_col]] = res.get(each_row[p_col], []) + [cleaned_review]

            print("Now Dealing with", file_name.split('/')[1], ' row ', r_count)
            r_count += 1

    return res


def main():
    if not os.path.isdir('LDA_model'):
        os.mkdir('LDA_model')

    if os.path.exists('LDA_model/train_cleaned_text.csv'):
        print("Start Loading Cleaned Data...")
        with open('LDA_model/train_cleaned_text.csv', 'r') as handle:
            rfile = csv.reader(handle, delimiter='|')
            trained_data = {}

            for each_row in rfile:
                trained_data[each_row[0]] = trained_data.get(each_row[0], []) + [each_row[1]]

            trained_data = trained_data.values()

            trained_data = [(" ".join(x)).split() for x in trained_data]
    else:
        print("Start Cleaning Input Data...")
        training_files = [
            'tripadvisor_attractions_Miami.csv',
            'tripadvisor_attractions_Chicago.csv',
            'tripadvisor_attractions_SanDiego.csv',
            'tripadvisor_attractions_Washington.csv'
        ]

        folder_name = 'trip_advisor_crawler/'

        trained_data = []

        for each_f in training_files:
            res = places_reviews(folder_name+each_f)

            with open('LDA_model/train_cleaned_text.csv', 'a', newline='') as whandle:
                spamwriter = csv.writer(whandle, delimiter='|')

                for k, v in res.items():
                    trained_data.append((" ".join(v)).split())
                    #print((" ".join(v)).split())
                    for each_review in v:
                        spamwriter.writerow([k, each_review])

    print("------------------------------------")

    # Not only consider the single word, but also frequent bigram
    #print("Start Finding Frequent Bigram...")
    #bi_model = Phrases(trained_data, min_count=1, threshold=5)

    #population = [bi_model[x] for x in trained_data]

    # Remove city name
    cleaned_population = []

    for each_doc in trained_data:
        tmp = [each_term for each_term in each_doc if each_term != 'san' and each_term != 'diego' and
               each_term != 'miami' and
               each_term != 'chicago' and
               each_term != 'dc' and each_term != 'washington' and
               each_term != 'nt']
        cleaned_population.append(tmp)

    population = cleaned_population
    #print("------------------------------------")

    lda_model = lda_train(population, 6)

    for i in range(lda_model.num_topics):
        print(lda_model.print_topics()[i])

    # Save to disk
    temp_file = datapath(os.getcwd() + "\\LDA_model\\lda_trained_model")
    lda_model.save(temp_file)





if __name__ == '__main__':
    main()

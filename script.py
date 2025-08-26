import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
nltk.download('punkt_tab')
import re
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
import matplotlib.pyplot as plt
import seaborn as sns
import keras
from keras._tf_keras.keras.layers import Dropout,LSTM,Dense,Embedding
from keras._tf_keras.keras.preprocessing.text import Tokenizer
import pickle

def plot_train_label_graphs(y_train):
    # Plot a bar graph for train label frequency

    #print(y_train["toxic"])


    plt.figure(figsize=(10,6))
    x = ["Toxic", "Severe Toxic", "Obscene", "Threat", "Insult", "Identity Hate"]
    y = [y_train["toxic"].sum(), y_train["severe_toxic"].sum(), y_train["obscene"].sum(), y_train["threat"].sum(), y_train["insult"].sum(), y_train["identity_hate"].sum()]
    plt.bar(x,y)
    #plt.ylim(0, 159570)
    plt.xlabel("Training Label")
    plt.ylabel("Frequency - Number of Occurrences")
    plt.show()

    

    # Plot correlation matrix for train label set
    plt.figure(figsize=(10,6))
    sns.heatmap(y_train.corr(method='pearson'), annot=True)
    #print(y_train.corr(method='pearson'))
    plt.show()

    # Plot pairwise label correlations to visualize relationships between toxicity categories

def sanitizer(s):
    # s represents a list of word strings that are tokenized already (ONE COMMENT)
    new = []
    for word in s:
        word = str(word)
        if (len(word) > 1 or word.lower() == 'i' or word.lower() == 'k') and word != "" and word != ' ' and word != "`" and word != "``" and word != 1:
            # eliminate all non-alphabetical or non-number characters
            # no more punctuation or leading/trailing spaces
            # use of regular expression
            cleaned_word = re.sub(r'[\W_]+', '', word.strip())
            if cleaned_word != "":
                new.append(cleaned_word.strip())
    # return the tokenized comment
    return new

## Data and Feature-engineering
def eda():
    df = pd.read_csv("Data/dataset.csv")

    # Feature engineering

    # Use a NLTK tokenizer to tokenize the data
    # split text into individual words or tokesn
    # use spacy or other token izers to handle punctuation

    #df["tokenized_comment_text"] = g.tokenize(str(df["comment_text"]), ' ')
    #print(df["tokenized_comment_text"])
    df["tokenized_comment_text"] = df["comment_text"].apply(word_tokenize)
    print('Tokenization Complete')
    df["tokenized_comment_text"] = df["tokenized_comment_text"].apply(sanitizer)
    df["comment_text"] = df["tokenized_comment_text"]
    df = df.drop(columns=["tokenized_comment_text", "id"])
    df["comment_text"] = df["comment_text"].apply(' '.join)
    print("Pre-processing complete!")
    #df.to_csv("delete_me.csv")

    # Perform TF-IDF transformation

    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(df["comment_text"])

    vectorizer.get_feature_names_out()
    #print(X.shape)
    #df["vectorized_comments"] = pd.Series(X)

    # Split dataset into test/train set

    y = df["toxic"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.25)

    plot_train_label_graphs(df[["toxic","severe_toxic","obscene","threat","insult","identity_hate"]])

    return X_train, X_test, y_train, y_test

## Modeling and Evaluation for the basic models
def model_attempts(X_train, X_test, y_train, y_test):

    # Try multinomial naive bayes to detect toxic comments

    print("Multinomial Naive Bayes Starting:")
    multinomial_naive_bayes_model = MultinomialNB()
    multinomial_naive_bayes_model.fit(X_train, y_train)
    multinomial_naive_bayes_output = multinomial_naive_bayes_model.predict(X_test)
    print("Multinomial Naive Bayes Complete. Accuracy:",str(100*accuracy_score(y_test, multinomial_naive_bayes_output, normalize=True))+"%.\n"+str(accuracy_score(y_test, multinomial_naive_bayes_output, normalize=False)),"toxic comments were correctly classified.")
    print(classification_report(y_test, multinomial_naive_bayes_output))

    # Try Logistic regression to detect toxic comments

    print("Logistic Regression Starting:")
    logistic_regression_model = LogisticRegression()
    logistic_regression_model.fit(X_train, y_train)
    logistic_output = logistic_regression_model.predict(X_test)
    print("Logistic Regression Complete. Accuracy:",str(100*accuracy_score(y_test, logistic_output, normalize=True))+"%.\n"+str(accuracy_score(y_test, logistic_output, normalize=False)),"toxic comments were correctly classified.")
    print(classification_report(y_test, logistic_output))

    # Try linear support vector classifier to detect toxic comments

    print("Linear Support Vector Classifier Starting:")
    SVC_model = LinearSVC()
    SVC_model.fit(X_train, y_train)
    SVC_output = SVC_model.predict(X_test)
    print("Linear Support Vector Classifier Complete. Accuracy:",str(100*accuracy_score(y_test, SVC_output, normalize=True))+"%.\n"+str(accuracy_score(y_test, SVC_output, normalize=False)),"toxic comments were correctly classified.")
    print(classification_report(y_test, SVC_output))

    # Plot confusion matrix for each model

    plt.figure(figsize=(10,6))
    sns.heatmap(confusion_matrix(y_test, multinomial_naive_bayes_output))
    plt.xlabel("Predicted Classifications")
    plt.ylabel("Actual Classifications")
    plt.title("Multinomial Naive Bayes Confusion Matrix")
    plt.show()

    plt.figure(figsize=(10,6))
    sns.heatmap(confusion_matrix(y_test, logistic_output))
    plt.xlabel("Predicted Classifications")
    plt.ylabel("Actual Classifications")
    plt.title("Logistic Regression Confusion Matrix")
    plt.show()

    plt.figure(figsize=(10,6))
    sns.heatmap(confusion_matrix(y_test, SVC_output))
    plt.xlabel("Predicted Classifications")
    plt.ylabel("Actual Classifications")
    plt.title("SVC Confusion Matrix")
    plt.show()

    # Use voting classifier or other enssemble method to combine predictions from each model and make a collective decision

    m_o = list(multinomial_naive_bayes_output)
    l_o = list(logistic_output)
    s_o = list(SVC_output)
    combination = []
    for i in range(len(list(y_test))):
        if l_o[i] == s_o[i] and m_o[i] == s_o[i]:
            combination.append(l_o[i])
        elif l_o[i] == s_o[i]:
            combination.append(l_o[i])
        elif l_o[i] == m_o[i]:
            combination.append(l_o[i])
        else:
            combination.append(m_o[i])

    print("Combination with majority rule from the three models has been computed. Accuracy:",str(100*accuracy_score(y_test, combination, normalize=True))+"%.\n"+str(accuracy_score(y_test, combination, normalize=False)),"toxic comments were correctly classified.")
    print(classification_report(y_test, combination))
    plt.figure(figsize=(10,6))
    sns.heatmap(confusion_matrix(y_test, combination))
    plt.xlabel("Predicted Classifications")
    plt.ylabel("Actual Classifications")
    plt.title("Combination")
    plt.show()
    

    # Plot graph comparing accuracies

    plt.figure(figsize=(10,6))
    plt.bar(["Multinomial Naive Bayes", "Logistic Regression", "Linear Support Vector Classifier"], [accuracy_score(y_test, multinomial_naive_bayes_output, normalize=True)*100, accuracy_score(y_test, logistic_output, normalize=True)*100, accuracy_score(y_test, SVC_output, normalize=True)*100])
    plt.xlabel("Model")
    plt.ylabel("Accuracy %")
    plt.ylim(90, 100)
    plt.show()

## The LSTM works best, so use the LSTM model for the final deployment
def final_lstm():

    # Do preprocessing again
    df = pd.read_csv("Data/dataset.csv")
    df["tokenized_comment_text"] = df["comment_text"].apply(word_tokenize)
    print('Tokenization Complete')
    df["tokenized_comment_text"] = df["tokenized_comment_text"].apply(sanitizer)
    df["comment_text"] = df["tokenized_comment_text"]
    df = df.drop(columns=["tokenized_comment_text", "id"])
    df["comment_text"] = df["comment_text"].apply(' '.join)
    print("Pre-processing complete!")

    # Apply the tokenizer on the feature set with a maximum number of words = 10000
    tokenizer = Tokenizer(num_words=10000)
    tokenizer.fit_on_texts(df["comment_text"])
    X = keras.utils.pad_sequences(tokenizer.texts_to_sequences(df["comment_text"]), maxlen=10)
    
    # Use GloVe word embedding

    # From geeksforgeeks documentation on glove embedding for nlp models
    # Create the embedding matrix
    embedding_matrix_vocab = np.zeros((len(tokenizer.word_index)+1, 300))
    # ill use the most broad file because there are so many comments and language is quite varied
    with open('glove.6B.300d.txt', encoding="utf8") as f:
        for line in f:
            word, *vector = line.split()
            if word in tokenizer.word_index:
                idx = tokenizer.word_index[word]
                embedding_matrix_vocab[idx] = np.array(
                    vector, dtype=np.float32)[:300]
    # end citation

    # label
    y = df["toxic"]

    # initialize the model
    LSTM_model = keras.Sequential()

    # Add the embedding, use the vocab as the weights
    LSTM_model.add(Embedding(
        input_dim=len(tokenizer.word_index)+1,
        # the text file I used has 300 dimensions
        output_dim=300,
        # Three dimensional weights but just one layer in the third dimension
        weights=[embedding_matrix_vocab],
        # Keep maximum sequence length=10
        input_length=10
    ))

    LSTM_model.add(LSTM(units=50, return_sequences=True))
    LSTM_model.add(Dropout(0.2))
    LSTM_model.add(LSTM(units=50, return_sequences=False))
    LSTM_model.add(Dropout(0.2))

    # Keep sigmoid as activation function
    LSTM_model.add(Dense(units=1, activation="sigmoid"))

    # Ill use binary crossentropy because it's either a 0 or 1
    LSTM_model.compile(optimizer='adam', loss='binary_crossentropy')
        
    # use the whole dataset; no need for splitting
    LSTM_model.fit(X, y, epochs=5, batch_size=5000)

    # These are my 5 comments for a test case
    test_comments = [
        # obvious non-toxic
        "I love how cleanly and clearly you have written this essay!",
        # meant to look toxic but the comment is not toxic
        "the word 'retarded' has a long and varied history in the english langauge",
        # obvious toxic but from only one word
        "FUCK!",
        # obvious toxic
        "I HATE YOU!",
        # obvious toxic but with mild language/insult
        "you're such a stupid head!"
    ]
    #test_labels = [0,0,1,1,1]
    
    with open("tokenizer.pkl", "wb") as f:
        pickle.dump(tokenizer, f)

    # Tokenize and sequence the custom data before passing it to the model
    fd = pd.DataFrame({"test_comments":test_comments})
    fd["test_comments"] = fd["test_comments"].apply(word_tokenize)
    fd["test_comments"] = fd["test_comments"].apply(sanitizer)
    fd["test_comments"] = fd["test_comments"].apply(' '.join)

    # tokenize it
    tokenizer2 = Tokenizer(num_words=10000)
    tokenizer2.fit_on_texts(fd["test_comments"])
    X = keras.utils.pad_sequences(tokenizer.texts_to_sequences(fd["test_comments"]), maxlen=10)

    # predict
    test_output = list(LSTM_model.predict(X))

    # print the passed comment along with the toxicity probability
    for i in range(len(test_comments)):
        print("Comment:\n",test_comments[i],"\nProbability of Toxicity:",str(test_output[i]*100)+"%.")
        print("\n")

    # Saved the trained model into my directory
    LSTM_model.save('lstm_model.h5')

X_train, X_test, y_train, y_test = eda()
model_attempts(X_train, X_test, y_train, y_test)
final_lstm()
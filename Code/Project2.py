import numpy as np
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
import pandas as pd
import re
from better_profanity import profanity

data = pd.read_csv('train.csv')
X = data[['Id', 'Tweet', 'following', 'followers', 'actions', 'is_retweet', 'location']].values
Y = data[['Type']]
#tweets = data[['Tweet']].values
X = np.array(X)
Y = np.array(Y)

y= []
for i in range(len(Y)):
    if (Y[i][0] == 'Quality'):
        y.append(0)
    elif (Y[i][0] == 'Spam'):
        y.append(1)

def numOfUrl(string):
    url = re.findall(r'(https?://[^\s]+)', string)
    counter = (len(url))
    return counter

def numOfHashtags(string):
    hashtags = re.findall("#([a-zA-Z0-9]{1,15})", string)
    counter = (len(hashtags))
    return counter

def numOfMentions(string):
    mentions = re.findall("@([a-zA-Z0-9]{1,15})", string)
    counter = (len(mentions))
    return counter

def numOfWords(string):
    word_list = string.split()

    number_of_words = len(word_list)
    return number_of_words

def numOfChars(string):
    num = len(string)
    return num

def numOfProfanity(string):
    word_list = string.split()
    number_of_profanity_words=0
    for p in range(len(word_list)):
        if(profanity.contains_profanity(word_list[p])):
            number_of_profanity_words=number_of_profanity_words +1
    return number_of_profanity_words

def rate(following, followers):
    r = 0.0
    if(following == 0 and followers ==0) :
        r = -1
    else:
        r = followers/(following + followers)
    return r
def MLP(features , y):
    clf = MLPClassifier()
    X_train, X_test, y_train, y_test =  train_test_split(features , y, test_size=0.2)
    clf.fit(X_train, y_train)
    prediction = clf.predict(X_test)
    accuracy = metrics.accuracy_score(y_test, prediction)
    print("accuracy of Neuron networks is: %", round(accuracy * 100, 2))

def GNB(features , y):
    features, y = load_iris(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(features, y, test_size=0.2)
    gnb = GaussianNB()
    prediction = gnb.fit(X_train, y_train).predict(X_test)
    accuracy = metrics.accuracy_score(y_test, prediction)
    print("accuracy of Naive Bayes is: %", round(accuracy * 100, 2))

def RFDT(features, y):
    knn = RandomForestClassifier()
    X_train, X_test, y_train, y_test =  train_test_split(features , y, test_size=0.2)
    knn.fit(X_train, y_train)
    prediction = knn.predict(X_test)
    accuracy = metrics.accuracy_score(y_test, prediction)
    print("accuracy of Random Forest Descision Tree is: %", round(accuracy * 100, 2))

features = []
for i in range(len(X)):
   f = []
   f.append(numOfUrl(X[i][1]))
   f.append(numOfHashtags(X[i][1]))
   f.append(numOfMentions(X[i][1]))
   f.append(numOfWords(X[i][1]))
   f.append(numOfChars(X[i][1]))
   f.append(numOfProfanity(X[i][1]))
   flag = 0
   if (np.isnan(X[i][2])):
        f.append(0)
        flag = 1
   else:
        f.append(X[i][2])

   if (np.isnan(X[i][3])):
        f.append(0)
        flag = 1
   else:
        f.append(X[i][3])

   if (flag == 0):
        f.append(rate(X[i][2],X[i][3]))
   else:
        f.append(0)

   if (np.isnan(X[i][4])):
        f.append(0)
   else:
        f.append(X[i][4])

   if (np.isnan(X[i][5])):
        f.append(0)
   else:
        f.append(X[i][5])
   features.append(f)

RFDT(features,y)
GNB(features,y)
MLP(features,y)

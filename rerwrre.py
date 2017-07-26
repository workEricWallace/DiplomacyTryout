from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import urllib

import numpy as np
#imcport tensorflow as tf
import sklearn.model_selection as sk

#throw out data that is during the betrayal and 1 friendship before it (if it was a betrayal)
def trimDataSet(diplomacy):
    for entry in diplomacy:
        if (entry ['betrayal'] == True):
            del entry['seasons'][len(entry['seasons']) - 1]
            del entry['seasons'][len(entry['seasons']) -1]
    return diplomacy


if __name__ == "__main__":
    #load diplomacy data from json
    import json
    from io import open
    with open ("diplomacy_data.json", "r") as f:
        diplomacy = json.load(f)

    outputData = []
    features = []
    diplomacy = trimDataSet(diplomacy)

    sentimentPositive = 0
    numSentences = 0
    negativeSentiment = 0
    numWords = 0
    DMFuture = 0
    DMContingency = 0
    DMExpansion = 0
    DMComparison = 0

    for entry in diplomacy:
        if (entry ['betrayal'] == True):
            for x in range (len(entry['seasons'])):
                outputData.append(1)
        else:
            for x in range (len(entry['seasons'])):
                outputData.append(0)
        for season in entry['seasons']:
            for betrayerMessage in season['messages']['betrayer']:
                numSentences = numSentences + betrayerMessage['n_sentences']
                numWords = numWords + betrayerMessage['n_words']
                negativeSentiment = negativeSentiment + betrayerMessage['sentiment']['negative']
                sentimentPositive = sentimentPositive + betrayerMessage['sentiment']['positive']

                dict = betrayerMessage['lexicon_words']
                if 'disc_temporal_future' in dict:
                    DMFuture = DMFuture + (len(dict['disc_temporal_future']))
                if 'disc_expansion' in dict:                
                    DMExpansion = DMExpansion + (len(dict['disc_expansion']))
                if 'disc_comparison' in dict:
                    DMComparison = DMComparison + (len(dict['disc_comparison']))
                if 'disc_temporal_contingency' in dict:
                    DMContingency = DMContingency + (len(dict['disc_temporal_contingency']))

            features.append(numSentences)
            features.append(numWords)
            
            if (numSentences != 0): 
                features.append(sentimentPositive / numSentences)
                features.append(negativeSentiment / numSentences)
                features.append(DMFuture / numSentences)
                features.append(DMExpansion / numSentences)
                features.append(DMComparison / numSentences)
                features.append(DMContingency / numSentences)

            else:
                features.append(0)
                features.append(0)
                features.append(0)
                features.append(0)
                features.append(0)
                features.append(0)
            sentimentPositive = 0
            numSentences = 0
            negativeSentiment = 0
            numWords = 0
            DMFuture = 0
            DMContingency = 0
            DMExpansion = 0
            DMComparison = 0

            

    
  #break data into train and testing sets, with 80% for test set
    inputData =  np.asarray(features)
    print(inputData)
    print(len(features))
    print(len(outputData))
    inputData = np.reshape(inputData, (8,2794))
    
 # outputData = 
 # X_train, X_test, y_train, y_test = train_test_split(features, outputData, test_size=0.20, random_state=18)

  # break data into train and test set (maybe use k folds like they did) 
  #TODO cross validation




# output data is true or false (make it 1 or 0)


#GOAL 1: Given the current seasons, predict support or betray
# baseline test multivariate logistic regression


#features

# num words
# 
# 



# baseline 2 neural network



# baseline 3 use bag of words or something


# test 4 rnn model (look at their paper suggestion that is hidden)


# TODO, also consider difference in number of sentences as a feature? or difference in other features??
# try sentiment not as a percent of sentences
#try word embeddings not number of discourse markers

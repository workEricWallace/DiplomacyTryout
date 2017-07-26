from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import urllib

import numpy as np
import tensorflow as tf
import sklearn.model_selection as sk


if __name__ == "__main__":
    #load diplomacy data from json
    import json
    from io import open
    with open ("diplomacy_data.json", "r") as f:
        diplomacy = json.load(f)

    outputData = []
    features = []
    for entry in diplomacy:
        if (entry ['betrayal'] == betrayalFlag):
            outputData = outputData.append(outputData, 1)
        else:
            outputData = outputData.append(outputData, 0)
        for season in entry['seasons']:
            for betrayerMessage in season['messages']['betrayer']:
                features = features.append(features, (betrayerMessage['sentiment']['positive'] / betrayerMessage['n_sentences']))
                features = features.append(features, betrayerMessage['n_sentences'])
                features = features.append(features, betrayerMessage['n_words'])
                features = features.append(features, (betrayerMessage['sentiment']['negative'] / betrayerMessage['n_sentences']))

                dict = Messages['lexicon_words']
                if 'disc_temporal_future' in dict:
                    features = features.append(features, (len(dict['disc_temporal_future']) / betrayerMessage['n_sentences']))
                if 'disc_expansion' in dict:
                    features = features.append(features, (len(dict['disc_expansion']) / betrayerMessage['n_sentences']))
                if 'disc_comparison' in dict:
                    features = features.append(features, (len(dict['disc_comparison']) / betrayerMessage['n_sentences']))
                if 'disc_temporal_contingency' in dict:
                    features = features.append(features, (len(dict['disc_temporal_contingency']) / betrayerMessage['n_sentences']))
                
    
  #break data into train and testing sets, with 80% for test set
    features = np.arrange(4000).reshape((8,500))
    
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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import urllib

import numpy as np
import tensorflow as tf
import sklearn.model_selection as sk
import pandas as pd

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




# preprocess data, select features, and prepare into tensors for use in training with TF
def input_fn(df):
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
                if 'disc_contingency' in dict:
                    DMContingency = DMContingency + (len(dict['disc_contingency']))

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
    inputData = np.reshape(inputData, (2794,8))
  
    X_train, X_test, y_train, y_test = sk.train_test_split(inputData, outputData, test_size=0.20, random_state=18)
    for x in range(500):
        print(X_train[x])
        print(y_train[x])
  # break data into train and test set (maybe use k folds like they did) 
  #TODO cross validation

     #create a dictionary
     inputColumns = {k : tf.constant(df[k]) 
                         for k in range (8)}




# def build_estimator(model_dir, model_type):
#   """Build an estimator."""
#   # Sparse base columns.
#   gender = tf.contrib.layers.sparse_column_with_keys(column_name="gender",
#                                                      keys=["female", "male"])
#   education = tf.contrib.layers.sparse_column_with_hash_bucket(
#       "education", hash_bucket_size=1000)
#   relationship = tf.contrib.layers.sparse_column_with_hash_bucket(
#       "relationship", hash_bucket_size=100)
#   workclass = tf.contrib.layers.sparse_column_with_hash_bucket(
#       "workclass", hash_bucket_size=100)
#   occupation = tf.contrib.layers.sparse_column_with_hash_bucket(
#       "occupation", hash_bucket_size=1000)
#   native_country = tf.contrib.layers.sparse_column_with_hash_bucket(
#       "native_country", hash_bucket_size=1000)

#   # Continuous base columns.
#   age = tf.contrib.layers.real_valued_column("age")
#   education_num = tf.contrib.layers.real_valued_column("education_num")
#   capital_gain = tf.contrib.layers.real_valued_column("capital_gain")
#   capital_loss = tf.contrib.layers.real_valued_column("capital_loss")
#   hours_per_week = tf.contrib.layers.real_valued_column("hours_per_week")

#   # Transformations.
#   age_buckets = tf.contrib.layers.bucketized_column(age,
#                                                     boundaries=[
#                                                         18, 25, 30, 35, 40, 45,
#                                                         50, 55, 60, 65
#                                                     ])

#   # Wide columns and deep columns.
#   wide_columns = [gender, native_country, education, occupation, workclass,
#                   relationship, age_buckets,
#                   tf.contrib.layers.crossed_column([education, occupation],
#                                                    hash_bucket_size=int(1e4)),
#                   tf.contrib.layers.crossed_column(
#                       [age_buckets, education, occupation],
#                       hash_bucket_size=int(1e6)),
#                   tf.contrib.layers.crossed_column([native_country, occupation],
#                                                    hash_bucket_size=int(1e4))]
#   deep_columns = [
#       tf.contrib.layers.embedding_column(workclass, dimension=8),
#       tf.contrib.layers.embedding_column(education, dimension=8),
#       tf.contrib.layers.embedding_column(gender, dimension=8),
#       tf.contrib.layers.embedding_column(relationship, dimension=8),
#       tf.contrib.layers.embedding_column(native_country,
#                                          dimension=8),
#       tf.contrib.layers.embedding_column(occupation, dimension=8),
#       age,
#       education_num,
#       capital_gain,
#       capital_loss,
#       hours_per_week,
#   ]

#   if model_type == "wide":
#     m = tf.contrib.learn.LinearClassifier(model_dir=model_dir,
#                                           feature_columns=wide_columns)
#   elif model_type == "deep":
#     m = tf.contrib.learn.DNNClassifier(model_dir=model_dir,
#                                        feature_columns=deep_columns,
#                                        hidden_units=[100, 50])
#   else:
#     m = tf.contrib.learn.DNNLinearCombinedClassifier(
#         model_dir=model_dir,
#         linear_feature_columns=wide_columns,
#         dnn_feature_columns=deep_columns,
#         dnn_hidden_units=[100, 50],
#         fix_global_step_increment_bug=True)
#   return m


# def input_fn(df):
#   """Input builder function."""
#   # Creates a dictionary mapping from each continuous feature column name (k) to
#   # the values of that column stored in a constant Tensor.
#   continuous_cols = {k: tf.constant(df[k].values) for k in CONTINUOUS_COLUMNS}
#   # Creates a dictionary mapping from each categorical feature column name (k)
#   # to the values of that column stored in a tf.SparseTensor.
#   categorical_cols = {
#       k: tf.SparseTensor(
#           indices=[[i, 0] for i in range(df[k].size)],
#           values=df[k].values,
#           dense_shape=[df[k].size, 1])
#       for k in CATEGORICAL_COLUMNS}
#   # Merges the two dictionaries into one.
#   feature_cols = dict(continuous_cols)
#   feature_cols.update(categorical_cols)
#   # Converts the label column into a constant Tensor.
#   label = tf.constant(df[LABEL_COLUMN].values)
#   # Returns the feature columns and the label.
#   return feature_cols, label


# def train_and_eval(model_dir, model_type, train_steps, train_data, test_data):
#   """Train and evaluate the model."""
#   train_file_name, test_file_name = maybe_download(train_data, test_data)
#   df_train = pd.read_csv(
#       tf.gfile.Open(train_file_name),
#       names=COLUMNS,
#       skipinitialspace=True,
#       engine="python")
#   df_test = pd.read_csv(
#       tf.gfile.Open(test_file_name),
#       names=COLUMNS,
#       skipinitialspace=True,
#       skiprows=1,
#       engine="python")

#   # remove NaN elements
#   df_train = df_train.dropna(how='any', axis=0)
#   df_test = df_test.dropna(how='any', axis=0)

#   df_train[LABEL_COLUMN] = (
#       df_train["income_bracket"].apply(lambda x: ">50K" in x)).astype(int)
#   df_test[LABEL_COLUMN] = (
#       df_test["income_bracket"].apply(lambda x: ">50K" in x)).astype(int)

#   model_dir = tempfile.mkdtemp() if not model_dir else model_dir
#   print("model directory = %s" % model_dir)

#   m = build_estimator(model_dir, model_type)
#   m.fit(input_fn=lambda: input_fn(df_train), steps=train_steps)
#   results = m.evaluate(input_fn=lambda: input_fn(df_test), steps=1)
#   for key in sorted(results):
#     print("%s: %s" % (key, results[key]))


# FLAGS = None


# def main(_):
#   train_and_eval(FLAGS.model_dir, FLAGS.model_type, FLAGS.train_steps,
#                  FLAGS.train_data, FLAGS.test_data)


# if __name__ == "__main__":
#   parser = argparse.ArgumentParser()
#   parser.register("type", "bool", lambda v: v.lower() == "true")
#   parser.add_argument(
#       "--model_dir",
#       type=str,
#       default="",
#       help="Base directory for output models."
#   )
#   parser.add_argument(
#       "--model_type",
#       type=str,
#       default="wide_n_deep",
#       help="Valid model types: {'wide', 'deep', 'wide_n_deep'}."
#   )
#   parser.add_argument(
#       "--train_steps",
#       type=int,
#       default=200,
#       help="Number of training steps."
#   )
#   parser.add_argument(
#       "--train_data",
#       type=str,
#       default="",
#       help="Path to the training data."
#   )
#   parser.add_argument(
#       "--test_data",
#       type=str,
#       default="",
#       help="Path to the test data."
#   )
#   FLAGS, unparsed = parser.parse_known_args()
#   tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)

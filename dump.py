#throw out data that is during the betrayal and 1 friendship before it (if it was a betrayal)
def trimDataSet(diplomacy):
    for entry in diplomacy:
        if (entry ['betrayal'] == True):
            del entry['seasons'][len(entry['seasons']) -1]
            del entry['seasons'][len(entry['seasons']) -1]
    return diplomacy

#collect percentage of positive and negative sentiment during betrayals or no betrayals
def sentimentSentencePercentage(betrayalFlag):
    betrayerPositiveMessageCount=0
    betrayerNegativeMessageCount=0
    betrayerSentenceCount=0
    victimPositiveMessageCount=0
    victimNegativeMessageCount=0
    victimSentenceCount=0
    for entry in diplomacy:
        if (entry ['betrayal'] == betrayalFlag):
            for season in entry['seasons']:
                for betrayerMessage in season['messages']['betrayer']:
                    betrayerPositiveMessageCount = betrayerPositiveMessageCount + betrayerMessage['sentiment']['positive']             
                    betrayerNegativeMessageCount = betrayerNegativeMessageCount + betrayerMessage['sentiment']['negative']             
                    betrayerSentenceCount = betrayerSentenceCount + betrayerMessage['n_sentences']
                for victimMessage in season['messages']['victim']:
                    victimPositiveMessageCount = victimPositiveMessageCount + victimMessage['sentiment']['positive']             
                    victimNegativeMessageCount = victimNegativeMessageCount + victimMessage['sentiment']['negative']             
                    victimSentenceCount = victimSentenceCount + victimMessage['n_sentences']

    betrayerPercentPositive = betrayerPositiveMessageCount / betrayerSentenceCount
    betrayerPercentNegative = betrayerNegativeMessageCount / betrayerSentenceCount

    victimPercentPositive = victimPositiveMessageCount / victimSentenceCount
    victimPercentNegative = victimNegativeMessageCount / victimSentenceCount

    return (betrayerPercentPositive, betrayerPercentNegative, victimPercentPositive, victimPercentNegative)

def sentimentPercentageResults(withBetrayalBetrayerPercentPositive, withBetrayalBetrayerPercentNegative, withBetrayalVictimPercentPositive, withBetrayalVictimPercentNegative,noBetrayalBetrayerPercentPositive, noBetrayalBetrayerPercentNegative, noBetrayalVictimPercentPositive, noBetrayalVictimPercentNegative):
    print ("With Betrayal: Betrayer Percent Positive")
    print (withBetrayalBetrayerPercentPositive)
    print ("With Betrayal: Betrayer Percent Negative")
    print (withBetrayalBetrayerPercentNegative)

    print ("With Betrayal: Victim Percent Positive")
    print (withBetrayalVictimPercentPositive)
    print ("With Betrayal: Victim Percent Negative")
    print (withBetrayalVictimPercentNegative)

    print ("No Betrayal: Betrayer Percent Positive")
    print (noBetrayalBetrayerPercentPositive)
    print ("No Betrayal: Betrayer Percent Negative")
    print (noBetrayalBetrayerPercentNegative)

    print ("No Betrayal: Victim Percent Positive")
    print (noBetrayalVictimPercentPositive)
    print ("No Betrayal: Victim Percent Negative")
    print (noBetrayalVictimPercentNegative)    

  #load diplomacy data from CSVs
  #import the csvs and use jsons or whatever necessary to get the data in the form I want it
  #inputData = 
  #outputData = 
  
  #break data into train and testing sets
  #maybe 80% for training and 20% for test
  # x_train 
  # y_train 
  
  # maybe something like X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.33, random_state=42)

if __name__ == "__main__":

    #import data and read the json
    import json
    from io import open
    with open ("diplomacy_data.json", "r") as f:
        diplomacy = json.load(f)

    diplomacy = trimDataSet(diplomacy)
    
    #double check all attacks were removed
    #for entry in diplomacy:
    #    for season in entry['seasons']:
    #        print (season['interaction']['betrayer'])
    #        if (season['interaction']['betrayer'] == 'attack'):
    #            print ("attack wasn't removed") 
            

    #collect results from sentiment analysis
    (withBetrayalBetrayerPercentPositive, withBetrayalBetrayerPercentNegative, withBetrayalVictimPercentPositive, withBetrayalVictimPercentNegative) = sentimentSentencePercentage(True)
    (noBetrayalBetrayerPercentPositive, noBetrayalBetrayerPercentNegative, noBetrayalVictimPercentPositive, noBetrayalVictimPercentNegative) = sentimentSentencePercentage(False)
    sentimentPercentageResults(withBetrayalBetrayerPercentPositive, withBetrayalBetrayerPercentNegative, withBetrayalVictimPercentPositive, withBetrayalVictimPercentNegative, noBetrayalBetrayerPercentPositive, noBetrayalBetrayerPercentNegative, noBetrayalVictimPercentPositive, noBetrayalVictimPercentNegative)

    #collect results from argumentation and discourse
    

    # collect politeness results

    # collect talkativeness


    #
    
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



# TODO, what about null (not support or betray)

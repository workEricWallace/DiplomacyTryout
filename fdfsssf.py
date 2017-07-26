#throw out data that is during the betrayal and 1 friendship before it (if it was a betrayal)
def trimDataSet(diplomacy):
    for entry in diplomacy:
        if (entry ['betrayal'] == True):
            
            print (entry['seasons'][len(entry['seasons']) -1]['interaction'])
            print (entry['seasons'][len(entry['seasons']) -2]['interaction'])
            del entry['seasons'][len(entry['seasons']) - 1]
            del entry['seasons'][len(entry['seasons']) -1]
    return diplomacy

#collect percentage of positive and negative sentiment during betrayals or no betrayals
def sentimentSentencePercentage(betrayalFlag):
    betrayerPositiveMessageCount = 0
    betrayerNegativeMessageCount = 0
    betrayerSentenceCount = 0
    victimPositiveMessageCount = 0
    victimNegativeMessageCount = 0
    victimSentenceCount = 0
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

#collect average politeness score per message
def politeness(betrayer, betrayalFlag):
    whichParty = 'victim'
    if (betrayer):
        whichParty = 'betrayer'
    politenessCount = 0
    numMessages = 0
    for entry in diplomacy:
        if (entry ['betrayal'] == betrayalFlag):
            for season in entry['seasons']:
                numMessages = numMessages + len(season['messages'])
                for Messages in season['messages'][whichParty]:
                    politenessCount = politenessCount + Messages['politeness']
    return (politenessCount / numMessages)


#collect average number of planning DM
def planning(betrayer, betrayalFlag):
    whichParty = 'victim'
    if (betrayer):
        whichParty = 'betrayer'
    planningCount = 0
    numSentences = 0
    for entry in diplomacy:
        if (entry ['betrayal'] == betrayalFlag):
            for season in entry['seasons']:
                for Messages in season['messages'][whichParty]:
                    numSentences = numSentences + Messages['n_sentences']
                    dict = Messages['lexicon_words']
                    if 'disc_temporal_future' in dict:
                        planningCount = planningCount + len(dict['disc_temporal_future'])
    return (planningCount / numSentences)
    
#collect, print, and analyze talkativeness results
def talkativenessResults(numMessages, averageNumSentences, averageNumWords):
    print ("Number of Messages:")
    print (numMessages)
    print ("Average Number of Sentences")
    print (averageNumSentences)
    print ("Average Number of Words")
    print (averageNumWords)  

#collect, print, and analyze talkativeness results
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

def talkativeness (betrayer, betrayalFlag):
    whichParty = 'victim'
    if (betrayer):
        whichParty = 'betrayer'
    numMessages = 0
    numSentences = 0
    numWords = 0
    for entry in diplomacy:
        if (entry ['betrayal'] == betrayalFlag):
            for season in entry['seasons']:
                numMessages = numMessages + len(season['messages'][whichParty])
                for Messages in season['messages'][whichParty]:
                    numSentences = numSentences + Messages['n_sentences']
                    numWords = numWords + Messages['n_words']
    return (numMessages, numSentences / numMessages, numWords / numSentences)
    
        
    
  

if __name__ == "__main__":

    #import data and read the json
    import json
    from io import open
    with open ("diplomacy_data.json", "r") as f:
        diplomacy = json.load(f)

    betrayalSeasons = 0
    victimSeasons = 0
    for entry in diplomacy:
        if (entry ['betrayal'] == True):
            count1 = count1 + len(entry['seasons'])
        else:
            count2 = count2 + len(entry['seasons'])

    print (count1)
    print (count2)
    #diplomacy = trimDataSet(diplomacy
)
    #double check all attacks were removed
    #for entry in diplomacy:
    #    for season in entry['seasons']:
    #        print (season['interaction']['betrayer'])
    #        if (season['interaction']['betrayer'] == 'attack'):
    #            print ("attack wasn't removed") 
            

    #collect results from sentiment analysis
    # match results in paper
    #(withBetrayalBetrayerPercentPositive, withBetrayalBetrayerPercentNegative, withBetrayalVictimPercentPositive, withBetrayalVictimPercentNegative) = sentimentSentencePercentage(True)
    #(noBetrayalBetrayerPercentPositive, noBetrayalBetrayerPercentNegative, noBetrayalVictimPercentPositive, noBetrayalVictimPercentNegative) = sentimentSentencePercentage(False)
    #sentimentPercentageResults(withBetrayalBetrayerPercentPositive, withBetrayalBetrayerPercentNegative, withBetrayalVictimPercentPositive, withBetrayalVictimPercentNegative, noBetrayalBetrayerPercentPositive, noBetrayalBetrayerPercentNegative, noBetrayalVictimPercentPositive, noBetrayalVictimPercentNegative)

    #collect results from argumentation and discourse

    #planning discourse markers
    print("With Betrayal: Betrayer Planning Avg. Discourse Markers")
    averagePlanning = planning(True, True)
    print(averagePlanning)
    
    print("No Betrayal: Betrayer Planning Avg. Discourse Markers")
    averagePlanning = planning(True, False)
    print(averagePlanning)

    print("With Betrayal: Victim Planning Avg. Discourse Markers")
    averagePlanning = planning(False, True)
    print(averagePlanning)
    
    print("No Betrayal: Victim Planning Avg. Discourse Markers")
    averagePlanning = planning(False, False)
    print(averagePlanning)

    
    
    # collect politeness results
    print("With Betrayal: Betrayer Politeness")
    averagePoliteness = politeness(True, True)
    print(averagePoliteness)
    
    print("No Betrayal: Betrayer Politeness")
    averagePoliteness = politeness(True, False)
    print(averagePoliteness)

    print("With Betrayal: Victim Politeness")
    averagePoliteness = politeness(False, True)
    print(averagePoliteness)
    
    print("No Betrayal: Victim Politeness")
    averagePoliteness = politeness(False, False)
    print(averagePoliteness)
    
    # collect talkativeness
    # might match results, unsure
    #print("With Betrayal: Betrayer Talkativeness")
    #(numMessages, averageNumSentences, averageNumWords) = talkativeness(True, True)
    #talkativenessResults(numMessages, averageNumSentences, averageNumWords)

    #print("No Betrayal: Betrayer Talkativeness")
    #(numMessages, averageNumSentences, averageNumWords) = talkativeness(True, False)
    #talkativenessResults(numMessages, averageNumSentences, averageNumWords)

    #print("With Betrayal: Victim Talkativeness")
    #(numMessages, averageNumSentences, averageNumWords) = talkativeness(False, True)
    #talkativenessResults(numMessages, averageNumSentences, averageNumWords)
    
    #print("No Betrayal: Victim Talkativeness")
    #(numMessages, averageNumSentences, averageNumWords) = talkativeness(False, False)
    #talkativenessResults(numMessages, averageNumSentences, averageNumWords)


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

# TODO, talkativeness seems off (jk it might just be numMessages is the only thing they looked at)

# TODO, politeness is off

# TODO, clean up planning

# TODO, do I need to clean up data with trim or not????

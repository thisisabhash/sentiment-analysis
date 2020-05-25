# Sentiment Analysis


One application of Naïve Bayes classifiers is sentiment analysis, which is a sub-field of AI that 
extracts affective states and subjective information from text. One common use of sentiment analysis 
is to determine if a text document expresses negative or positive feelings. This repo contains a Java implementation
of a Naïve Bayes classifier for categorizing movie reviews as either POSITIVE or NEGATIVE. 
The dataset provided consists of online movie reviews derived from an 
IMDb dataset: https://ai.stanford.edu/~amaas/data/sentiment/ that have been labeled based on the review scores.
A negative review has a score ≤ 4 out of 10, and a positive review has a score ≥ 7 out of 10. 

Each row in the training set and test set files contains one review, where the first word in each line is the class label 
(1 represents POSITIVE and 0 represents NEGATIVE) and the remainder of the line is the review text.

# Execution

`javac *.java`

`java SentimentAnalysis <mode> <trainFilename> [<testFilename> | <K>]`

where trainingFilename and testFilename are the names of the training set and test set files, respectively. 
mode is an integer from 0 to 3, controlling what the program will output. 

When mode is 0 or 1, there are only two arguments, mode and trainFilename; 

when the mode is 2 the third argument is testFilename; 

when mode is 3, the third argument is K, the number of folds used for cross validation. 

For example, the command

`java SentimentAnalysis 2 train.txt test.txt`

should train the classifier using the data in train.txt and print the predicted class for every
review in test.txt

The command

`java SentimentAnalysis 3 train.txt 5`

should perform 5-fold cross-validation on train.txt

# Output 
The output for these four modes are:
0. Prints the number of documents for each label in the training set
1. Prints the number of words for each label in the training set
2. For each instance in test set, prints a line displaying the predicted class and the log
probabilities for both classes
3. Prints the accuracy score for K-fold cross validation

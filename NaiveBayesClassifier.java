import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * Your implementation of a naive bayes classifier. Please implement all four methods.
 */

public class NaiveBayesClassifier implements Classifier {

    /**
     * Trains the classifier with the provided training data and vocabulary size
     */

    private Map<Label, Integer> wordsCountPerLabelMap;
    private Map<Label, Integer> documentsCountPerLabelMap;
    private Map<String, Integer> wordCountForPositiveLabelMap;
    private Map<String, Integer> wordCountForNegativeLabelMap;
    private int vocabularySize;
    private static final int DELTA = 1;

    @Override
    public void train(List<Instance> trainData, int v) {
        // Hint: First, calculate the documents and words counts per label and store them.
        // Then, for all the words in the documents of each label, count the number of occurrences of each word.
        // Save these information as you will need them to calculate the log probabilities later.
        //
        // e.g.
        // Assume m_map is the map that stores the occurrences per word for positive documents
        // m_map.get("catch") should return the number of "catch" es, in the documents labeled positive
        // m_map.get("asdasd") would return null, when the word has not appeared before.
        // Use m_map.put(word,1) to put the first count in.
        // Use m_map.replace(word, count+1) to update the value

        wordsCountPerLabelMap = getWordsCountPerLabel(trainData);
        documentsCountPerLabelMap = getDocumentsCountPerLabel(trainData);
        vocabularySize = v;
        initializeWordCountForLabelMaps(trainData);
    }

    /*
     * Counts the number of words for each label
     */
    @Override
    public Map<Label, Integer> getWordsCountPerLabel(List<Instance> trainData) {
        Map<Label, Integer> result = new HashMap<>();
        int positiveLabelWordCount = 0;
        int negativeLabelWordCount = 0;

        for (Instance review : trainData) {
            if (Label.POSITIVE.equals(review.label)) {
                positiveLabelWordCount += review.words.size();
            } else {
                negativeLabelWordCount += review.words.size();
            }
        }

        result.put(Label.POSITIVE, positiveLabelWordCount);
        result.put(Label.NEGATIVE, negativeLabelWordCount);
        return result;
    }


    /*
     * Counts the total number of documents for each label
     */
    @Override
    public Map<Label, Integer> getDocumentsCountPerLabel(List<Instance> trainData) {
        Map<Label, Integer> result = new HashMap<>();
        int positiveLabelCount = 0;
        int negativeLabelCount = 0;

        for (Instance review : trainData) {
            if (Label.POSITIVE.equals(review.label)) {
                positiveLabelCount++;
            } else {
                negativeLabelCount++;
            }
        }

        result.put(Label.POSITIVE, positiveLabelCount);
        result.put(Label.NEGATIVE, negativeLabelCount);
        return result;
    }

    /**
     * Initializes word count maps for each label and stores them
     */
    private void initializeWordCountForLabelMaps(List<Instance> trainData) {
        wordCountForPositiveLabelMap = new HashMap<>();
        wordCountForNegativeLabelMap = new HashMap<>();

        // loop through reviews and increment word counts
        for (Instance review : trainData) {
            if (Label.POSITIVE.equals(review.label)) {
                for (String word : review.words) {
                    wordCountForPositiveLabelMap.put(word, wordCountForPositiveLabelMap.getOrDefault(word, 0) + 1);
                }
            } else {
                for (String word : review.words) {
                    wordCountForNegativeLabelMap.put(word, wordCountForNegativeLabelMap.getOrDefault(word, 0) + 1);
                }
            }
        }

    }

    /**
     * Returns the prior probability of the label parameter, i.e. P(POSITIVE) or P(NEGATIVE)
     */
    private double p_l(Label label) {
        // Calculate the probability for the label. No smoothing here.
        // Just the number of label counts divided by the number of documents.
        int numberOfDocumentsWithGivenLabel = documentsCountPerLabelMap.get(label);
        int totalNumberOfDocuments = documentsCountPerLabelMap.get(Label.POSITIVE) + documentsCountPerLabelMap.get(Label.NEGATIVE);
        return ((double) numberOfDocumentsWithGivenLabel) / ((double) totalNumberOfDocuments);
    }

    /**
     * Returns the smoothed conditional probability of the word given the label, i.e. P(word|POSITIVE) or
     * P(word|NEGATIVE)
     */
    private double p_w_given_l(String word, Label label) {
        // Calculate the probability with Laplace smoothing for word in class(label)
        int wordCountForGivenLabel;
        if (Label.POSITIVE.equals(label)) {
            wordCountForGivenLabel = wordCountForPositiveLabelMap.getOrDefault(word, 0);
        } else {
            wordCountForGivenLabel = wordCountForNegativeLabelMap.getOrDefault(word, 0);
        }
        int totalWordCountForGivenLabel = wordsCountPerLabelMap.get(label);

        return ((double) wordCountForGivenLabel + (double) DELTA) / ((double) totalWordCountForGivenLabel + (double) (vocabularySize * DELTA));
    }

    /**
     * Classifies an array of words as either POSITIVE or NEGATIVE.
     */
    @Override
    public ClassifyResult classify(List<String> review) {
        // Sum up the log probabilities for each word in the input data, and the probability of the label
        // Set the label to the class with larger log probability

        double g_positive_given_words = Math.log(p_l(Label.POSITIVE));
        double g_negative_given_words = Math.log(p_l(Label.NEGATIVE));
        for (String word : review) {
            g_positive_given_words += Math.log(p_w_given_l(word, Label.POSITIVE));
            g_negative_given_words += Math.log(p_w_given_l(word, Label.NEGATIVE));
        }

        Double g_positive_given_words_object = g_positive_given_words;
        Double g_negative_given_words_object = g_negative_given_words;
        if(g_positive_given_words_object.isNaN() || g_negative_given_words_object.isNaN()){
            ClassifyResult result = new ClassifyResult();
            result.label = Label.POSITIVE;
            Map<Label, Double> resultMap = new HashMap<>();
            resultMap.put(Label.POSITIVE, Double.NEGATIVE_INFINITY);
            resultMap.put(Label.NEGATIVE, Double.NEGATIVE_INFINITY);
            result.logProbPerLabel = resultMap;
            return result;
        }

        Map<Label, Double> resultMap = new HashMap<>();
        resultMap.put(Label.POSITIVE, g_positive_given_words);
        resultMap.put(Label.NEGATIVE, g_negative_given_words);

        Label resultLabel = (g_positive_given_words >= g_negative_given_words) ? Label.POSITIVE : Label.NEGATIVE;
        ClassifyResult result = new ClassifyResult();
        result.label = resultLabel;
        result.logProbPerLabel = resultMap;
        return result;
    }


}

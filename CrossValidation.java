import java.util.ArrayList;
import java.util.List;

public class CrossValidation {
    /*
     * Returns the k-fold cross validation score of classifier clf on training data.
     */
    public static double kFoldScore(Classifier clf, List<Instance> trainData, int k, int v) {
        if (k < 2 || k > trainData.size()) {
            return 0.0;
        }

        double kFoldAccuracy = 0.0;
        for (int i = 0; i < k; i++) {
            int testSetLeftIndex = i * (trainData.size() / k);
            int testSetRightIndex = testSetLeftIndex + (trainData.size() / k);

            List<Instance> kFoldTestData = trainData.subList(testSetLeftIndex, testSetRightIndex);
            List<Instance> kFoldTrainData = new ArrayList<>();
            for (int j = 0; j < trainData.size(); j++) {
                if (!(j >= testSetLeftIndex && j < testSetRightIndex)) {
                    kFoldTrainData.add(trainData.get(j));
                }
            }
            clf.train(kFoldTrainData, v);
            int correctPredictions = 0;
            int totalPredictions = kFoldTestData.size();
            for (Instance instance : kFoldTestData) {
                if (clf.classify(instance.words).label.equals(instance.label)) {
                    correctPredictions++;
                }
            }
            kFoldAccuracy += ((double) correctPredictions) / ((double) totalPredictions);
        }
        return (kFoldAccuracy / k);
    }
}

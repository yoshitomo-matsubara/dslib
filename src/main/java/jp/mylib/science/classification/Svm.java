package jp.mylib.science.classification;

import jp.mylib.science.common.FeatureVector;

import java.util.List;

public abstract class Svm {
    public abstract void train(FeatureVector[] featureVectors);

    public abstract void train(List<FeatureVector> featureVectorList);

    public abstract int predict(FeatureVector featureVector);

    public abstract void reset();

    public abstract double doLeaveOneOutCrossValidation(List<FeatureVector> featureVectorList);

    public abstract double doLeaveOneOutCrossValidation(FeatureVector[] featureVectors);

    public abstract void inputModel(String modelFilePath);

    public abstract void outputModel(String modelFilePath);
}

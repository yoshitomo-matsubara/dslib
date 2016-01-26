package jp.mylib.science.classification;

import jp.mylib.science.common.BasicAlgebra;
import jp.mylib.science.common.FeatureVector;

import java.util.List;

public abstract class SVM
{
    public double[][] calcKernelMatrix(FeatureVector[] featureVectors, String kernelType)
    {
        double[][] kernelMatrix = new double[featureVectors.length][featureVectors.length];
        for(int i=0;i<kernelMatrix.length;i++)
            for(int j=i;j<kernelMatrix[0].length;j++)
            {
                kernelMatrix[i][j] = BasicAlgebra.kernelFunction(featureVectors[i].getAllValues(), featureVectors[j].getAllValues(), kernelType);
                kernelMatrix[j][i] = kernelMatrix[i][j];
            }
        return kernelMatrix;
    }

    public double[][] calcKernelMatrix(FeatureVector[] featureVectors, String kernelType, double[] kernelParams)
    {
        double[][] kernelMatrix = new double[featureVectors.length][featureVectors.length];
        for(int i=0;i<kernelMatrix.length;i++)
            for(int j=i;j<kernelMatrix[0].length;j++)
            {
                kernelMatrix[i][j] = BasicAlgebra.kernelFunction(featureVectors[i].getAllValues(), featureVectors[j].getAllValues(), kernelType, kernelParams);
                kernelMatrix[j][i] = kernelMatrix[i][j];
            }
        return kernelMatrix;
    }

    public abstract void train(FeatureVector[] featureVectors);
    public abstract void train(List<FeatureVector> featureVectorList);
    public abstract int predict(FeatureVector featureVector);
    public abstract double leaveOneOutCrossValidation(List<FeatureVector> featureVectorList);
    public abstract double leaveOneOutCrossValidation(FeatureVector[] featureVectors);
    public abstract void reset();
    public abstract void inputModel(String modelFilePath);
    public abstract void outputModel(String modelFilePath);
}

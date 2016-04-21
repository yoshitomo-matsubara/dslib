package jp.mylib.science.statistics;

import jp.mylib.science.common.BasicAlgebra;
import jp.mylib.science.common.FeatureVector;

import java.util.Arrays;
import java.util.Random;

public class DensityEstimator
{
    public static double calcReachabilityDistance(double[] arrayX, double[] arrayY, double kthDist)
    {
        double dist = BasicAlgebra.calcEuclideanDistance(arrayX, arrayY);
        return (dist > kthDist)? dist : kthDist;
    }

    public static double calcLocalReachabilityDensity(FeatureVector targetVector, FeatureVector[] featureVectors, int k, double kthDist)
    {
        double lrd = 0.0d;
        for(int i=0;i<k;i++)
            lrd += calcReachabilityDistance(targetVector.getAllValues(), featureVectors[i].getAllValues(), kthDist);

        return 1.0d / lrd / (double)k;
    }

    public static double[] optimizeKlParams(FeatureVector[] trainingFeatureVectors, Kernel kernel, double epsilon, double tolerance)
    {
        double[] alphas = new double[trainingFeatureVectors.length];
        double[] ones = new double[alphas.length];
        Random rand = new Random();
        for(int i=0;i<alphas.length;i++)
        {
            ones[i] = 1.0d;
            alphas[i] = rand.nextDouble();
        }

        double[][] kernelMatrix = kernel.calcKernelMatrix(trainingFeatureVectors);
        double[] kernelSumArray = new double[kernelMatrix.length];
        for(int i=0;i<kernelMatrix.length;i++)
        {
            kernelSumArray[i] = 0.0d;
            for(int j=0;j<kernelMatrix[0].length;j++)
                kernelSumArray[i] += kernelMatrix[i][j];

            kernelSumArray[i] /= (double)kernelMatrix[0].length;
        }

        double ip = BasicAlgebra.calcInnerProduct(kernelSumArray, kernelSumArray);
        double diff = Double.MAX_VALUE;
        while(Math.abs(diff) > tolerance)
        {
            double magnitude = BasicAlgebra.calcMagnitude(alphas);
            double[] array = BasicAlgebra.calcVectorDiff(ones, BasicAlgebra.calcMatrixProduct(kernelMatrix, alphas));
            double[] arrayA = BasicAlgebra.calcMatrixProduct(BasicAlgebra.scalarMultiple(epsilon, kernelMatrix), array);
            double[] alphasA = BasicAlgebra.calcVectorSum(alphas, arrayA);
            double ipB = BasicAlgebra.calcInnerProduct(kernelSumArray, alphasA);
            double[] arrayB = BasicAlgebra.scalarMultiple((1.0d - ipB) / ip, kernelSumArray);
            double[] alphasB = BasicAlgebra.calcVectorSum(alphasA, arrayB);
            for(int i=0;i<alphasB.length;i++)
                alphasB[i] = (alphasB[i] > 0.0d)? alphasB[i] : 0.0d;

            double ipC = BasicAlgebra.calcInnerProduct(kernelSumArray, alphasB);
            alphas = BasicAlgebra.scalarMultiple(1.0d / ipC, alphasB);
            diff = magnitude - BasicAlgebra.calcMagnitude(alphas);
        }

        return alphas;
    }

    public static double[] estimateDensityRatioKullbackLeibler(FeatureVector[] trainingFeatureVectors, Kernel kernel, double epsilon, double tolerance, FeatureVector... testFeatureVectors)
    {
        double[] alphas = optimizeKlParams(trainingFeatureVectors, kernel, epsilon, tolerance);
        double[] densityRatios = new double[testFeatureVectors.length];
        for(int i=0;i<testFeatureVectors.length;i++)
        {
            densityRatios[i] = 0.0d;
            for(int j=0;j<trainingFeatureVectors.length;j++)
                densityRatios[i] += alphas[j] * kernel.kernelFunction(testFeatureVectors[i].getAllValues(), trainingFeatureVectors[j].getAllValues());
        }

        return densityRatios;
    }

    public static double[] estimateDensityRatioKullbackLeibler(FeatureVector[] trainingFeatureVectors, Kernel kernel, double[] alphas, FeatureVector... testFeatureVectors)
    {
        double[] densityRatios = new double[testFeatureVectors.length];
        for(int i=0;i<testFeatureVectors.length;i++)
        {
            densityRatios[i] = 0.0d;
            for(int j=0;j<trainingFeatureVectors.length;j++)
                densityRatios[i] += alphas[j] * kernel.kernelFunction(testFeatureVectors[i].getAllValues(), trainingFeatureVectors[j].getAllValues());
        }

        return densityRatios;
    }
}

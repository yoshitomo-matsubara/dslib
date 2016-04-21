package jp.mylib.science.statistics;

import jp.mylib.science.common.BasicAlgebra;
import jp.mylib.science.common.FeatureVector;

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
        for(int i=0;i<alphas.length;i++)
            alphas[i] = ones[i] = 1.0d;

        double[][] kernelMatrix = kernel.calcKernelMatrix(trainingFeatureVectors);
        double[] kernelSumArray = new double[kernelMatrix.length];
        double ip = BasicAlgebra.calcInnerProduct(kernelSumArray, kernelSumArray);
        for(int i=0;i<kernelMatrix.length;i++)
        {
            kernelSumArray[i] = 0.0d;
            for(int j=0;j<kernelMatrix[0].length;j++)
                kernelSumArray[i] += kernelMatrix[i][j];

            kernelSumArray[i] /= (double)kernelMatrix[0].length;
        }

        double diff = Double.MAX_VALUE;
        while(Math.abs(diff) > tolerance)
        {
            double magnitude = BasicAlgebra.calcMagnitude(alphas);
            double[] arrayA = BasicAlgebra.calcVectorDiff(ones, BasicAlgebra.calcMatrixProduct(kernelMatrix, alphas));
            arrayA = BasicAlgebra.calcMatrixProduct(BasicAlgebra.scalarMultiple(epsilon, kernelMatrix), arrayA);
            alphas = BasicAlgebra.calcVectorSum(alphas, arrayA);
            double ipB = BasicAlgebra.calcInnerProduct(kernelSumArray, alphas);
            double[] arrayB = BasicAlgebra.scalarMultiple((1.0d - ipB) / ip, kernelSumArray);
            alphas = BasicAlgebra.calcVectorSum(alphas, arrayB);
            for(int i=0;i<alphas.length;i++)
                alphas[i] = (alphas[i] > 0.0d)? alphas[i] : 0.0d;

            double ipC = BasicAlgebra.calcInnerProduct(kernelSumArray, alphas);
            alphas = BasicAlgebra.scalarMultiple(1.0d / ipC, alphas);
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

    public static double[] estimateDensityRatioKullbackLeibler(FeatureVector[] trainingFeatureVectors, FeatureVector[] testFeatureVectors, Kernel kernel, double[] alphas)
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

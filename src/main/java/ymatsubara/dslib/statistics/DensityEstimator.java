package ymatsubara.dslib.statistics;

import ymatsubara.dslib.common.BasicAlgebra;
import ymatsubara.dslib.common.FeatureVector;

import java.util.Random;

public class DensityEstimator {
    public static double calcReachabilityDistance(double[] arrayX, double[] arrayY, double kthDist) {
        double dist = BasicAlgebra.calcEuclideanDistance(arrayX, arrayY);
        return (dist > kthDist) ? dist : kthDist;
    }

    public static double calcLocalReachabilityDensity(FeatureVector targetVector, FeatureVector[] featureVectors, int k, double kthDist) {
        double lrd = 0.0d;
        for (int i = 0; i < k; i++) {
            lrd += calcReachabilityDistance(targetVector.getAllValues(), featureVectors[i].getAllValues(), kthDist);
        }
        return 1.0d / lrd / (double) k;
    }

    public static double[] optimizeKlParams(FeatureVector[] trainingFeatureVectors, FeatureVector[] testFeatureVectors, Kernel kernel, double epsilon, double tolerance) {
        double[] alphas = new double[trainingFeatureVectors.length];
        Random rand = new Random();
        for (int i = 0; i < alphas.length; i++) {
            alphas[i] = rand.nextDouble();
        }

        double[][] kernelMatrix = kernel.calcKernelMatrix(trainingFeatureVectors);
        double[] kernelSumArray = new double[trainingFeatureVectors.length];
        for (int i = 0; i < kernelSumArray.length; i++) {
            kernelSumArray[i] = 0.0d;
            for (int j = 0; j < testFeatureVectors.length; j++) {
                kernelSumArray[i] += kernel.kernelFunction(trainingFeatureVectors[i].getAllValues(), testFeatureVectors[j].getAllValues());
            }
            kernelSumArray[i] /= (double) kernelSumArray.length;
        }

        double btb = BasicAlgebra.calcInnerProduct(kernelSumArray, kernelSumArray);
        double diff = Double.MAX_VALUE;
        while (Math.abs(diff) > tolerance) {
            double magnitude = BasicAlgebra.calcMagnitude(alphas);
            double[] matProd = BasicAlgebra.calcMatrixProduct(kernelMatrix, alphas);
            double[] array = new double[matProd.length];
            for (int i = 0; i < array.length; i++) {
                array[i] = 1.0d / matProd[i];
            }

            double[] arrayA = BasicAlgebra.calcMatrixProduct(BasicAlgebra.scalarMultiple(epsilon, BasicAlgebra.transposeMatrix(kernelMatrix)), array);
            double[] alphasA = BasicAlgebra.calcVectorSum(alphas, arrayA);
            double ipB = BasicAlgebra.calcInnerProduct(kernelSumArray, alphasA);
            double[] arrayB = BasicAlgebra.scalarMultiple((1.0d - ipB) / btb, kernelSumArray);
            double[] alphasB = BasicAlgebra.calcVectorSum(alphasA, arrayB);
            for (int i = 0; i < alphasB.length; i++) {
                alphasB[i] = (alphasB[i] > 0.0d) ? alphasB[i] : 0.0d;
            }

            double ipC = BasicAlgebra.calcInnerProduct(kernelSumArray, alphasB);
            alphas = BasicAlgebra.scalarMultiple(1.0d / ipC, alphasB);
            diff = magnitude - BasicAlgebra.calcMagnitude(alphas);
        }
        return alphas;
    }

    public static double[] estimateDensityRatioKullbackLeibler(FeatureVector[] trainingFeatureVectors, Kernel kernel, double epsilon, double tolerance, FeatureVector... testFeatureVectors) {
        double[] alphas = optimizeKlParams(trainingFeatureVectors, testFeatureVectors, kernel, epsilon, tolerance);
        double[] densityRatios = new double[testFeatureVectors.length];
        for (int i = 0; i < testFeatureVectors.length; i++) {
            densityRatios[i] = 0.0d;
            for (int j = 0; j < trainingFeatureVectors.length; j++)
                densityRatios[i] += alphas[j] * kernel.kernelFunction(testFeatureVectors[i].getAllValues(), trainingFeatureVectors[j].getAllValues());
        }
        return densityRatios;
    }

    public static double[] estimateDensityRatioKullbackLeibler(FeatureVector[] trainingFeatureVectors, Kernel kernel, double[] alphas, FeatureVector... testFeatureVectors) {
        double[] densityRatios = new double[testFeatureVectors.length];
        for (int i = 0; i < testFeatureVectors.length; i++) {
            densityRatios[i] = 0.0d;
            for (int j = 0; j < trainingFeatureVectors.length; j++) {
                densityRatios[i] += alphas[j] * kernel.kernelFunction(testFeatureVectors[i].getAllValues(), trainingFeatureVectors[j].getAllValues());
            }
        }
        return densityRatios;
    }
}

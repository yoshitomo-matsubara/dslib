package jp.mylib.science.classification;

import jp.mylib.science.common.BasicAlgebra;
import jp.mylib.science.common.BasicMath;
import jp.mylib.science.common.FeatureVector;
import jp.mylib.science.statistics.DensityEstimator;
import jp.mylib.science.statistics.Kernel;

import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;

public class AnomalyDetector
{
    private double calcLocalOutlierFactor(FeatureVector targetVector, FeatureVector[] featureVectors, int k)
    {
        ArrayList<Double> distList = new ArrayList<Double>();
        HashMap<Double, ArrayList<Integer>> distMap = new HashMap<Double, ArrayList<Integer>>();
        for(int i=0;i<featureVectors.length;i++)
        {
            FeatureVector featureVector = featureVectors[i];
            double dist = BasicAlgebra.calcEuclideanDistance(featureVector.getAllValues(), targetVector.getAllValues());
            distList.add(dist);
            if(!distMap.containsKey(dist))
                distMap.put(dist, new ArrayList<Integer>());
            distMap.get(dist).add(i);
        }

        Collections.sort(distList);
        ArrayList<Integer> topKIndexList = new ArrayList<Integer>();
        while(topKIndexList.size() >= k)
        {
            ArrayList<Integer> indexList = distMap.get(topKIndexList.size());
            topKIndexList.addAll(indexList);
        }

        FeatureVector[] topKFeatureVectors = new FeatureVector[k];
        for(int i=0;i<k;i++)
            topKFeatureVectors[i] = featureVectors[topKIndexList.get(i)];
        double kthDist = distList.get(k - 1);

        double lof = 0.0d;
        for(int i=0;i<k;i++)
            lof += DensityEstimator.calcLocalReachabilityDensity(featureVectors[i], featureVectors, k, kthDist);

        return lof / (double)k / DensityEstimator.calcLocalReachabilityDensity(targetVector, featureVectors, k, kthDist);
    }

    public double[] getLocalOutlierFactors(FeatureVector[] featureVectors, int k)
    {
        double[] lofs = new double[featureVectors.length];
        for(int i=0;i<lofs.length;i++)
            calcLocalOutlierFactor(featureVectors[i], featureVectors, k);

        return lofs;
    }

    // Local Outlier Factor
    public int[] getOutlierIndicesBasedOnLof(FeatureVector[] featureVectors, int k, double threshold)
    {
        ArrayList<Integer> outlierIdxList = new ArrayList<Integer>();
        for(int i=0;i<featureVectors.length;i++)
        {
            double lof = calcLocalOutlierFactor(featureVectors[i], featureVectors, k);
            if(lof >= threshold)
                outlierIdxList.add(i);
        }

        int[] indices = new int[outlierIdxList.size()];
        for(int i=0;i<outlierIdxList.size();i++)
            indices[i] = outlierIdxList.get(i);

        return indices;
    }

    // Kullback-Leibler
    public int[] getOutlierIndicesBasedOnKl(FeatureVector[] trainingFeatureVectors, FeatureVector[] testFeatureVectors, Kernel kernel, double epsilon, double tolerance, double threshold)
    {
        ArrayList<Integer> outlierIdxList = new ArrayList<Integer>();
        double[] densityRatios = DensityEstimator.estimateDensityRatioKullbackLeibler(trainingFeatureVectors, testFeatureVectors, kernel, epsilon, tolerance);
        for(int i=0;i<densityRatios.length;i++)
            if(densityRatios[i] < threshold)
                outlierIdxList.add(i);

        int[] indices = new int[outlierIdxList.size()];
        for(int i=0;i<outlierIdxList.size();i++)
            indices[i] = outlierIdxList.get(i);

        return indices;
    }
}

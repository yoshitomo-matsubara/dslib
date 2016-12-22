package ymatsubara.dslib.classification;

import ymatsubara.dslib.common.BasicAlgebra;
import ymatsubara.dslib.structure.FeatureVector;
import ymatsubara.dslib.statistics.DensityEstimator;
import ymatsubara.dslib.statistics.Kernel;

import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;

public class AnomalyDetector {
    private double calcLocalOutlierFactor(FeatureVector targetVec, FeatureVector[] vecs, int k) {
        List<Double> distList = new ArrayList<>();
        HashMap<Double, List<Integer>> distMap = new HashMap<>();
        for (int i = 0; i < vecs.length; i++) {
            FeatureVector vec = vecs[i];
            double dist = BasicAlgebra.calcEuclideanDistance(vec.getAllValues(), targetVec.getAllValues());
            distList.add(dist);
            if (!distMap.containsKey(dist)) {
                distMap.put(dist, new ArrayList<Integer>());
            }
            distMap.get(dist).add(i);
        }

        Collections.sort(distList);
        List<Integer> topKIndexList = new ArrayList<>();
        while (topKIndexList.size() >= k) {
            List<Integer> indexList = distMap.get(distList.get(topKIndexList.size()));
            topKIndexList.addAll(indexList);
        }

        FeatureVector[] topkVecs = new FeatureVector[k];
        for (int i = 0; i < k; i++) {
            topkVecs[i] = vecs[topKIndexList.get(i)];
        }

        double kthDist = distList.get(k - 1);
        double lof = 0.0d;
        for (int i = 0; i < k; i++) {
            lof += DensityEstimator.calcLocalReachabilityDensity(vecs[i], vecs, k, kthDist);
        }
        return lof / (double) k / DensityEstimator.calcLocalReachabilityDensity(targetVec, vecs, k, kthDist);
    }

    public double[] getLocalOutlierFactors(FeatureVector[] vecs, int k) {
        double[] lofs = new double[vecs.length];
        for (int i = 0; i < lofs.length; i++) {
            lofs[i] = calcLocalOutlierFactor(vecs[i], vecs, k);
        }
        return lofs;
    }

    // Local Outlier Factor
    public int[] getOutlierIndicesBasedOnLof(FeatureVector[] vecs, int k, double threshold) {
        List<Integer> outlierIdxList = new ArrayList<>();
        for (int i = 0; i < vecs.length; i++) {
            double lof = calcLocalOutlierFactor(vecs[i], vecs, k);
            if (lof >= threshold) {
                outlierIdxList.add(i);
            }
        }

        int[] indices = new int[outlierIdxList.size()];
        for (int i = 0; i < outlierIdxList.size(); i++) {
            indices[i] = outlierIdxList.get(i);
        }
        return indices;
    }

    // Kullback Leibler
    public int[] getOutlierIndicesBasedOnKl(FeatureVector[] trainingVecs, FeatureVector[] testVecs, Kernel kernel, double epsilon, double tolerance, double threshold) {
        List<Integer> outlierIdxList = new ArrayList<>();
        double[] densityRatios = DensityEstimator.estimateDensityRatioKullbackLeibler(trainingVecs, kernel, epsilon, tolerance, testVecs);
        for (int i = 0; i < densityRatios.length; i++) {
            if (densityRatios[i] < threshold) {
                outlierIdxList.add(i);
            }
        }

        int[] indices = new int[outlierIdxList.size()];
        for (int i = 0; i < outlierIdxList.size(); i++) {
            indices[i] = outlierIdxList.get(i);
        }
        return indices;
    }
}

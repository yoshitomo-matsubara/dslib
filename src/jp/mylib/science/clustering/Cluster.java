package jp.mylib.science.clustering;

import jp.mylib.science.common.BasicMath;
import jp.mylib.science.common.FeatureVector;
import jp.mylib.science.common.FeatureVectorUtils;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.Random;

public class Cluster
{
    public static void kMeans(int clusterSize, FeatureVector[] featureVectors, double tolerance)
    {
        double[][] centers = new double[clusterSize][featureVectors[0].getSize()];
        double[] minValues = new double[centers.length];
        double[] maxValues = new double[centers.length];
        FeatureVectorUtils.getEachIndexMinMax(featureVectors, minValues, maxValues);

        // init weighted centers
        for(int i=0;i<centers.length;i++)
            for(int j=0;j<centers[0].length;j++)
            {
                Random rand = new Random();
                double center = maxValues[j] - minValues[j];
                centers[i][j] = center * rand.nextDouble() + minValues[j];
            }

        double diff = Double.MAX_VALUE;
        HashMap<Integer, ArrayList<Integer>> labelMap = new HashMap<Integer, ArrayList<Integer>>();
        while(diff > tolerance)
        {
            diff = 0.0d;
            labelMap = new HashMap<Integer, ArrayList<Integer>>();
            // find the nearest cluster
            for(int i=0;i<featureVectors.length;i++)
            {
                double minDist = Double.MAX_VALUE;
                int minIndex = 0;
                for(int j=0;j<centers.length;j++)
                {
                    double dist = BasicMath.calcEuclideanDistance(centers[j], featureVectors[i].getAllValues());
                    if(dist < minDist)
                    {
                        minDist = dist;
                        minIndex = j;
                    }
                }

                if(!labelMap.containsKey(minIndex))
                    labelMap.put(minIndex, new ArrayList<Integer>());

                labelMap.get(minIndex).add(i);
            }

            // update labels
            for(int label : labelMap.keySet())
            {
                double[] preCenters = new double[centers[0].length];
                for(int i=0;i<centers[0].length;i++)
                {
                    preCenters[i] = centers[label][i];
                    centers[label][i] = 0.0d;
                }

                ArrayList<Integer> indexList = labelMap.get(label);
                for(int index : indexList)
                    for(int i=0;i<centers[0].length;i++)
                        centers[label][i] += featureVectors[index].getValue(i);

                for(int i=0;i<centers[0].length;i++)
                    centers[label][i] /= (double)indexList.size();

                diff += BasicMath.calcEuclideanDistance(centers[label], preCenters);
            }
        }

        // set labels
        for(int label : labelMap.keySet())
        {
            ArrayList<Integer> indexList = labelMap.get(label);
            for(int index : indexList)
                featureVectors[index].setLabel(String.valueOf(label));
        }
    }

    public static void kMeans(int clusterSize, FeatureVector[] featureVectors)
    {
        kMeans(clusterSize, featureVectors, 1e-4d);
    }

    public static void kernelKMeans(int clusterSize, FeatureVector[] featureVectors, double tolerance)
    {

    }

    public static void kernelKMeans(int clusterSize, FeatureVector[] featureVectors)
    {

    }
}

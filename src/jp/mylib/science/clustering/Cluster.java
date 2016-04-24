package jp.mylib.science.clustering;

import jp.mylib.science.common.BasicAlgebra;
import jp.mylib.science.common.FeatureVector;
import jp.mylib.science.common.FeatureVectorUtil;
import jp.mylib.science.statistics.Kernel;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.Random;

public class Cluster
{
    public static final double DEFAULT_KMEANS_TOLERANCE = 1e-4d;
    public static final int DEFAULT_KMEANS_TOLERANCE_COUNT = 1;

    public static void kMeans(int clusterSize, FeatureVector[] featureVectors, double tolerance)
    {
        double[][] centers = new double[clusterSize][featureVectors[0].getSize()];
        double[] minValues = new double[centers.length];
        double[] maxValues = new double[centers.length];
        FeatureVectorUtil.getEachIndexMinMax(featureVectors, minValues, maxValues);

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
                    double dist = BasicAlgebra.calcEuclideanDistance(centers[j], featureVectors[i].getAllValues());
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

                diff += BasicAlgebra.calcEuclideanDistance(centers[label], preCenters);
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
        kMeans(clusterSize, featureVectors, DEFAULT_KMEANS_TOLERANCE);
    }

    public static void kernelKMeans(int clusterSize, FeatureVector[] featureVectors, Kernel kernel, int tolerance)
    {
        if(kernel.getType().equals(Kernel.LINEAR_KERNEL_TYPE))
            kMeans(clusterSize, featureVectors, tolerance);
        else
        {
            HashMap<Integer, ArrayList<Integer>> labelMap = new HashMap<Integer, ArrayList<Integer>>();
            // init weighted centers
            for(int i=0;i<featureVectors.length;i++)
            {
                Random rand = new Random();
                int index = rand.nextInt(clusterSize);
                if(!labelMap.containsKey(index))
                    labelMap.put(index, new ArrayList<Integer>());

                labelMap.get(index).add(i);
            }

            int diff = Integer.MAX_VALUE;
            while(diff > tolerance)
            {
                diff = 0;
                HashMap<Integer, ArrayList<Integer>> tmpLabelMap = new HashMap<Integer, ArrayList<Integer>>();
                double[] commonDists = new double[clusterSize];
                for(int i=0;i<commonDists.length;i++)
                    commonDists[i] = Double.MIN_VALUE;

                // find the nearest cluster
                for(int i=0;i<featureVectors.length;i++)
                {
                    double minDist = Double.MAX_VALUE;
                    int minIndex = 0;
                    for(int j=0;j<clusterSize;j++)
                    {
                        double dist = 0.0d;
                        ArrayList<Integer> indexList = labelMap.get(j);
                        double sum = 0.0d;
                        int indexSize = indexList.size();
                        for(int k=0;k<indexSize;k++)
                            sum += kernel.kernelFunction(featureVectors[indexList.get(k)].getAllValues(), featureVectors[i].getAllValues());

                        dist = -sum * 2.0d / (double)indexSize;
                        if(commonDists[j] == Double.MIN_VALUE)
                        {
                            sum = 0.0d;
                            for(int k=0;k<indexSize;k++)
                                for(int l=k;l<indexSize;l++)
                                        sum += kernel.kernelFunction(featureVectors[indexList.get(k)].getAllValues(), featureVectors[indexList.get(l)].getAllValues());
                        }
                        else
                            dist += commonDists[j] * 2.0d / Math.pow((double)indexList.size(), 2.0d);

                        if(dist < minDist)
                        {
                            minDist = dist;
                            minIndex = j;
                        }
                    }

                    if(!tmpLabelMap.containsKey(minIndex))
                        tmpLabelMap.put(minIndex, new ArrayList<Integer>());

                    tmpLabelMap.get(minIndex).add(i);
                }

                // update labels
                ArrayList<Integer> labelCountList = new ArrayList<Integer>();
                for(int key : labelMap.keySet())
                    labelCountList.add(labelMap.get(key).size());

                labelMap = new HashMap<Integer, ArrayList<Integer>>();
                int count = 0;
                for(int label : tmpLabelMap.keySet())
                {
                    ArrayList<Integer> indexList = tmpLabelMap.get(label);
                    labelMap.put(label, indexList);
                    diff += Math.abs(indexList.size() - labelCountList.get(count));
                    count++;
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
    }

    public static void kernelKMeans(int clusterSize, FeatureVector[] featureVectors, Kernel kernel)
    {
        kernelKMeans(clusterSize, featureVectors, kernel, DEFAULT_KMEANS_TOLERANCE_COUNT);
    }
}

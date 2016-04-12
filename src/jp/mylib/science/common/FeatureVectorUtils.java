package jp.mylib.science.common;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.util.ArrayList;
import java.util.List;

public class FeatureVectorUtils
{
    public static final String COMMENT_OUT = "//";
    public static final String NORMALIZATION = "Normalization";
    public static final String STANDARDIZATION = "Standardization";

    /*
    [id] is optional
    Delimiter: [\t], [,] or [ ]
    Comment out: [//]
    Dense vector type
    [id]\t[label]\t[value1]\t[value2]...
    [id]\t[label]\t[value1]\t[value2]...
    [id]\t[label]\t[value1]\t[value2]...
    ...
            or
    Sparse vector type
    [id]\t[label]\t[index1:value1]\t[index2:value2]\t[index3:value3]...
    [id]\t[label]\t[index3:value3]\t[index4:value4]\t[index7:value7]...
    [id]\t[label]\t[index2:value2]\t[index5:value5]\t[index7:value7]...
    ...
     */
    public static FeatureVector[] generateFeatureVectors(String inputFilePath)
    {
        File inputFile = new File(inputFilePath);
        List<FeatureVector> vecList = new ArrayList<FeatureVector>();
        try
        {
            BufferedReader br = new BufferedReader(new FileReader(inputFile));
            String line;
            while((line = br.readLine()) != null && line.length() > 0)
            {
                if(line.startsWith(COMMENT_OUT))
                    continue;

                String[] params = line.split("\t");
                if(params.length < 3)
                {
                    params = line.split(",");
                    if(params.length < 3)
                        params = line.split(" ");
                }

                FeatureVector featureVector = new FeatureVector(params[0], params[1], params.length - 2);
                List<Integer> indexList = new ArrayList<Integer>();
                List<Double> valueList = new ArrayList<Double>();
                for(int i=2;i<params.length;i++)
                {
                    String[] keyValue = params[i].split(":");
                    if(keyValue.length == 2)
                    {
                        indexList.add(Integer.parseInt(keyValue[0]));
                        valueList.add(Double.parseDouble(keyValue[1]));
                    }
                    else
                        valueList.add(Double.parseDouble(params[i]));
                }

                if(indexList.size() == valueList.size())
                    featureVector.setValues(valueList, indexList);
                else
                    featureVector.setValues(valueList);

                vecList.add(featureVector);
            }

            br.close();
        }
        catch(Exception e)
        {
            System.err.println("Invalid file for FeatureVector class : " + inputFile.getName());
        }

        return vecList.toArray(new FeatureVector[vecList.size()]);
    }

    public static FeatureVector[] getTargetVectors(FeatureVector[] featureVectors, String targetLabel)
    {
        ArrayList<FeatureVector> vectorList = new ArrayList<FeatureVector>();
        for(FeatureVector featureVector : featureVectors)
            if(featureVector.getLabel().equals(targetLabel))
                vectorList.add(featureVector);

        FeatureVector[] targetVectors = new FeatureVector[vectorList.size()];
        for(int i=0;i<targetVectors.length;i++)
            targetVectors[i] = vectorList.get(i);

        return targetVectors;
    }

    public static List<FeatureVector> getTargetVectorList(List<FeatureVector> featureVectorList, String targetLabel)
    {
        ArrayList<FeatureVector> vectorList = new ArrayList<FeatureVector>();
        for(FeatureVector featureVector : featureVectorList)
            if(featureVector.getLabel().equals(targetLabel))
                vectorList.add(featureVector);

        return vectorList;
    }

    public static void getEachIndexMinMax(FeatureVector[] featureVectors, double[] minValues, double[] maxValues)
    {
        for(int i=0;i<minValues.length;i++)
        {
            minValues[i] = featureVectors[0].getValue(i);
            maxValues[i] = featureVectors[0].getValue(i);
        }

        for(int i=1;i<featureVectors.length;i++)
            for(int j=0;j<minValues.length;j++)
            {
                double value = featureVectors[i].getValue(j);
                if(value < minValues[j])
                    minValues[j] = value;
                if(value > maxValues[j])
                    maxValues[j] = value;
            }
    }

    public static void getEachIndexAveSd(FeatureVector[] featureVectors, double[] aveValues, double[] sdValues)
    {
        double[][] matrix = new double[aveValues.length][featureVectors.length];
        for(int i=0;i<featureVectors.length;i++)
            for(int j=0;j<aveValues.length;j++)
                matrix[j][i] = featureVectors[i].getValue(j);

        for(int i=0;i<featureVectors.length;i++)
        {
            aveValues[i] = BasicMath.calcAverage(matrix[i]);
            sdValues[i] = BasicMath.calcStandardDeviation(matrix[i], aveValues[i]);
        }
    }

    public static void doScaling(FeatureVector[] featureVectors, FeatureVector[] baseVectors, String type)
    {
        if(type.equals(NORMALIZATION))
        {
            double[] minValues = new double[featureVectors[0].getSize()];
            double[] maxValues = new double[featureVectors[0].getSize()];
            getEachIndexMinMax(baseVectors, minValues, maxValues);
            for(int i=0;i<featureVectors.length;i++)
            {
                double[] scaledValues = new double[featureVectors[i].getSize()];
                for(int j=0;j<scaledValues.length;j++)
                    scaledValues[j] = DataProcessor.normalize(featureVectors[i].getValue(j), minValues[j], maxValues[j]);

                featureVectors[i].replaceAllValues(scaledValues);
            }
        }
        else if(type.equals(STANDARDIZATION))
        {
            double[] aveValues = new double[featureVectors[0].getSize()];
            double[] sdValues = new double[featureVectors[0].getSize()];
            getEachIndexAveSd(baseVectors, aveValues, sdValues);
            for(int i=0;i<featureVectors.length;i++)
            {
                double[] scaledValues = new double[featureVectors[i].getSize()];
                for(int j=0;j<scaledValues.length;j++)
                    scaledValues[j] = DataProcessor.standardize(featureVectors[i].getValue(j), aveValues[j], sdValues[j]);

                featureVectors[i].replaceAllValues(scaledValues);
            }
        }
    }

    public static void doScaling(FeatureVector[] featureVectors, String type)
    {
        doScaling(featureVectors, featureVectors, type);
    }

    public static void doScaling(FeatureVector[] featureVectors)
    {
        doScaling(featureVectors, NORMALIZATION);
    }
}

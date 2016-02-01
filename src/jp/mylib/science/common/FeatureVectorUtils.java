package jp.mylib.science.common;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.util.ArrayList;
import java.util.List;

public class FeatureVectorUtils
{
    public static void getEachIndexMinMax(FeatureVector[] featureVectors, double[] minValues, double[] maxValues)
    {
        minValues = new double[featureVectors[0].getSize()];
        maxValues = new double[minValues.length];
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

    /*
    [id] is optional
    Delimiter: [\t], [,] or [ ]
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
    public static List<FeatureVector> generateFeatureVectors(String inputFilePath)
    {
        File inputFile = new File(inputFilePath);
        List<FeatureVector> vecList = new ArrayList<FeatureVector>();
        try
        {
            BufferedReader br = new BufferedReader(new FileReader(inputFile));
            String line;
            while((line = br.readLine()) != null)
            {
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

        return vecList;
    }
}

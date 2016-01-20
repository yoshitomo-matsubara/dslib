package jp.mylib.science.common;

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
}

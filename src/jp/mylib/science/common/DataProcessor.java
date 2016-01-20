package jp.mylib.science.common;

import java.util.ArrayList;
import java.util.List;

public class DataProcessor
{
    public static double normalize(double value, double min, double max)
    {
        return (value - min) / (max - min);
    }

    public static double[] normalize(double[] values, double min, double max)
    {
        double[] array = new double[values.length];
        for(int i=0;i<values.length;i++)
            array[i] = normalize(values[i], min, max);
        return array;
    }

    public static List<Double> normalize(List<Double> valueList, double min, double max)
    {
        List<Double> list = new ArrayList<Double>();
        for(int i=0;i<list.size();i++)
            list.add(normalize(valueList.get(i), min, max));
        return list;
    }

    public static double standardize(double value, double ave, double sd)
    {
        return (value - ave) / sd;
    }

    public static double[] standardize(double[] values, double min, double max)
    {
        double[] array = new double[values.length];
        for(int i=0;i<values.length;i++)
            array[i] = standardize(values[i], min, max);
        return array;
    }

    public static List<Double> standardize(List<Double> valueList, double min, double max)
    {
        List<Double> list = new ArrayList<Double>();
        for(int i=0;i<list.size();i++)
            list.add(standardize(valueList.get(i), min, max));
        return list;
    }

    public static double[] doCentering(double[] values)
    {
        double ave = BasicMath.calcAverage(values);
        double[] array = new double[values.length];
        for(int i=0;i<values.length;i++)
            array[i] = values[i] - ave;
        return array;
    }

    public static List<Double> doCentering(List<Double> valueList)
    {
        double ave = BasicMath.calcAverage(valueList);
        List<Double> list = new ArrayList<Double>();
        for(int i=0;i<valueList.size();i++)
            list.add(valueList.get(i) - ave);
        return list;
    }
}

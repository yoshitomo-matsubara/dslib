package jp.mylib.science.common;

import java.util.List;

public class BasicMath
{
    public static double calcAverage(double[] array)
    {
        double sum = 0.0d;
        for(double value : array)
            sum += value;
        return sum / (double)array.length;
    }

    public static double calcAverage(List<Double> list)
    {
        double sum = 0.0d;
        for(double value : list)
            sum += value;
        return sum / (double)list.size();
    }

    public static double calcStandardDeviation(double[] array)
    {
        double ave = calcAverage(array);
        double sum = 0.0d;
        for(double value : array)
            sum += Math.pow(value - ave, 2.0d);
        return Math.sqrt(sum / (double)array.length);
    }

    public static double calcStandardDeviation(List<Double> list)
    {
        double ave = calcAverage(list);
        double sum = 0.0d;
        for(double value : list)
            sum += Math.pow(value - ave, 2.0d);
        return Math.sqrt(sum / (double)list.size());
    }

    public static double calcStandardDeviation(double[] array, double ave)
    {
        double sum = 0.0d;
        for(double value : array)
            sum += Math.pow(value - ave, 2.0d);
        return Math.sqrt(sum / (double)array.length);
    }

    public static double calcStandardDeviation(List<Double> list, double ave)
    {
        double sum = 0.0d;
        for(double value : list)
            sum += Math.pow(value - ave, 2.0d);
        return Math.sqrt(sum / (double)list.size());
    }

    public static double calcMagnitude(double[] array)
    {
        double magnitude = 0.0d;
        for(double value : array)
            magnitude += Math.pow(value, 2.0d);
        return Math.sqrt(magnitude);
    }

    public static double calcMagnitude(List<Double> list)
    {
        double magnitude = 0.0d;
        for(double value : list)
            magnitude += Math.pow(value, 2.0d);
        return Math.sqrt(magnitude);
    }

    public static double calcInnerProduct(double[] arrayX, double[] arrayY)
    {
        double ip = 0.0d;
        for(int i=0;i<arrayX.length;i++)
            ip += arrayX[i] * arrayY[i];
        return ip;
    }

    public static double calcInnerProduct(List<Double> listX, List<Double> listY)
    {
        double ip = 0.0d;
        for(int i=0;i<listX.size();i++)
            ip += listX.get(i) * listY.get(i);
        return ip;
    }

    public static double calcCosineDistance(double[] arrayX, double[] arrayY)
    {
        return 1.0d - calcInnerProduct(arrayX, arrayY) / (calcMagnitude(arrayX) * calcMagnitude(arrayY));
    }

    public static double calcCosineDistance(List<Double> listX, List<Double> listY)
    {
        return 1.0d - calcInnerProduct(listX, listY) / (calcMagnitude(listX) * calcMagnitude(listY));
    }

    public static double calcEuclideanDistance(double[] arrayX, double[] arrayY)
    {
        double sum = 0.0d;
        for(int i=0;i<arrayX.length;i++)
            sum += Math.pow(arrayX[i] - arrayY[i], 2.0d);
        return Math.sqrt(sum);
    }

    public static double calcEuclideanDistance(List<Double> listX, List<Double> listY)
    {
        double sum = 0.0d;
        for(int i=0;i<listX.size();i++)
            sum += Math.pow(listX.get(i) - listY.get(i), 2.0d);
        return Math.sqrt(sum);
    }

    public static double calcManhattanDistance(double[] arrayX, double[] arrayY)
    {
        double sum = 0.0d;
        for(int i=0;i<arrayX.length;i++)
            sum += Math.abs(arrayX[i] - arrayY[i]);
        return sum;
    }

    public static double calcManhattanDistance(List<Double> listX, List<Double> listY)
    {
        double sum = 0.0d;
        for(int i=0;i<listX.size();i++)
            sum += Math.abs(listX.get(i) - listY.get(i));
        return sum;
    }

    public static double calcMahalanobisDistance(double[] arrayX, double[] arrayY)
    {
        double sum = 0.0d;
        double sd = calcStandardDeviation(arrayX, calcAverage(arrayX));
        for(int i=0;i<arrayX.length;i++)
            sum += Math.pow((arrayX[i] - arrayY[i]) / sd, 2.0d);
        return Math.sqrt(sum);
    }

    public static double calcMahalanobisDistance(List<Double> listX, List<Double> listY)
    {
        double sum = 0.0d;
        double sd = calcStandardDeviation(listX, calcAverage(listX));
        for(int i=0;i<listX.size();i++)
            sum += Math.pow((listX.get(i) - listY.get(i)) / sd, 2.0d);
        return Math.sqrt(sum);
    }
}

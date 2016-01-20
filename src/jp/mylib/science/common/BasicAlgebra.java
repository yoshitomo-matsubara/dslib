package jp.mylib.science.common;

import java.util.List;

public class BasicAlgebra
{
    public static final double DEFAULT_POLYNOMIAL_KERNEL_CONSTANT = 1.0d;
    public static final double DEFAULT_POLYNOMIAL_KERNEL_POWER = 1.0d;
    public static final double DEFAULT_GAUSSIAN_KERNEL_SD = 0.3d;

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
        double sd = BasicMath.calcStandardDeviation(arrayX, BasicMath.calcAverage(arrayX));
        for(int i=0;i<arrayX.length;i++)
            sum += Math.pow((arrayX[i] - arrayY[i]) / sd, 2.0d);
        return Math.sqrt(sum);
    }

    public static double calcMahalanobisDistance(List<Double> listX, List<Double> listY)
    {
        double sum = 0.0d;
        double sd = BasicMath.calcStandardDeviation(listX, BasicMath.calcAverage(listX));
        for(int i=0;i<listX.size();i++)
            sum += Math.pow((listX.get(i) - listY.get(i)) / sd, 2.0d);
        return Math.sqrt(sum);
    }

    public static double polynomialKernel(double[] arrayX, double[] arrayY, double c, double p)
    {
        return Math.pow(calcInnerProduct(arrayX, arrayY) + c, p);
    }

    public static double polynomialKernel(double[] arrayX, double[] arrayY)
    {
        return polynomialKernel(arrayX, arrayY, DEFAULT_POLYNOMIAL_KERNEL_CONSTANT, DEFAULT_POLYNOMIAL_KERNEL_POWER);
    }

    public static double gaussianKernel(double[] arrayX, double[] arrayY, double sd)
    {
        return Math.exp(BasicAlgebra.calcInnerProduct(arrayX, arrayY) / (2.0d * Math.pow(sd, 2.0d)));
    }

    public static double gaussianKernel(double[] arrayX, double[] arrayY)
    {
        return gaussianKernel(arrayX, arrayY, DEFAULT_GAUSSIAN_KERNEL_SD);
    }
}

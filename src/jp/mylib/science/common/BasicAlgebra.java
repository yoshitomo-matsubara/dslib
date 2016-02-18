package jp.mylib.science.common;

import java.util.List;

public class BasicAlgebra
{
    public static final String LINEAR_KERNEL_TYPE = "LINEAR KERNEL";
    public static final String POLYNOMIAL_KERNEL_TYPE = "POLYNOMIAL KERNEL";
    public static final String GAUSSIAN_KERNEL_TYPE = "GAUSSIAN KERNEL";
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

    public static double[] getRow(double[][] matrix, int row)
    {
        double[] array = new double[matrix.length];
        for(int i=0;i<matrix.length;i++)
            array[i] = matrix[i][row];

        return array;
    }

    public static double[][] transposeMatrix(double[][] matrix)
    {
        double[][] matrixT = new double[matrix[0].length][matrix.length];
        for(int i=0;i<matrix.length;i++)
            for(int j=0;j<matrix[0].length;j++)
                matrixT[j][i] = matrix[i][j];

        return matrixT;
    }

    public static double[][] generateIdentityMatrix(int size)
    {
        double[][] matrix = new double[size][size];
        for(int i=0;i<matrix.length;i++)
            for(int j=0;j<matrix[0].length;j++)
                matrix[i][j] = (i == j)? 1.0d : 0.0d;

        return matrix;
    }

    public static double calcDeterminant(double[][] matrix)
    {
        double det = 0.0d;
        if(matrix.length != matrix[0].length)
            return det;

        double prod = 1.0d;
        if(matrix.length == 2 && matrix[0].length == 2)
        {
            det = matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0];
            return det;
        }

        for(int i=0;i<matrix.length;i++)
        {
            // calc det in a recursive way calcDeterminant(part of "matrix");
        }

        return det;
    }

    public static double[][] calcInverseMatrix(double[][] matrix)
    {
        double[][] invMatrix = new double[matrix.length][matrix.length];
        if(matrix.length != matrix[0].length)
            return invMatrix;

        double det = calcDeterminant(matrix);
        if(det == 0.0d)
            return invMatrix;

        // adjugate matrix

        return invMatrix;
    }

    public static double[][] calcMatrixProduct(double[][] matrixX, double[][] matrixY)
    {
        double[][] matrix = new double[matrixX.length][matrixY[0].length];
        for(int i=0;i<matrix.length;i++)
            for(int j=0;j<matrix[0].length;j++)
                matrix[i][j] = 0.0d;

        if(matrixY.length != matrixX[0].length)
            return matrix;

        for(int i=0;i<matrix.length;i++)
            for(int j=0;j<matrix[0].length;j++)
                for(int k=0;k<matrix[0].length;k++)
                    matrix[i][j] += matrixX[i][k] * matrixY[k][j];

        return matrix;
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
        return Math.exp(-BasicAlgebra.calcEuclideanDistance(arrayX, arrayY) / (2.0d * Math.pow(sd, 2.0d)));
    }

    public static double gaussianKernel(double[] arrayX, double[] arrayY)
    {
        return gaussianKernel(arrayX, arrayY, DEFAULT_GAUSSIAN_KERNEL_SD);
    }

    public static double kernelFunction(double[] arrayX, double[] arrayY, String kernelType)
    {
        if(kernelType.equals(POLYNOMIAL_KERNEL_TYPE))
            return polynomialKernel(arrayX, arrayY);
        else if(kernelType.equals(GAUSSIAN_KERNEL_TYPE))
            return gaussianKernel(arrayX, arrayY);

        return Double.NaN;
    }

    public static double kernelFunction(double[] arrayX, double[] arrayY, String kernelType, double[] kernelParams)
    {
        if(kernelType.equals(POLYNOMIAL_KERNEL_TYPE))
            return polynomialKernel(arrayX, arrayY, kernelParams[0], kernelParams[1]);
        else if(kernelType.equals(GAUSSIAN_KERNEL_TYPE))
            return gaussianKernel(arrayX, arrayY, kernelParams[0]);

        return Double.NaN;
    }
}

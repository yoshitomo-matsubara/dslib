package jp.mylib.science.common;

import java.util.List;

public class BasicAlgebra
{
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

    public static double[][] generateDiagonalMatrix(double[] array)
    {
        double[][] diagMatrix = new double[array.length][array.length];
        for(int i=0;i<diagMatrix.length;i++)
            for(int j=0;j<diagMatrix[0].length;j++)
                diagMatrix[i][j] = (i == j)? array[i] : 0.0d;

        return diagMatrix;
    }

    public static double[][] generateIdentityMatrix(int size)
    {
        double[] array = new double[size];
        for(int i=0;i<array.length;i++)
            array[i] = 1.0d;

        return generateDiagonalMatrix(array);
    }

    public static double[][] get2dRotationMatrix(double rad)
    {
        return new double[][]{{Math.cos(rad), -Math.sin(rad)}, {Math.sin(rad), Math.cos(rad)}};
    }

    public static double[][] get3dxRotationMatrix(double rad)
    {
        return new double[][]{{1.0d, 0.0d, 0.0d}, {0.0d, Math.cos(rad), -Math.sin(rad)}, {0.0d, Math.sin(rad), Math.cos(rad)}};
    }

    public static double[][] get3dyRotationMatrix(double rad)
    {
        return new double[][]{{Math.cos(rad), 0.0d, Math.sin(rad)}, {0.0d, 1.0d, 0.0d}, {-Math.sin(rad), 0.0d, Math.cos(rad)}};
    }

    public static double[][] get3dzRotationMatrix(double rad)
    {
        return new double[][]{{Math.cos(rad), -Math.sin(rad), 0.0d}, {Math.sin(rad), Math.cos(rad), 0.0d}, {0.0d, 0.0d, 1.0d}};
    }

    public static double calcDeterminant(double[][] matrix)
    {
        double det = 0.0d;
        if(matrix.length != matrix[0].length)
            return det;

        if(matrix.length == 2 && matrix[0].length == 2)
        {
            det = matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0];
            return det;
        }

        for(int i=0;i<matrix.length;i++)
        {
            double[][] subMatrix = new double[matrix.length - 1][matrix.length - 1];
            int l = 0;
            for(int j=0;j<matrix.length;j++)
                if(i != j)
                {
                    for(int k=1;k<matrix.length;k++)
                        subMatrix[l][k - 1] = matrix[j][k];

                    l++;
                }

            double sign = (i % 2 == 0)? 1.0d : -1.0d;
            det += sign * matrix[i][0] * calcDeterminant(subMatrix);
        }

        return det;
    }

    public static double[][] calcInverseMatrix(double[][] matrix)
    {
        if(matrix.length != matrix[0].length)
            return new double[matrix.length][matrix.length];

        double det = calcDeterminant(matrix);
        if(det == 0.0d)
            return new double[matrix.length][matrix.length];

        // sweep out method
        double[][] invMatrix = generateIdentityMatrix(matrix.length);
        double buf = 0.0d;
        for(int i=0;i<matrix.length;i++)
        {
            buf = 1.0d / matrix[i][i];
            for(int j=0;j<matrix.length;j++)
            {
                matrix[i][j] *= buf;
                invMatrix[i][j] *= buf;
            }

            for(int j=0;j<matrix.length;j++)
            {
                if(i != j)
                {
                    buf = matrix[j][i];
                    for(int k=0;k<matrix.length;k++)
                    {
                        matrix[j][k] -= matrix[i][k] * buf;
                        invMatrix[j][k] -= invMatrix[i][k] * buf;
                    }
                }
            }
        }

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
        double dist = 0.0d;
        for(int i=0;i<arrayX.length;i++)
            dist += Math.pow(arrayX[i] - arrayY[i], 2.0d);

        return Math.sqrt(dist);
    }

    public static double calcEuclideanDistance(List<Double> listX, List<Double> listY)
    {
        double dist = 0.0d;
        for(int i=0;i<listX.size();i++)
            dist += Math.pow(listX.get(i) - listY.get(i), 2.0d);

        return Math.sqrt(dist);
    }

    public static double calcManhattanDistance(double[] arrayX, double[] arrayY)
    {
        double dist = 0.0d;
        for(int i=0;i<arrayX.length;i++)
            dist += Math.abs(arrayX[i] - arrayY[i]);

        return dist;
    }

    public static double calcManhattanDistance(List<Double> listX, List<Double> listY)
    {
        double dist = 0.0d;
        for(int i=0;i<listX.size();i++)
            dist += Math.abs(listX.get(i) - listY.get(i));

        return dist;
    }

    public static double calcMahalanobisDistance(double[] arrayX, double[] arrayY, double[] arraySd)
    {
        double dist = 0.0d;
        for(int i=0;i<arrayX.length;i++)
            dist += Math.pow((arrayX[i] - arrayY[i]) / arraySd[i], 2.0d);

        return Math.sqrt(dist);
    }

    public static double calcMahalanobisDistance(List<Double> listX, List<Double> listY, List<Double> listSd)
    {
        double sum = 0.0d;
        for(int i=0;i<listX.size();i++)
            sum += Math.pow((listX.get(i) - listY.get(i)) / listSd.get(i), 2.0d);

        return Math.sqrt(sum);
    }
}

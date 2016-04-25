package jp.mylib.science.statistics;

import jp.mylib.science.common.BasicAlgebra;
import jp.mylib.science.common.FeatureVector;

public class Kernel
{
    public static final String LINEAR_KERNEL_TYPE = "LINEAR KERNEL";
    public static final String POLYNOMIAL_KERNEL_TYPE = "POLYNOMIAL KERNEL";
    public static final String RBF_KERNEL_TYPE = "RBF KERNEL";
    public static final String GAUSSIAN_KERNEL_TYPE = "GAUSSIAN KERNEL";
    public static final String SIGMOID_KERNEL_TYPE = "SIGMOID KERNEL";
    private String type;
    private double[] params;

    public Kernel(String type, double... params)
    {
        this.type = type;
        this.params = new double[params.length];
        for(int i=0;i<params.length;i++)
            this.params[i] = params[i];
    }

    public Kernel(String type)
    {
        this(type, Double.NaN);
    }

    public String getType()
    {
        return this.type;
    }

    public double[] getParams()
    {
        return this.params;
    }

    public void setType(String type)
    {
        this.type = type;
    }

    public void setParams(double[] params)
    {
        this.params = new double[params.length];
        for(int i=0;i<params.length;i++)
            this.params[i] = params[i];
    }

    public void setParam(double param, int index)
    {
        this.params[index] = param;
    }

    public double linearKernel(double[] arrayX, double[] arrayY)
    {
        return BasicAlgebra.calcInnerProduct(arrayX, arrayY);
    }

    public double polynomialKernel(double[] arrayX, double[] arrayY, double c, double p)
    {
        return Math.pow(BasicAlgebra.calcInnerProduct(arrayX, arrayY) + c, p);
    }

    public double radialBasisFunctionKernel(double[] arrayX, double[] arrayY, double gamma)
    {
        return Math.exp(-gamma * Math.pow(BasicAlgebra.calcEuclideanDistance(arrayX, arrayY), 2.0d));
    }

    public double gaussianKernel(double[] arrayX, double[] arrayY, double sigma)
    {
        return Math.exp(-Math.pow(BasicAlgebra.calcEuclideanDistance(arrayX, arrayY), 2.0d) / (2.0d * Math.pow(sigma, 2.0d)));
    }

    public double sigmoidKernel(double[] arrayX, double[] arrayY, double c, double theta)
    {
        return Math.tanh(c * BasicAlgebra.calcInnerProduct(arrayX, arrayY) + theta);
    }

    public double kernelFunction(double[] arrayX, double[] arrayY)
    {
        if(this.type.equals(LINEAR_KERNEL_TYPE))
            return linearKernel(arrayX, arrayY);
        else if(this.type.equals(POLYNOMIAL_KERNEL_TYPE))
            return polynomialKernel(arrayX, arrayY, this.params[0], this.params[1]);
        else if(this.type.equals(RBF_KERNEL_TYPE))
            return radialBasisFunctionKernel(arrayX, arrayY, this.params[0]);
        else if(this.type.equals(GAUSSIAN_KERNEL_TYPE))
            return gaussianKernel(arrayX, arrayY, this.params[0]);
        else if(this.type.equals(SIGMOID_KERNEL_TYPE))
            return sigmoidKernel(arrayX, arrayY, this.params[0], this.params[1]);

        return Double.NaN;
    }

    public double[][] calcKernelMatrix(FeatureVector[] featureVectors)
    {
        double[][] kernelMatrix = new double[featureVectors.length][featureVectors.length];
        for(int i=0;i<kernelMatrix.length;i++)
            for(int j=i;j<kernelMatrix[0].length;j++)
            {
                kernelMatrix[i][j] = kernelFunction(featureVectors[i].getAllValues(), featureVectors[j].getAllValues());
                kernelMatrix[j][i] = kernelMatrix[i][j];
            }

        return kernelMatrix;
    }
}

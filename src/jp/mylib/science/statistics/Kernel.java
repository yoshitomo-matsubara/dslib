package jp.mylib.science.statistics;

import jp.mylib.science.common.BasicAlgebra;
import jp.mylib.science.common.FeatureVector;

public class Kernel
{
    public static final String LINEAR_KERNEL_TYPE = "LINEAR KERNEL";
    public static final String POLYNOMIAL_KERNEL_TYPE = "POLYNOMIAL KERNEL";
    public static final String GAUSSIAN_KERNEL_TYPE = "GAUSSIAN KERNEL";
    public static final double DEFAULT_POLYNOMIAL_KERNEL_CONSTANT = 1.0d;
    public static final double DEFAULT_POLYNOMIAL_KERNEL_POWER = 1.0d;
    public static final double DEFAULT_GAUSSIAN_KERNEL_SD = 0.3d;
    private String type;
    private double[] params;

    public Kernel(String type, double[] params)
    {
        this.type = type;
        this.params = new double[params.length];
        for(int i=0;i<params.length;i++)
            this.params[i] = params[i];
    }

    public Kernel(String type, double param)
    {
        this(type, new double[]{param});
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

    public double polynomialKernel(double[] arrayX, double[] arrayY)
    {
        return polynomialKernel(arrayX, arrayY, DEFAULT_POLYNOMIAL_KERNEL_CONSTANT, DEFAULT_POLYNOMIAL_KERNEL_POWER);
    }

    public double gaussianKernel(double[] arrayX, double[] arrayY, double sd)
    {
        return Math.exp(-BasicAlgebra.calcEuclideanDistance(arrayX, arrayY) / (2.0d * Math.pow(sd, 2.0d)));
    }

    public double gaussianKernel(double[] arrayX, double[] arrayY)
    {
        return gaussianKernel(arrayX, arrayY, DEFAULT_GAUSSIAN_KERNEL_SD);
    }

    public double kernelFunction(double[] arrayX, double[] arrayY)
    {
        if(this.type.equals(LINEAR_KERNEL_TYPE))
            return linearKernel(arrayX, arrayY);
        else if(this.type.equals(POLYNOMIAL_KERNEL_TYPE))
            return polynomialKernel(arrayX, arrayY, this.params[0], this.params[1]);
        else if(this.type.equals(GAUSSIAN_KERNEL_TYPE))
            return gaussianKernel(arrayX, arrayY, this.params[0]);

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

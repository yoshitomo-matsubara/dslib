package jp.mylib.science.classification;

import jp.mylib.science.common.BasicAlgebra;
import jp.mylib.science.common.FeatureVector;

public class OneClassSVM extends SVM
{
    public static final String SCHOLKOPF = "Scholkopf";
    public static final String TAX_AND_DUIN = "Tax and Duin";
    private FeatureVector weightedVector;
    private String svmType, kernelType;
    private double regParam, radius;
    private double[] kernelParams;

    public OneClassSVM(double regParam, String svmType, String kernelType, double[] kernelParams)
    {
        this.regParam = regParam;
        this.svmType = svmType;
        this.kernelType = kernelType;
        this.radius = Double.NaN;
        this.kernelParams = new double[kernelParams.length];
        for(int i=0;i<this.kernelParams.length;i++)
            this.kernelParams[i] = kernelParams[i];
    }

    private boolean isConvergent()
    {
        return false;
    }

    private int[] workingSetSelection3(double[][] kernelMatrix, double[] lagrangeAlphas, double[] gradients)
    {
        int[] indices = new int[2];
        // eq(20)

        return indices;
    }

    private double[] updateGradients(double[][] kernelMatrix, double[] lagrangeAlphas)
    {
        double[] gradients = new double[lagrangeAlphas.length];
        for(int i=0;i<gradients.length;i++)
            gradients[i] = BasicAlgebra.calcInnerProduct(kernelMatrix[i], lagrangeAlphas) - 1.0d;
        return gradients;
    }

    // B. Scholkopf et. al. "Support Vector Method for Novelty Detection"
    //      and R. Fan et. al. "Working Set Selection Using Second Order Information for Training Support Vector Machines"
    private void trainScholkopf(FeatureVector[] featureVectors)
    {
        int trainingSize = featureVectors.length;
        double[][] kernelMatrix = calcKernelMatrix(featureVectors, this.kernelType);
        // init an alpha array (Working Set Selection 3)
        double[] lagrangeAlphas = new double[trainingSize];
        double[] gradients = new double[trainingSize];
        for(int i=0;i<lagrangeAlphas.length;i++)
            lagrangeAlphas[i] = (i <= (int)(this.regParam * (double)trainingSize))? 1.0d / (double)lagrangeAlphas.length : 0.0d;

        gradients = updateGradients(kernelMatrix, lagrangeAlphas);
        while(isConvergent())
        {
            int[] targetIndices = workingSetSelection3(kernelMatrix, lagrangeAlphas, gradients);
            int i = targetIndices[0];
            int j = targetIndices[1];
            // Either of labels I or J should be -1.0d for binary classification, depending on feature vector's label
            double labelI = 1.0d;
            double labelJ = 1.0d;
            double a = kernelMatrix[i][i] + kernelMatrix[j][j] - 2.0d * kernelMatrix[i][j];
            double b = -labelI * gradients[i] + labelJ * gradients[j];
            double oldAlphaI = lagrangeAlphas[i];
            double oldAlphaJ = lagrangeAlphas[j];
            lagrangeAlphas[i] += labelI * b / a;
            lagrangeAlphas[j] -= labelJ * b / a;
            double c = 1.0d / ((double)trainingSize * this.regParam);
            if(!(0.0d <= lagrangeAlphas[i] && lagrangeAlphas[i] <= c))
            {
                if(lagrangeAlphas[i] < 0.0d)
                    lagrangeAlphas[i] = 0.0d;
                else if(lagrangeAlphas[i] > c)
                    lagrangeAlphas[i] = c;

                lagrangeAlphas[j] = lagrangeAlphas[j] + labelI * labelJ * (oldAlphaI - lagrangeAlphas[i]);
            }
            else if(!(0.0d <= lagrangeAlphas[j] && lagrangeAlphas[j] <= c))
            {
                if(lagrangeAlphas[j] < 0.0d)
                    lagrangeAlphas[j] = 0.0d;
                else if(lagrangeAlphas[j] > c)
                    lagrangeAlphas[j] = c;

                lagrangeAlphas[i] = lagrangeAlphas[i] + labelI * labelJ * (oldAlphaJ - lagrangeAlphas[j]);
            }

            gradients = updateGradients(kernelMatrix, lagrangeAlphas);
        }
    }

    private void trainTaxAndDuin(FeatureVector[] featureVectors)
    {

    }

    @Override
    public void train(FeatureVector[] featureVectors)
    {
        this.weightedVector = new FeatureVector(this.svmType, featureVectors[0].getAllIndices().length);
        if(this.svmType.equals(SCHOLKOPF))
            trainScholkopf(featureVectors);
        else if(this.svmType.equals(TAX_AND_DUIN))
            trainTaxAndDuin(featureVectors);
        else
            System.err.println(this.svmType + " is an invalid SVM type.");
    }

    @Override
    public void predict(FeatureVector[] featureVectors)
    {
        if(this.radius == Double.NaN)
        {
            System.err.println("The train method must be called before the predict method.");
            return;
        }
    }

    @Override
    public void inputModel(String modelFilePath)
    {

    }

    @Override
    public void outputModel(String modelFilePath)
    {

    }
}

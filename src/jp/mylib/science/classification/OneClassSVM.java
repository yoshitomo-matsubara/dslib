package jp.mylib.science.classification;

import jp.mylib.science.common.BasicAlgebra;
import jp.mylib.science.common.BasicMath;
import jp.mylib.science.common.FeatureVector;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public class OneClassSVM extends SVM
{
    public static final String SCHOLKOPF = "Scholkopf";
    public static final String TAX_AND_DUIN = "Tax and Duin";
    public static final double WSS3_TAU = 1.0e-4d;
    public static final double DEFAULT_TOLERANCE = 1.0e-4d;
    public static final int NORMAL_VALUE = 1;
    public static final int OUTLIER = -1;
    private String id, svmType, kernelType;
    private double regParam, tolerance, rho, radius;
    private double[] kernelParams, alphas;
    private double[][] kernelMatrix;
    private FeatureVector[] trainedFeatureVectors;

    public OneClassSVM(String id, double regParam, double tolerance, String svmType, String kernelType, double[] kernelParams)
    {
        this.id = id;
        this.regParam = regParam;
        this.svmType = svmType;
        this.kernelType = kernelType;
        this.tolerance = tolerance;
        this.rho = Double.NaN;
        this.radius = Double.NaN;
        this.kernelParams = new double[kernelParams.length];
        for(int i=0;i<this.kernelParams.length;i++)
            this.kernelParams[i] = kernelParams[i];
    }

    private void setUpLowIndexLists(double[] gradients, ArrayList<Integer> upIndexList, ArrayList<Integer> lowIndexList)
    {
        double c = (this.svmType.equals(SCHOLKOPF))? 1.0d / ((double)gradients.length * this.regParam) : this.regParam;
        for(int i=0;i<this.alphas.length;i++)
        {
            if(gradients[i] < c)
                upIndexList.add(i);
            else if(gradients[i] > 0.0d)
                lowIndexList.add(i);
        }
    }

    // select a value pair by WSS3 for one-class svm
    private double[] selectValuePairByWorkingSetSelection3(double[] gradients)
    {
        ArrayList<Integer> upIndexList = new ArrayList<Integer>();
        ArrayList<Integer> lowIndexList = new ArrayList<Integer>();
        setUpLowIndexLists(gradients, upIndexList, lowIndexList);

        // select i
        int i = 0;
        double max = Double.MIN_VALUE;
        for(int index : upIndexList)
            if(-gradients[index] > max)
            {
                max = -gradients[index];
                i = index;
            }

        // select j
        double min = Double.MAX_VALUE;
        for(int index : lowIndexList)
        {
            double a = this.kernelMatrix[i][i] + this.kernelMatrix[index][index] - 2.0d * this.kernelMatrix[i][index];
            double aBar = (a > 0.0d)? a : WSS3_TAU;
            double b = -gradients[i] + gradients[index];
            double value = -Math.pow(b, 2.0d) / aBar;
            if(gradients[index] < gradients[i] && value < min)
                min = value;
        }

        double[] values = {max, min};
        return values;
    }

    private boolean isConvergent(double[] gradients, boolean[] changedArray)
    {
        for(boolean changed : changedArray)
            if(!changed)
                return false;

        // KKT optimality condition in C. Chang and C. Lin "LIBSVM: A Library for Support Vector Machines"
        ArrayList<Integer> upIndexList = new ArrayList<Integer>();
        ArrayList<Integer> lowIndexList = new ArrayList<Integer>();
        setUpLowIndexLists(gradients, upIndexList, lowIndexList);
        double[] targetValues = selectValuePairByWorkingSetSelection3(gradients);
        return targetValues[0] - targetValues[1] <= this.tolerance;
    }

    // select a index pair by WSS3 for one-class svm
    private int[] selectIndexPairByWorkingSetSelection3(double[] gradients)
    {
        int[] indices = new int[2];
        ArrayList<Integer> upIndexList = new ArrayList<Integer>();
        ArrayList<Integer> lowIndexList = new ArrayList<Integer>();
        setUpLowIndexLists(gradients, upIndexList, lowIndexList);

        // select i = indices[0]
        double max = Double.MIN_VALUE;
        for(int index : upIndexList)
            if(-gradients[index] > max)
            {
                max = -gradients[index];
                indices[0] = index;
            }

        // select j = indices[1]
        double min = Double.MAX_VALUE;
        int i = indices[0];
        for(int index : lowIndexList)
        {
            double a = this.kernelMatrix[i][i] + this.kernelMatrix[index][index] - 2.0d * this.kernelMatrix[i][index];
            double aBar = (a > 0.0d)? a : WSS3_TAU;
            double b = -gradients[i] + gradients[index];
            double value = -Math.pow(b, 2.0d) / aBar;
            if(gradients[index] < gradients[i] && value < min)
            {
                min = value;
                indices[1] = index;
            }
        }

        return indices;
    }

    private double[] updateGradients()
    {
        double[] gradients = new double[this.alphas.length];
        for(int i=0;i<gradients.length;i++)
            gradients[i] = BasicAlgebra.calcInnerProduct(this.kernelMatrix[i], this.alphas) - 1.0d;
        return gradients;
    }

    // B. Scholkopf et. al. "Support Vector Method for Novelty Detection"
    //      and R. Fan et. al. "Working Set Selection Using Second Order Information for Training Support Vector Machines"
    private void trainScholkopf()
    {
        int trainingSize = this.trainedFeatureVectors.length;
        this.kernelMatrix = calcKernelMatrix(this.trainedFeatureVectors, this.kernelType);
        // init an alpha array (Working Set Selection 3)
        this.alphas = new double[trainingSize];
        double[] gradients = new double[trainingSize];
        for(int i=0;i<this.alphas.length;i++)
            this.alphas[i] = (i <= (int)(this.regParam * (double)trainingSize))? 1.0d / (double)this.alphas.length : 0.0d;

        // SMO
        boolean[] changedArray = new boolean[trainingSize];
        for(int i=0;i<changedArray.length;i++)
            changedArray[i] = false;

        gradients = updateGradients();
        double c = (this.svmType.equals(SCHOLKOPF))? 1.0d / ((double)trainingSize * this.regParam) : this.regParam;
        while(!isConvergent(gradients, changedArray))
        {
            int[] targetIndices = selectIndexPairByWorkingSetSelection3(gradients);
            int i = targetIndices[0];
            int j = targetIndices[1];
            // Either of labels I or J should be -1.0d for binary classification, depending on feature vector's label
            double labelI = 1.0d;
            double labelJ = 1.0d;
            double a = this.kernelMatrix[i][i] + this.kernelMatrix[j][j] - 2.0d * this.kernelMatrix[i][j];
            double b = -labelI * gradients[i] + labelJ * gradients[j];
            double oldAlphaI = this.alphas[i];
            double oldAlphaJ = this.alphas[j];
            this.alphas[i] += labelI * b / a;
            this.alphas[j] -= labelJ * b / a;
            if(0.0d > this.alphas[i] || this.alphas[i] > c)
            {
                if(this.alphas[i] < 0.0d)
                    this.alphas[i] = 0.0d;
                else if(this.alphas[i] > c)
                    this.alphas[i] = c;

                this.alphas[j] = this.alphas[j] + labelI * labelJ * (oldAlphaI - this.alphas[i]);
            }
            else if(0.0d > this.alphas[j] || this.alphas[j] > c)
            {
                if(this.alphas[j] < 0.0d)
                    this.alphas[j] = 0.0d;
                else if(this.alphas[j] > c)
                    this.alphas[j] = c;

                this.alphas[i] = this.alphas[i] + labelI * labelJ * (oldAlphaJ - this.alphas[j]);
            }

            changedArray[i] = Math.abs(this.alphas[i] - oldAlphaI) > DEFAULT_TOLERANCE;
            changedArray[j] = Math.abs(this.alphas[j] - oldAlphaJ) > DEFAULT_TOLERANCE;
            gradients = updateGradients();
        }

        // calculate rho
        this.alphas = new double[this.alphas.length];
        ArrayList<Integer> indexList = new ArrayList<Integer>();
        for(int i=0;i<this.alphas.length;i++)
        {
            this.alphas[i] = this.alphas[i];
            if(0.0d < this.alphas[i] && this.alphas[i] < c)
                indexList.add(i);
        }
        double intercept = 0.0d;
        for(int index : indexList)
            intercept += gradients[index];

        this.rho = intercept / (double)indexList.size();
        for(int i=0;i<this.kernelMatrix.length;i++)
            for(int j=i;j<this.kernelMatrix[0].length;j++)
                this.kernelMatrix[i][j] = this.kernelMatrix[j][i] =kernelMatrix[i][j];
    }

    private void trainTaxAndDuin()
    {

    }

    @Override
    public void train(FeatureVector[] featureVectors)
    {
        this.trainedFeatureVectors = new FeatureVector[featureVectors.length];
        for(int i=0;i<featureVectors.length;i++)
            this.trainedFeatureVectors[i] = featureVectors[i];

        if(this.svmType.equals(SCHOLKOPF))
            trainScholkopf();
        else if(this.svmType.equals(TAX_AND_DUIN))
            trainTaxAndDuin();
        else
            System.err.println(this.svmType + " is an invalid SVM type.");
    }

    @Override
    public void train(List<FeatureVector> featureVectorList)
    {
        train((FeatureVector[]) featureVectorList.toArray());
    }

    private int predictScholkopf(FeatureVector featureVector)
    {
        double score = 0.0d;
        if(this.kernelParams.length > 1)
            for(int i=0;i<this.alphas.length;i++)
                score += this.alphas[i] * BasicAlgebra.kernelFunction(this.trainedFeatureVectors[i].getAllValues(), featureVector.getAllValues(), this.kernelType, this.kernelParams);
        else
            for(int i=0;i<this.alphas.length;i++)
                score += this.alphas[i] * BasicAlgebra.kernelFunction(this.trainedFeatureVectors[i].getAllValues(), featureVector.getAllValues(), this.kernelType);

        return BasicMath.sgn(score - this.rho);
    }

    private int predictWithoutTraining()
    {
        System.err.println("The train method must be called before the predict method.");
        return Integer.MIN_VALUE;
    }

    @Override
    public int predict(FeatureVector featureVector)
    {
        if(this.svmType.equals(SCHOLKOPF))
        {
            if(this.rho != Double.NaN)
                return predictScholkopf(featureVector);

            else
                return predictWithoutTraining();
        }

        if(this.radius == Double.NaN)
            return predictWithoutTraining();

        return predictWithoutTraining();
    }

    @Override
    public double leaveOneOutCrossValidation(List<FeatureVector> featureVectorList)
    {
        int successCount = 0;
        int size = featureVectorList.size();
        for(int i=0;i<size;i++)
        {
            FeatureVector testFeatureVector = featureVectorList.get(0);
            featureVectorList.remove(0);
            train(featureVectorList);
            if(predict(testFeatureVector) == NORMAL_VALUE)
                successCount++;

            featureVectorList.add(testFeatureVector);
        }

        // return accuracy
        return (double)successCount / (double)size;
    }

    @Override
    public double leaveOneOutCrossValidation(FeatureVector[] featureVectors)
    {
        return leaveOneOutCrossValidation(Arrays.asList(featureVectors));
    }

    @Override
    public void reset()
    {
        this.alphas = new double[0];
        this.kernelMatrix = new double[0][0];
        this.rho = Double.NaN;
        this.radius = Double.NaN;
        this.trainedFeatureVectors = new FeatureVector[0];
    }

    public void reset(double regParam, double tolerance, String kernelType, double[] kernelParams)
    {
        reset();
        this.regParam = regParam;
        this.tolerance = tolerance;
        this.kernelType = kernelType;
        this.kernelParams = new double[kernelParams.length];
        for(int i=0;i<kernelParams.length;i++)
            this.kernelParams[i] = kernelParams[i];
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

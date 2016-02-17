package jp.mylib.science.classification;

import jp.mylib.science.common.BasicAlgebra;
import jp.mylib.science.common.BasicMath;
import jp.mylib.science.common.FeatureVector;

import java.io.*;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public class OneClassSvm extends Svm
{
    public static final String ONE_CLASS_SVM = "One-class SVM";
    public static final String SCHOLKOPF = "Scholkopf";
    public static final String TAX_AND_DUIN = "Tax and Duin";
    public static final String DELIMITER = "\t";
    public static final double WSS3_TAU = 1.0e-12d;
    public static final double DEFAULT_TOLERANCE = 1.0e-3d;
    public static final int NORMAL_LABEL = 1;
    public static final int OUTLIER_LABEL = -1;
    private String id, method, kernelType;
    private double regParam, tolerance, rho, squaredRadius;
    private double[] kernelParams, alphas, gradients;
    private double[][] kernelMatrix;
    private FeatureVector[] trainingFeatureVectors;

    public OneClassSvm(String id, double regParam, double tolerance, String method, String kernelType, double[] kernelParams)
    {
        this.id = id;
        this.regParam = regParam;
        this.method = method;
        this.kernelType = kernelType;
        this.tolerance = tolerance;
        this.rho = Double.NaN;
        this.squaredRadius = Double.NaN;
        this.kernelParams = new double[kernelParams.length];
        for(int i=0;i<this.kernelParams.length;i++)
            this.kernelParams[i] = kernelParams[i];
    }

    public OneClassSvm(String id, double regParam, double tolerance, String method, String kernelType, double kernelParam)
    {
        this(id, regParam, tolerance, method, kernelType, new double[]{kernelParam});
    }

    public OneClassSvm(String id, double regParam, double tolerance, String method)
    {
        this(id, regParam, tolerance, method, BasicAlgebra.GAUSSIAN_KERNEL_TYPE, new double[]{BasicAlgebra.DEFAULT_GAUSSIAN_KERNEL_SD});
    }

    public OneClassSvm(String id, double regParam)
    {
        this(id, regParam, DEFAULT_TOLERANCE, SCHOLKOPF, BasicAlgebra.GAUSSIAN_KERNEL_TYPE, new double[]{BasicAlgebra.DEFAULT_GAUSSIAN_KERNEL_SD});
    }

    public OneClassSvm(String modelFilePath)
    {
        this.rho = Double.NaN;
        this.squaredRadius = Double.NaN;
        inputModel(modelFilePath);
    }

    // R. Fan et. al. "Working Set Selection Using Second Order Information for Training Support Vector Machines"
    private void solveQpUsingWss3(int trainingSize, double c)
    {
        this.kernelMatrix = calcKernelMatrix(this.trainingFeatureVectors, this.kernelType, this.kernelParams);
        // initialize an alpha array (Working Set Selection 3)
        this.alphas = new double[trainingSize];
        int[] labels = new int[trainingSize];
        this.gradients = new double[trainingSize];
        double vl = this.regParam * (double)trainingSize;
        for(int i=0;i<this.alphas.length;i++)
        {
            labels[i] = NORMAL_LABEL;
            if(i < (int)Math.floor(vl) - 1)
                this.alphas[i] = 1.0d / vl;
            else if(i < (int)Math.floor(vl))
                this.alphas[i] = vl - Math.floor(vl);
            else
                this.alphas[i] = 0.0d;
        }

        for(int i=0;i<gradients.length;i++)
            gradients[i] = BasicAlgebra.calcInnerProduct(this.kernelMatrix[i], this.alphas);

        while(true)
        {
            int[] workingSet = SvmUtil.workingSetSelection3(c, WSS3_TAU, this.tolerance, labels, this.kernelMatrix, this.alphas, gradients);
            int i = workingSet[0];
            int j = workingSet[1];
            if(j == -1)
                break;

            double a = this.kernelMatrix[i][i] + this.kernelMatrix[j][j] - 2.0d * (double)(labels[i] * labels[j]) * this.kernelMatrix[i][j];
            double b = -(double)labels[i] * gradients[i] + (double)labels[j] * gradients[j];
            if(a <= 0.0d)
                a = WSS3_TAU;

            // update alphas
            double oldAlphaI = this.alphas[i];
            double oldAlphaJ = this.alphas[j];
            this.alphas[i] += (double)labels[i] * b / a;
            this.alphas[j] -= (double)labels[j] * b / a;
            // project alphas back to the feasible region
            double sum = (double)labels[i] * oldAlphaI + (double)labels[j] * oldAlphaJ;
            if(this.alphas[i] > c)
                this.alphas[i] = c;

            if(this.alphas[i] < 0.0d)
                this.alphas[i] = 0.0d;

            this.alphas[j] = (double)labels[j] * (sum - (double)labels[i] * this.alphas[i]);
            if(this.alphas[j] > c)
                this.alphas[j] = c;

            if(this.alphas[j] < 0.0d)
                this.alphas[j] = 0.0d;

            this.alphas[i] = (double)labels[i] * (sum - (double)labels[j] * this.alphas[j]);
            // update gradients
            double deltaAlphaI = this.alphas[i] - oldAlphaI;
            double deltaAlphaJ = this.alphas[j] - oldAlphaJ;
            for(int t = 0;t<gradients.length;t++)
                gradients[t] += this.kernelMatrix[t][i] * deltaAlphaI + this.kernelMatrix[t][j] * deltaAlphaJ;
        }
    }

    // B. Scholkopf et. al. "Support Vector Method for Novelty Detection"
    private void trainScholkopf()
    {
        int trainingSize = this.trainingFeatureVectors.length;
        double c = 1.0d / ((double)trainingSize * this.regParam);
        solveQpUsingWss3(trainingSize, c);
        // calculate rho
        ArrayList<Integer> indexList = new ArrayList<Integer>();
        for(int i=0;i<this.alphas.length;i++)
            if(0.0d < this.alphas[i] && this.alphas[i] < c)
                indexList.add(i);

        double intercept = 0.0d;
        for(int index : indexList)
            intercept += this.gradients[index];

        this.rho = intercept / (double)indexList.size();
    }

    // D. Tax and R. Duin "Support Vector Data Description"
    private void trainTaxAndDuin()
    {
        int trainingSize = this.trainingFeatureVectors.length;
        double c = this.regParam;
        solveQpUsingWss3(trainingSize, c);
        // calculate radius
        ArrayList<Integer> indexList = new ArrayList<Integer>();
        for(int i=0;i<this.alphas.length;i++)
            if(0.0d < this.alphas[i] && this.alphas[i] < c)
                indexList.add(i);

        // here, adopt the first index(k) satisfying 0 < a[k] < c
        int k = indexList.get(0);
        double r2 = this.kernelMatrix[k][k];
        double sum = 0.0d;
        for(int i=0;i<this.alphas.length;i++)
            sum += this.alphas[i] * this.kernelMatrix[i][k];

        r2 -= 2.0d * sum;
        for(int i=0;i<this.alphas.length;i++)
            for(int j=0;j<this.alphas.length;j++)
                r2 += this.alphas[i] * this.alphas[j] * this.kernelMatrix[i][j];

        this.squaredRadius = r2;
    }

    @Override
    public void train(FeatureVector[] featureVectors)
    {
        this.trainingFeatureVectors = new FeatureVector[featureVectors.length];
        for(int i=0;i<featureVectors.length;i++)
            this.trainingFeatureVectors[i] = featureVectors[i];

        if(this.method.equals(SCHOLKOPF))
            trainScholkopf();
        else if(this.method.equals(TAX_AND_DUIN))
            trainTaxAndDuin();
        else
            System.err.println(this.method + " is an invalid Svm type.");
    }

    @Override
    public void train(List<FeatureVector> featureVectorList)
    {
        train(featureVectorList.toArray(new FeatureVector[featureVectorList.size()]));
    }

    private int predictScholkopf(FeatureVector featureVector)
    {
        double score = 0.0d;
        if(this.kernelParams.length > 1)
            for(int i=0;i<this.alphas.length;i++)
                score += this.alphas[i] * BasicAlgebra.kernelFunction(this.trainingFeatureVectors[i].getAllValues(), featureVector.getAllValues(), this.kernelType, this.kernelParams);
        else
            for(int i=0;i<this.alphas.length;i++)
                score += this.alphas[i] * BasicAlgebra.kernelFunction(this.trainingFeatureVectors[i].getAllValues(), featureVector.getAllValues(), this.kernelType);

        return (BasicMath.sgn(score - this.rho) == OUTLIER_LABEL)? OUTLIER_LABEL : NORMAL_LABEL;
    }

    private int predictTaxAndDuin(FeatureVector featureVector)
    {
        double score = BasicAlgebra.calcInnerProduct(featureVector.getAllValues(), featureVector.getAllValues());
        double sum = 0.0d;
        for(int i=0;i<this.alphas.length;i++)
            sum += this.alphas[i] * BasicAlgebra.kernelFunction(featureVector.getAllValues(), this.trainingFeatureVectors[i].getAllValues(), this.kernelType, this.kernelParams);

        score -= 2.0d * sum;
        for(int i=0;i<this.alphas.length;i++)
            for(int j=0;j<this.alphas.length;j++)
                score += this.alphas[i] * this.alphas[j] * this.kernelMatrix[i][j];

        return (score > this.squaredRadius)? OUTLIER_LABEL : NORMAL_LABEL;
    }

    private int predictWithoutTraining()
    {
        System.err.println("The train method must be called before the predict method.");
        return Integer.MIN_VALUE;
    }

    @Override
    public int predict(FeatureVector featureVector)
    {
        if(this.method.equals(SCHOLKOPF) && this.rho != Double.NaN)
                return predictScholkopf(featureVector);
        else if(this.method.equals(TAX_AND_DUIN) && this.squaredRadius == Double.NaN)
                return predictTaxAndDuin(featureVector);

        return predictWithoutTraining();
    }

    @Override
    public void reset()
    {
        this.alphas = new double[0];
        this.kernelMatrix = new double[0][0];
        this.rho = Double.NaN;
        this.squaredRadius = Double.NaN;
        this.trainingFeatureVectors = new FeatureVector[0];
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
    public double leaveOneOutCrossValidation(List<FeatureVector> featureVectorList)
    {
        int successCount = 0;
        int size = featureVectorList.size();
        for(int i=0;i<size;i++)
        {
            FeatureVector testFeatureVector = featureVectorList.get(0);
            featureVectorList.remove(0);
            train(featureVectorList);
            if(predict(testFeatureVector) == NORMAL_LABEL)
                successCount++;

            featureVectorList.add(testFeatureVector);
            reset(this.regParam, this.tolerance, this.kernelType, this.kernelParams);
        }

        // return accuracy
        return (double)successCount / (double)size;
    }

    @Override
    public double leaveOneOutCrossValidation(FeatureVector[] featureVectors)
    {
        return leaveOneOutCrossValidation(new ArrayList<FeatureVector>(Arrays.asList(featureVectors)));
    }

    public void doParamsGridSearch(FeatureVector[] featureVectors, double[] regParams, double[][] kernelParamMatrix, boolean changeable)
    {
        // array[0]: start, array[1]: end, array[2]: step size
        // array[x][0]: start, array[x][1]: end, array[x][2]: step size
        double orgRegParam = this.regParam;
        double[] orgKernelParams = new double[this.kernelParams.length];
        for(int i=0;i<orgKernelParams.length;i++)
            orgKernelParams[i] = this.kernelParams[i];

        double bestAccuracy = -Double.MAX_VALUE;
        double bestRegParam = Double.NaN;
        double[] bestKernelParams = new double[kernelParamMatrix.length];
        for(double c=regParams[0];c<=regParams[1];c+=regParams[2])
        {
            if(orgKernelParams.length == 1)
                for(double a=kernelParamMatrix[0][0];a<=kernelParamMatrix[0][1];a+=kernelParamMatrix[0][2])
                {
                    this.regParam = c;
                    this.kernelParams[0] = a;
                    double accuracy = leaveOneOutCrossValidation(featureVectors);
                    if(accuracy > bestAccuracy)
                    {
                        bestAccuracy = accuracy;
                        bestRegParam = c;
                        bestKernelParams[0] = a;
                    }
                }
            else
                for(double a=kernelParamMatrix[0][0];a<=kernelParamMatrix[0][1];a+=kernelParamMatrix[0][2])
                    for(double b=kernelParamMatrix[1][0];b<=kernelParamMatrix[1][1];b+=kernelParamMatrix[1][2])
                    {
                        this.regParam = c;
                        this.kernelParams[0] = a;
                        this.kernelParams[1] = b;
                        double accuracy = leaveOneOutCrossValidation(featureVectors);
                        if(accuracy > bestAccuracy)
                        {
                            bestAccuracy = accuracy;
                            bestRegParam = c;
                            bestKernelParams[0] = a;
                            bestKernelParams[1] = b;
                        }
                    }
        }

        System.out.println("Best accuracy = " + (bestAccuracy * 100.0d));
        System.out.println("Kernel type = " + this.kernelType);
        System.out.println("regParam = " + bestRegParam);
        for(int i=0;i<bestKernelParams.length;i++)
            System.out.println("param" + (i + 1) + " = " + bestKernelParams[i]);

        this.regParam = (changeable)? bestRegParam : orgRegParam;
        for(int i=0;i<orgKernelParams.length;i++)
            this.kernelParams[i] = (changeable)? bestKernelParams[i] : orgKernelParams[i];
    }

    @Override
    public void inputModel(String modelFilePath)
    {
        String errorMsg = "This model is not for One-class SVM.";
        File modelFile = new File(modelFilePath);
        try
        {
            BufferedReader br = new BufferedReader(new FileReader(modelFile));
            if(!br.readLine().equals(ONE_CLASS_SVM))
            {
                System.err.println(errorMsg);
                return;
            }

            this.id = br.readLine().split(DELIMITER)[1];
            this.method = br.readLine().split(DELIMITER)[1];;
            if(!this.method.equals(SCHOLKOPF) && !this.method.equals(TAX_AND_DUIN))
            {
                System.err.println(errorMsg);
                return;
            }

            this.regParam = Double.parseDouble(br.readLine().split(DELIMITER)[1]);
            this.tolerance = Double.parseDouble(br.readLine().split(DELIMITER)[1]);
            String[] params = br.readLine().split(DELIMITER);
            this.kernelType = params[1];
            for(int i=2;i<params.length;i++)
                this.kernelParams[i - 1] = Double.parseDouble(params[i]);

            if(this.method.equals(SCHOLKOPF))
                this.rho = Double.parseDouble(br.readLine().split(DELIMITER)[1]);
            else if(this.method.equals(TAX_AND_DUIN))
                this.squaredRadius = Double.parseDouble(br.readLine().split(DELIMITER)[1]);

            if(!br.readLine().equals("alpha"))
            {
                System.err.println(errorMsg);
                return;
            }

            params = br.readLine().split(DELIMITER);
            this.alphas = new double[params.length];
            for(int i=0;i<this.alphas.length;i++)
                this.alphas[i] = Double.parseDouble(params[i]);

            if(!br.readLine().equals("kernel matrix"))
            {
                System.err.println(errorMsg);
                return;
            }

            this.kernelMatrix = new double[this.alphas.length][this.alphas.length];
            for(int i=0;i<this.alphas.length;i++)
            {
                params = br.readLine().split(DELIMITER);
                for(int j=0;j<params.length;j++)
                    this.kernelMatrix[i][j] = Double.parseDouble(params[j]);
            }

            br.close();
        }
        catch(Exception e)
        {
            System.err.println("Exception @ inputModel(String) : " + e.toString());
        }
    }

    @Override
    public void outputModel(String modelFilePath)
    {
        File modelFile = new File(modelFilePath);
        try
        {
            BufferedWriter bw = new BufferedWriter(new FileWriter(modelFile));
            bw.write(ONE_CLASS_SVM);
            bw.newLine();
            bw.write("id" + DELIMITER + this.id);
            bw.newLine();
            bw.write("method" + DELIMITER + this.method);
            bw.newLine();
            bw.write("regulation param" + DELIMITER + this.regParam);
            bw.newLine();
            bw.write("tolerance" + DELIMITER + this.tolerance);
            bw.newLine();
            bw.write("kernel" + DELIMITER + this.kernelType);
            for(int i=0;i<this.kernelParams.length;i++)
                bw.write("" + DELIMITER + this.kernelParams[i]);

            bw.newLine();
            if(this.method.equals(SCHOLKOPF))
                bw.write("rho" + DELIMITER + this.rho);
            else if(this.method.equals(TAX_AND_DUIN))
                bw.write("squared radius" + DELIMITER + this.squaredRadius);

            bw.newLine();
            bw.write("alpha vector");
            bw.newLine();
            for(int i=0;i<this.alphas.length;i++)
                bw.write((i == 0)? String.valueOf(this.alphas[i]) : DELIMITER + this.alphas[i]);

            bw.newLine();
            bw.write("kernel matrix");
            bw.newLine();
            for(int i=0;i<this.kernelMatrix.length;i++)
                for(int j=0;j<this.kernelMatrix[0].length;j++)
                    bw.write((j == 0)? String.valueOf(this.kernelMatrix[i][j]) : "" + DELIMITER + this.kernelMatrix[i][j]);

            bw.close();
        }
        catch(Exception e)
        {
            System.err.println("Exception @ outputModel(String) : " + e.toString());
        }
    }
}

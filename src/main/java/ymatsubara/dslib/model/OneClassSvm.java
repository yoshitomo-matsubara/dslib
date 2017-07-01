package ymatsubara.dslib.model;

import ymatsubara.dslib.common.BasicAlgebra;
import ymatsubara.dslib.common.BasicMath;
import ymatsubara.dslib.optimization.Decomposition;
import ymatsubara.dslib.statistics.Kernel;
import ymatsubara.dslib.structure.Result;
import ymatsubara.dslib.structure.FeatureVector;
import ymatsubara.dslib.structure.SymmetricMatrix;
import ymatsubara.dslib.util.FeatureVectorUtil;

import java.io.*;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public class OneClassSvm extends AbstractModel {
    public static final String TYPE = "One-class SVM";
    public static final String SCHOLKOPF = "Scholkopf";
    public static final String TAX_AND_DUIN = "Tax and Duin";
    public static final String DELIMITER = "\t";
    public static final String NORMAL_LABEL = "1";
    public static final String OUTLIER_LABEL = "-1";
    public static final double WSS3_TAU = 1.0e-12d;
    public static final int NORMAL_VALUE = 1;
    public static final int OUTLIER_VALUE = -1;
    private String id, method;
    private double regParam, tolerance, rho, squaredRadius;
    private Kernel kernel;
    private double[] alphas, gradients;
    private SymmetricMatrix kernelMatrix;
    private FeatureVector[] trainingVecs;

    public OneClassSvm(String id, double regParam, double tolerance, String method, Kernel kernel) {
        this.id = id;
        this.regParam = regParam;
        this.method = method;
        this.tolerance = tolerance;
        this.kernel = kernel;
        this.rho = Double.NaN;
        this.squaredRadius = Double.NaN;
    }

    public OneClassSvm(String id, double regParam, double tolerance, String method, String kernelType, double[] kernelParams) {
        this(id, regParam, tolerance, method, new Kernel(kernelType, kernelParams));
    }

    public OneClassSvm(String id, double regParam, double tolerance, String method, String kernelType, double kernelParam) {
        this(id, regParam, tolerance, method, kernelType, new double[]{kernelParam});
    }

    public OneClassSvm(String modelFilePath) {
        this.rho = Double.NaN;
        this.squaredRadius = Double.NaN;
        inputModel(modelFilePath);
    }

    // R. Fan et. al. "Working Set Selection Using Second Order Information for Training Support Vector Machines"
    private void solveQpUsingWss3(int trainingSize, double c) {
        this.kernelMatrix = new SymmetricMatrix(this.kernel.calcKernelMatrix(this.trainingVecs));
        // initialize an alpha array (Working Set Selection 3)
        this.alphas = new double[trainingSize];
        int[] labels = new int[trainingSize];
        this.gradients = new double[trainingSize];
        double vl = this.regParam * (double) trainingSize;
        // refer to 4.1.6 p.11
        for (int i = 0; i < this.alphas.length; i++) {
            labels[i] = NORMAL_VALUE;
            if (i < (int) Math.floor(vl)) {
                this.alphas[i] = 1.0d;
            } else if (i < (int) Math.floor(vl) + 1) {
                this.alphas[i] = vl - Math.floor(vl);
            } else {
                this.alphas[i] = 0.0d;
            }
        }

        if (this.method.equals(TAX_AND_DUIN)) {
            for (int i = 0; i < this.alphas.length; i++) {
                this.alphas[i] /= vl;
            }
        }

        for (int i = 0; i < this.gradients.length; i++) {
            this.gradients[i] = BasicAlgebra.calcInnerProduct(this.kernelMatrix.getRow(i), this.alphas);
        }

        while (true) {
            int[] workingSet = Decomposition.workingSetSelection3(c, WSS3_TAU, this.tolerance, labels, this.kernelMatrix, this.alphas, this.gradients);
            int i = workingSet[0];
            int j = workingSet[1];
            if (j == -1) {
                break;
            }

            double a = this.kernelMatrix.get(i, i) + this.kernelMatrix.get(j, j) - 2.0d * (double) (labels[i] * labels[j]) * this.kernelMatrix.get(i, j);
            double b = -(double) labels[i] * this.gradients[i] + (double) labels[j] * this.gradients[j];
            if (a <= 0.0d) {
                a = WSS3_TAU;
            }

            // update alphas
            double oldAlphaI = this.alphas[i];
            double oldAlphaJ = this.alphas[j];
            this.alphas[i] += (double) labels[i] * b / a;
            this.alphas[j] -= (double) labels[j] * b / a;
            // project alphas back to the feasible region
            double sum = (double) labels[i] * oldAlphaI + (double) labels[j] * oldAlphaJ;
            if (this.alphas[i] > c) {
                this.alphas[i] = c;
            }

            if (this.alphas[i] < 0.0d) {
                this.alphas[i] = 0.0d;
            }

            this.alphas[j] = (double) labels[j] * (sum - (double) labels[i] * this.alphas[i]);
            if (this.alphas[j] > c) {
                this.alphas[j] = c;
            }

            if (this.alphas[j] < 0.0d) {
                this.alphas[j] = 0.0d;
            }

            this.alphas[i] = (double) labels[i] * (sum - (double) labels[j] * this.alphas[j]);
            // update gradients
            double deltaAlphaI = this.alphas[i] - oldAlphaI;
            double deltaAlphaJ = this.alphas[j] - oldAlphaJ;
            for (int t = 0; t < this.gradients.length; t++) {
                this.gradients[t] += this.kernelMatrix.get(t, i) * deltaAlphaI + this.kernelMatrix.get(t, j) * deltaAlphaJ;
            }
        }
    }

    // B. Scholkopf et. al. "Support Vector Method for Novelty Detection"
    private void trainScholkopf() {
        int trainingSize = this.trainingVecs.length;
        double c = 1.0d;
        solveQpUsingWss3(trainingSize, c);
        // calculate rho
        List<Integer> indexList = new ArrayList<>();
        for (int i = 0; i < this.alphas.length; i++) {
            if (0.0d < this.alphas[i] && this.alphas[i] < c) {
                indexList.add(i);
            }
        }

        double intercept = 0.0d;
        for (int index : indexList) {
            intercept += this.gradients[index];
        }
        this.rho = intercept / (double) indexList.size();
    }

    // D. Tax and R. Duin "Support Vector Data Description"
    private void trainTaxAndDuin() {
        int trainingSize = this.trainingVecs.length;
        double c = this.regParam;
        solveQpUsingWss3(trainingSize, c);
        // calculate radius
        List<Integer> indexList = new ArrayList<>();
        for (int i = 0; i < this.alphas.length; i++) {
            if (0.0d < this.alphas[i] && this.alphas[i] < c) {
                indexList.add(i);
            }
        }

        // here, adopt the first index(k) satisfying 0 < a[k] < c
        int k = indexList.get(0);
        double r2 = this.kernelMatrix.get(k, k);
        double sum = 0.0d;
        for (int i = 0; i < this.alphas.length; i++) {
            sum += this.alphas[i] * this.kernelMatrix.get(i, k);
        }

        r2 -= 2.0d * sum;
        for (int i = 0; i < this.alphas.length; i++) {
            for (int j = 0; j < this.alphas.length; j++) {
                r2 += this.alphas[i] * this.alphas[j] * this.kernelMatrix.get(i, j);
            }
        }
        this.squaredRadius = r2;
    }

    @Override
    public void train(FeatureVector[] vecs) {
        this.trainingVecs = FeatureVectorUtil.getTargetVectors(vecs, NORMAL_LABEL);
        if (this.method.equals(SCHOLKOPF)) {
            trainScholkopf();
        } else if (this.method.equals(TAX_AND_DUIN)) {
            trainTaxAndDuin();
        } else {
            System.err.println(this.method + " is an invalid Svm type.");
        }
    }

    @Override
    public void train(List<FeatureVector> vecList) {
        train(vecList.toArray(new FeatureVector[vecList.size()]));
    }

    private Result predictScholkopf(FeatureVector vec) {
        double score = 0.0d;
        if (this.kernel.getParams().length > 1) {
            for (int i = 0; i < this.alphas.length; i++) {
                score += this.alphas[i] * this.kernel.kernelFunction(this.trainingVecs[i].getAllValues(), vec.getAllValues());
            }
        } else {
            for (int i = 0; i < this.alphas.length; i++) {
                score += this.alphas[i] * this.kernel.kernelFunction(this.trainingVecs[i].getAllValues(), vec.getAllValues());
            }
        }

        score -= this.rho;
        String label = (BasicMath.sgn(score) == OUTLIER_VALUE) ? OUTLIER_LABEL : NORMAL_LABEL;
        return new Result(vec, score, label);
    }

    private Result predictTaxAndDuin(FeatureVector vec) {
        double score = BasicAlgebra.calcInnerProduct(vec.getAllValues(), vec.getAllValues());
        double sum = 0.0d;
        for (int i = 0; i < this.alphas.length; i++) {
            sum += this.alphas[i] * this.kernel.kernelFunction(vec.getAllValues(), this.trainingVecs[i].getAllValues());
        }

        score -= 2.0d * sum;
        for (int i = 0; i < this.alphas.length; i++) {
            for (int j = 0; j < this.alphas.length; j++) {
                score += this.alphas[i] * this.alphas[j] * this.kernelMatrix.get(i, j);
            }
        }

        score -= this.squaredRadius;
        String label = score > 0.0d ? OUTLIER_LABEL : NORMAL_LABEL;
        return new Result(vec, score, label);
    }

    private Result predictWithoutTraining() {
        System.err.println("The train method must be called before the predict method.");
        return null;
    }

    @Override
    public Result predict(FeatureVector vec) {
        if (this.method.equals(SCHOLKOPF) && this.rho != Double.NaN) {
            return predictScholkopf(vec);
        } else if (this.method.equals(TAX_AND_DUIN) && this.squaredRadius != Double.NaN) {
            return predictTaxAndDuin(vec);
        }
        return predictWithoutTraining();
    }

    @Override
    public void reset() {
        this.alphas = new double[0];
        this.kernelMatrix = new SymmetricMatrix(new double[0][0]);
        this.rho = Double.NaN;
        this.squaredRadius = Double.NaN;
        this.trainingVecs = new FeatureVector[0];
    }

    public void reset(double regParam, double tolerance, Kernel kernel) {
        reset();
        this.regParam = regParam;
        this.tolerance = tolerance;
        this.kernel = kernel;
    }

    public void reset(double regParam, double tolerance, String kernelType, double[] kernelParams) {
        reset(regParam, tolerance, new Kernel(kernelType, kernelParams));
    }

    public double[] doLeaveOneOutCrossValidationTprTfr(List<FeatureVector> vecList) {
        int normalSize = 0;
        int outlierSize = 0;
        int tpCount = 0;
        int tnCount = 0;
        List<Result> resultList = doLeaveOneOutCrossValidation(vecList);
        for (Result result : resultList) {
            if (result.trueLabel.equals(NORMAL_LABEL)) {
                normalSize++;
                if (result.predictedLabel.equals(NORMAL_LABEL)) {
                    tpCount++;
                }
            } else if (result.trueLabel.equals(OUTLIER_LABEL)){
                outlierSize++;
                if (result.predictedLabel.equals(OUTLIER_LABEL)) {
                    tnCount++;
                }
            }
        }

        // return TPR and TFR
        double tpr = normalSize > 0 ? (double) tpCount / (double) normalSize : 0.0d;
        double tnr = outlierSize > 0 ? (double) tnCount / (double) outlierSize : 0.0d;
        return new double[]{tpr, tnr};
    }

    public double[] doLeaveOneOutCrossValidationTprTfr(FeatureVector[] vecs) {
        return doLeaveOneOutCrossValidationTprTfr(Arrays.asList(vecs));
    }

    public void doParamsGridSearch(FeatureVector[] vecs, double[] regParams, double[][] kernelParamMatrix, boolean changeable) {
        // array[0]: start, array[1]: end, array[2]: step size
        // array[x][0]: start, array[x][1]: end, array[x][2]: step size
        double orgRegParam = this.regParam;
        double[] params = this.kernel.getParams();
        double[] orgKernelParams = new double[params.length];
        for (int i = 0; i < orgKernelParams.length; i++) {
            orgKernelParams[i] = params[i];
        }

        double bestAccuracy = -Double.MAX_VALUE;
        double bestRegParam = Double.NaN;
        double[] bestKernelParams = new double[kernelParamMatrix.length];
        for (double c = regParams[0]; c <= regParams[1]; c += regParams[2]) {
            if (orgKernelParams.length == 1) {
                for (double a = kernelParamMatrix[0][0]; a <= kernelParamMatrix[0][1]; a += kernelParamMatrix[0][2]) {
                    this.regParam = c;
                    this.kernel.setParam(a, 0);
                    double[] tprTfr = doLeaveOneOutCrossValidationTprTfr(vecs);
                    double accuracy = BasicMath.calcAverage(tprTfr);
                    if (accuracy > bestAccuracy) {
                        bestAccuracy = accuracy;
                        bestRegParam = c;
                        bestKernelParams[0] = a;
                    }
                }
            } else {
                for (double a = kernelParamMatrix[0][0]; a <= kernelParamMatrix[0][1]; a += kernelParamMatrix[0][2]) {
                    for (double b = kernelParamMatrix[1][0]; b <= kernelParamMatrix[1][1]; b += kernelParamMatrix[1][2]) {
                        this.regParam = c;
                        this.kernel.setParam(a, 0);
                        this.kernel.setParam(b, 1);
                        double[] tprTfr = doLeaveOneOutCrossValidationTprTfr(vecs);
                        double accuracy = BasicMath.calcAverage(tprTfr);
                        if (accuracy > bestAccuracy) {
                            bestAccuracy = accuracy;
                            bestRegParam = c;
                            bestKernelParams[0] = a;
                            bestKernelParams[1] = b;
                        }
                    }
                }
            }
        }

        System.out.println("Best accuracy = " + (bestAccuracy * 100.0d));
        System.out.println("Kernel type = " + this.kernel.getType());
        System.out.println("regParam = " + bestRegParam);
        for (int i = 0; i < bestKernelParams.length; i++) {
            System.out.println("param" + (i + 1) + " = " + bestKernelParams[i]);
        }

        this.regParam = (changeable) ? bestRegParam : orgRegParam;
        this.kernel.setParams((changeable) ? bestKernelParams : orgKernelParams);
    }

    @Override
    public void inputModel(String modelFilePath) {
        String errorMsg = "This model is not for One-class SVM.";
        File modelFile = new File(modelFilePath);
        try {
            BufferedReader br = new BufferedReader(new FileReader(modelFile));
            if (!br.readLine().equals(TYPE)) {
                System.err.println(errorMsg);
                return;
            }

            this.id = br.readLine().split(DELIMITER)[1];
            this.method = br.readLine().split(DELIMITER)[1];
            if (!this.method.equals(SCHOLKOPF) && !this.method.equals(TAX_AND_DUIN)) {
                System.err.println(errorMsg);
                return;
            }

            this.regParam = Double.parseDouble(br.readLine().split(DELIMITER)[1]);
            this.tolerance = Double.parseDouble(br.readLine().split(DELIMITER)[1]);
            String[] params = br.readLine().split(DELIMITER);
            double[] array = new double[params.length - 1];
            for (int i = 2; i < params.length; i++) {
                array[i - 2] = Double.parseDouble(params[i]);
            }

            this.kernel = new Kernel(params[1], array);
            if (this.method.equals(SCHOLKOPF)) {
                this.rho = Double.parseDouble(br.readLine().split(DELIMITER)[1]);
            } else if (this.method.equals(TAX_AND_DUIN)) {
                this.squaredRadius = Double.parseDouble(br.readLine().split(DELIMITER)[1]);
            }

            if (!br.readLine().equals("alpha")) {
                System.err.println(errorMsg);
                return;
            }

            params = br.readLine().split(DELIMITER);
            this.alphas = new double[params.length];
            for (int i = 0; i < this.alphas.length; i++) {
                this.alphas[i] = Double.parseDouble(params[i]);
            }

            if (!br.readLine().equals("kernel matrix")) {
                System.err.println(errorMsg);
                return;
            }

            double[][] matrix = new double[this.alphas.length][this.alphas.length];
            for (int i = 0; i < this.alphas.length; i++) {
                params = br.readLine().split(DELIMITER);
                for (int j = 0; j < params.length; j++) {
                    matrix[i][j] = Double.parseDouble(params[j]);
                }
            }

            this.kernelMatrix = new SymmetricMatrix(matrix);
            br.close();
        } catch (Exception e) {
            System.err.println("Exception @ inputModel(String) : " + e.toString());
        }
    }

    @Override
    public void outputModel(String modelFilePath) {
        File modelFile = new File(modelFilePath);
        try {
            BufferedWriter bw = new BufferedWriter(new FileWriter(modelFile));
            bw.write(TYPE);
            bw.newLine();
            bw.write("id" + DELIMITER + this.id);
            bw.newLine();
            bw.write("method" + DELIMITER + this.method);
            bw.newLine();
            bw.write("regulation param" + DELIMITER + this.regParam);
            bw.newLine();
            bw.write("tolerance" + DELIMITER + this.tolerance);
            bw.newLine();
            bw.write("kernel" + DELIMITER + this.kernel.getType());
            double[] params = this.kernel.getParams();
            for (int i = 0; i < params.length; i++) {
                bw.write("" + DELIMITER + params[i]);
            }

            bw.newLine();
            if (this.method.equals(SCHOLKOPF)) {
                bw.write("rho" + DELIMITER + this.rho);
            } else if (this.method.equals(TAX_AND_DUIN)) {
                bw.write("squared radius" + DELIMITER + this.squaredRadius);
            }

            bw.newLine();
            bw.write("alpha vector");
            bw.newLine();
            for (int i = 0; i < this.alphas.length; i++) {
                bw.write((i == 0) ? String.valueOf(this.alphas[i]) : DELIMITER + this.alphas[i]);
            }

            bw.newLine();
            bw.write("kernel matrix");
            bw.newLine();
            for (int i = 0; i < this.kernelMatrix.getRowSize(); i++) {
                for (int j = 0; j < this.kernelMatrix.getColumnSize(); j++) {
                    bw.write((j == 0) ? String.valueOf(this.kernelMatrix.get(i, j)) : "" + DELIMITER + this.kernelMatrix.get(i, j));
                }
            }
            bw.close();
        } catch (Exception e) {
            System.err.println("Exception @ outputModel(String) : " + e.toString());
        }
    }
}

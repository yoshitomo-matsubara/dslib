package ymatsubara.dslib.optimization;

import ymatsubara.dslib.structure.SymmetricMatrix;

public class Decomposition {
    public static String POSITIVE_LABEL = "1";
    public static String NEGATIVE_LABEL = "-1";
    public static int POSITIVE_VALUE = 1;
    public static int NEGATIVE_VALUE = -1;

    // R. Fan et. al. "Working Set Selection Using Second Order Information for Training Support Vector Machines"
    public static int[] workingSetSelection3(double c, double tau, double tolerance, int[] labels, SymmetricMatrix kernelMatrix, double[] alphas, double[] gradients) {
        // select i
        int i = -1;
        double gradientMax = -Double.MAX_VALUE;
        for (int t = 0; t < labels.length; t++) {
            if ((labels[t] == POSITIVE_VALUE && alphas[t] < c) || (labels[t] == NEGATIVE_VALUE && alphas[t] > 0)) {
                double gradient = -(double) labels[t] * gradients[t];
                if (gradient >= gradientMax) {
                    i = t;
                    gradientMax = gradient;
                }
            }
        }

        // select j
        int j = -1;
        double gradientMin = Double.MAX_VALUE;
        double valueMin = Double.MAX_VALUE;
        for (int t = 0; t < labels.length; t++) {
            if ((labels[t] == POSITIVE_VALUE && alphas[t] > 0) || (labels[t] == NEGATIVE_VALUE && alphas[t] < c)) {
                double b = gradientMax + (double) labels[t] * gradients[t];
                double gradient = -(double) labels[t] * gradients[t];
                if (gradient <= gradientMin) {
                    gradientMin = gradient;
                }

                if (b > 0.0d) {
                    double a = kernelMatrix.get(i, i) + kernelMatrix.get(t, t) - 2.0d * (double) (labels[i] * labels[t]) * kernelMatrix.get(i, t);
                    if (a <= 0.0d) {
                        a = tau;
                    }

                    double value = -Math.pow(b, 2.0d) / a;
                    if (value <= valueMin) {
                        j = t;
                        valueMin = value;
                    }
                }
            }
        }

        if (gradientMax - gradientMin < tolerance) {
            return new int[]{-1, -1};
        }
        return new int[]{i, j};
    }
}
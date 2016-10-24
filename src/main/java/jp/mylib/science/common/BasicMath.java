package jp.mylib.science.common;

import java.util.List;

public class BasicMath {
    public static int sgn(double value) {
        if (value == 0.0d) {
            return 0;
        }
        return (value > 0.0d) ? 1 : -1;
    }

    public static double calcAverage(double[] array) {
        double sum = 0.0d;
        for (double value : array) {
            sum += value;
        }
        return sum / (double) array.length;
    }

    public static double calcAverage(List<Double> list) {
        double sum = 0.0d;
        for (double value : list) {
            sum += value;
        }
        return sum / (double) list.size();
    }

    public static double calcStandardDeviation(double[] array) {
        double ave = calcAverage(array);
        double sum = 0.0d;
        for (double value : array) {
            sum += Math.pow(value - ave, 2.0d);
        }
        return Math.sqrt(sum / (double) array.length);
    }

    public static double calcStandardDeviation(List<Double> list) {
        double ave = calcAverage(list);
        double sum = 0.0d;
        for (double value : list) {
            sum += Math.pow(value - ave, 2.0d);
        }
        return Math.sqrt(sum / (double) list.size());
    }

    public static double calcStandardDeviation(double[] array, double ave) {
        double sum = 0.0d;
        for (double value : array) {
            sum += Math.pow(value - ave, 2.0d);
        }
        return Math.sqrt(sum / (double) array.length);
    }

    public static double calcStandardDeviation(List<Double> list, double ave) {
        double sum = 0.0d;
        for (double value : list) {
            sum += Math.pow(value - ave, 2.0d);
        }
        return Math.sqrt(sum / (double) list.size());
    }
}

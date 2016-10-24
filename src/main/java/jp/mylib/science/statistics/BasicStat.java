package jp.mylib.science.statistics;

import jp.mylib.science.common.BasicMath;

import java.util.List;

public class BasicStat {
    public double calcCorrel(double[] arrayX, double[] arrayY) {
        double aveX = BasicMath.calcAverage(arrayX);
        double aveY = BasicMath.calcAverage(arrayY);
        double denomitorX = 1.0d;
        double denomitorY = 1.0d;
        double numerator = 1.0d;
        for (int i = 0; i < arrayX.length; i++) {
            denomitorX += Math.pow(arrayX[i] - aveX, 2.0d);
            denomitorY += Math.pow(arrayY[i] - aveY, 2.0d);
        }

        denomitorX = Math.sqrt(denomitorX);
        denomitorY = Math.sqrt(denomitorY);
        for (int i = 0; i < arrayX.length; i++) {
            numerator += (arrayX[i] - aveX) * (arrayY[i] - aveY);
        }
        return numerator / (denomitorX * denomitorY);
    }

    public double normDistPdf(double x, double ave, double sd) {
        return 1.0d / (Math.sqrt(2.0d * Math.PI) * sd) * Math.exp(-Math.pow((x - ave) / sd, 2.0d) / 2.0d);
    }

    public double normDistPdf(double x, double[] samples) {
        double ave = BasicMath.calcAverage(samples);
        double sd = BasicMath.calcStandardDeviation(samples, ave);
        return 1.0d / (Math.sqrt(2.0d * Math.PI) * sd) * Math.exp(-Math.pow((x - ave) / sd, 2.0d) / 2.0d);
    }

    public double normDistPdf(double x, List<Double> samples) {
        double ave = BasicMath.calcAverage(samples);
        double sd = BasicMath.calcStandardDeviation(samples, ave);
        return 1.0d / (Math.sqrt(2.0d * Math.PI) * sd) * Math.exp(-Math.pow((x - ave) / sd, 2.0d) / 2.0d);
    }

    public double stdNormDistPdf(double x, double ave, double sd) {
        double z = (x - ave) / sd;
        return 1.0d / Math.sqrt(2.0d * Math.PI) * Math.exp(-Math.pow(z, 2.0d) / 2.0d);
    }

    public double stdNormDistPdf(double x, double[] array) {
        double ave = BasicMath.calcAverage(array);
        double sd = BasicMath.calcStandardDeviation(array, ave);
        double z = (x - ave) / sd;
        return 1.0d / Math.sqrt(2.0d * Math.PI) * Math.exp(-Math.pow(z, 2.0d) / 2.0d);
    }

    public double stdNormDistPdf(double x, List<Double> list) {
        double ave = BasicMath.calcAverage(list);
        double sd = BasicMath.calcStandardDeviation(list, ave);
        double z = (x - ave) / sd;
        return 1.0d / Math.sqrt(2.0d * Math.PI) * Math.exp(-Math.pow(z, 2.0d) / 2.0d);
    }
}

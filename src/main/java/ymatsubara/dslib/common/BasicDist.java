package ymatsubara.dslib.common;

import ymatsubara.dslib.common.BasicMath;

import java.util.List;

public class BasicDist {
    public double calcCorrel(double[] arrayX, double[] arrayY) {
        double aveX = BasicMath.calcAverage(arrayX);
        double aveY = BasicMath.calcAverage(arrayY);
        double denominatorX = 1.0d;
        double denominatorY = 1.0d;
        double numerator = 1.0d;
        for (int i = 0; i < arrayX.length; i++) {
            denominatorX += Math.pow(arrayX[i] - aveX, 2.0d);
            denominatorY += Math.pow(arrayY[i] - aveY, 2.0d);
        }

        denominatorX = Math.sqrt(denominatorX);
        denominatorY = Math.sqrt(denominatorY);
        for (int i = 0; i < arrayX.length; i++) {
            numerator += (arrayX[i] - aveX) * (arrayY[i] - aveY);
        }
        return numerator / (denominatorX * denominatorY);
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

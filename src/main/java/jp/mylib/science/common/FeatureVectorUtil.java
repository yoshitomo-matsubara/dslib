package jp.mylib.science.common;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.util.ArrayList;
import java.util.List;

public class FeatureVectorUtil {
    public static final String COMMENT_OUT = "//";
    public static final String NORMALIZATION = "min-max";
    public static final String STANDARDIZATION = "std";

    /*
    [id] is optional
    Delimiter: [\t], [,] or [ ]
    Comment out: [//]
    Dense vector type
    [id]\t[label]\t[value1]\t[value2]...
    [id]\t[label]\t[value1]\t[value2]...
    [id]\t[label]\t[value1]\t[value2]...
    ...
     */
    public static FeatureVector[] generateFeatureVectors(String inputFilePath, boolean hasId) {
        File inputFile = new File(inputFilePath);
        List<FeatureVector> vecList = new ArrayList<FeatureVector>();
        int startIndex = (hasId)? 2 : 1;
        try {
            BufferedReader br = new BufferedReader(new FileReader(inputFile));
            String line;
            int vecCount = 0;
            while ((line = br.readLine()) != null && line.length() > 0) {
                if (line.startsWith(COMMENT_OUT)) {
                    continue;
                }

                String[] params = line.split("\t");
                if (params.length < startIndex + 1) {
                    params = line.split(",");
                    if (params.length < startIndex + 1) {
                        params = line.split(" ");
                    }
                }

                List<Integer> indexList = new ArrayList<Integer>();
                List<Double> valueList = new ArrayList<Double>();
                String id = (hasId) ? params[0] : String.valueOf(vecCount);
                String label = (hasId) ? params[1] : params[0];
                FeatureVector featureVector = new FeatureVector(id, label, params.length - startIndex);
                for (int i = startIndex; i < params.length; i++) {
                    String[] keyValue = params[i].split(":");
                    if (keyValue.length == 2) {
                        indexList.add(Integer.parseInt(keyValue[0]));
                        valueList.add(Double.parseDouble(keyValue[1]));
                    } else {
                        valueList.add(Double.parseDouble(params[i]));
                    }
                }

                featureVector.setValues(valueList);
                vecList.add(featureVector);
                vecCount++;
            }
            br.close();
        } catch (Exception e) {
            System.err.println("Invalid file for FeatureVector class : " + inputFile.getName());
        }
        return vecList.toArray(new FeatureVector[vecList.size()]);
    }

    public static FeatureVector[] getTargetVectors(FeatureVector[] featureVectors, String targetLabel) {
        ArrayList<FeatureVector> vectorList = new ArrayList<FeatureVector>();
        for (FeatureVector featureVector : featureVectors) {
            if (featureVector.getLabel().equals(targetLabel)) {
                vectorList.add(featureVector);
            }
        }

        FeatureVector[] targetVectors = new FeatureVector[vectorList.size()];
        for (int i = 0; i < targetVectors.length; i++) {
            targetVectors[i] = vectorList.get(i);
        }
        return targetVectors;
    }

    public static List<FeatureVector> getTargetVectorList(List<FeatureVector> featureVectorList, String targetLabel) {
        ArrayList<FeatureVector> vectorList = new ArrayList<FeatureVector>();
        for (FeatureVector featureVector : featureVectorList) {
            if (featureVector.getLabel().equals(targetLabel)) {
                vectorList.add(featureVector);
            }
        }
        return vectorList;
    }

    public static void getEachIndexMinMax(FeatureVector[] featureVectors, double[] minValues, double[] maxValues) {
        for (int i = 0; i < minValues.length; i++) {
            minValues[i] = featureVectors[0].getValue(i);
            maxValues[i] = featureVectors[0].getValue(i);
        }

        for (int i = 1; i < featureVectors.length; i++) {
            for (int j = 0; j < minValues.length; j++) {
                double value = featureVectors[i].getValue(j);
                if (value < minValues[j]) {
                    minValues[j] = value;
                }

                if (value > maxValues[j]) {
                    maxValues[j] = value;
                }
            }
        }
    }

    public static void getEachIndexMinMax(List<FeatureVector> vecList, double[] minValues, double[] maxValues) {
        getEachIndexMinMax(vecList.toArray(new FeatureVector[vecList.size()]), minValues, maxValues);
    }

    public static void getEachIndexAveSd(FeatureVector[] featureVectors, double[] aveValues, double[] sdValues) {
        double[][] matrix = new double[aveValues.length][featureVectors.length];
        for (int i = 0; i < featureVectors.length; i++) {
            for (int j = 0; j < aveValues.length; j++) {
                matrix[j][i] = featureVectors[i].getValue(j);
            }
        }

        for (int i = 0; i < featureVectors.length; i++) {
            aveValues[i] = BasicMath.calcAverage(matrix[i]);
            sdValues[i] = BasicMath.calcStandardDeviation(matrix[i], aveValues[i]);
        }
    }

    public static void getEachIndexAveSd(List<FeatureVector> vecList, double[] aveValues, double[] sdValues) {
        getEachIndexAveSd(vecList.toArray(new FeatureVector[vecList.size()]), aveValues, sdValues);
    }

    public static void doScaling(FeatureVector featureVector, double[] valuesX, double[] valuesY, String type) {
        if (type.equals(NORMALIZATION)) {
            double[] scaledValues = new double[featureVector.getSize()];
            for (int j = 0; j < scaledValues.length; j++) {
                scaledValues[j] = DataProcessor.normalize(featureVector.getValue(j), valuesX[j], valuesY[j]);
            }
            featureVector.replaceAllValues(scaledValues);
        } else if (type.equals(STANDARDIZATION)) {
            double[] scaledValues = new double[featureVector.getSize()];
            for (int j = 0; j < scaledValues.length; j++) {
                scaledValues[j] = DataProcessor.standardize(featureVector.getValue(j), valuesX[j], valuesY[j]);
            }
            featureVector.replaceAllValues(scaledValues);
        }
    }

    public static void doScaling(FeatureVector[] featureVectors, double[] valuesX, double[] valuesY, String type) {
        if (type.equals(NORMALIZATION)) {
            for (int i = 0; i < featureVectors.length; i++) {
                double[] scaledValues = new double[featureVectors[i].getSize()];
                for (int j = 0; j < scaledValues.length; j++) {
                    scaledValues[j] = DataProcessor.normalize(featureVectors[i].getValue(j), valuesX[j], valuesY[j]);
                }
                featureVectors[i].replaceAllValues(scaledValues);
            }
        } else if (type.equals(STANDARDIZATION)) {
            for (int i = 0; i < featureVectors.length; i++) {
                double[] scaledValues = new double[featureVectors[i].getSize()];
                for (int j = 0; j < scaledValues.length; j++) {
                    scaledValues[j] = DataProcessor.standardize(featureVectors[i].getValue(j), valuesX[j], valuesY[j]);
                }
                featureVectors[i].replaceAllValues(scaledValues);
            }
        }
    }

    public static void doScaling(List<FeatureVector> vecList, double[] valuesX, double[] valuesY, String type) {
        int vecSize = vecList.size();
        if (type.equals(NORMALIZATION)) {
            for (int i = 0; i < vecSize; i++) {
                double[] scaledValues = new double[vecList.get(i).getSize()];
                for (int j = 0; j < scaledValues.length; j++) {
                    scaledValues[j] = DataProcessor.normalize(vecList.get(i).getValue(j), valuesX[j], valuesY[j]);
                }
                vecList.get(i).replaceAllValues(scaledValues);
            }
        } else if (type.equals(STANDARDIZATION)) {
            for (int i = 0; i < vecSize; i++) {
                double[] scaledValues = new double[vecList.get(i).getSize()];
                for (int j = 0; j < scaledValues.length; j++) {
                    scaledValues[j] = DataProcessor.standardize(vecList.get(i).getValue(j), valuesX[j], valuesY[j]);
                }
                vecList.get(i).replaceAllValues(scaledValues);
            }
        }
    }

    public static void doScaling(FeatureVector[] featureVectors, FeatureVector[] baseVectors, String type) {
        if (type.equals(NORMALIZATION)) {
            double[] minValues = new double[featureVectors[0].getSize()];
            double[] maxValues = new double[featureVectors[0].getSize()];
            getEachIndexMinMax(baseVectors, minValues, maxValues);
            doScaling(featureVectors, minValues, maxValues, type);
        } else if (type.equals(STANDARDIZATION)) {
            double[] aveValues = new double[featureVectors[0].getSize()];
            double[] sdValues = new double[featureVectors[0].getSize()];
            getEachIndexAveSd(baseVectors, aveValues, sdValues);
            doScaling(featureVectors, aveValues, sdValues, type);
        }
    }

    public static void doScaling(List<FeatureVector> vecList, List<FeatureVector> baseVecList, String type) {
        if (type.equals(NORMALIZATION)) {
            double[] minValues = new double[vecList.get(0).getSize()];
            double[] maxValues = new double[minValues.length];
            getEachIndexMinMax(baseVecList.toArray(new FeatureVector[baseVecList.size()]), minValues, maxValues);
            doScaling(vecList, minValues, maxValues, type);
        } else if (type.equals(STANDARDIZATION)) {
            double[] aveValues = new double[vecList.get(0).getSize()];
            double[] sdValues = new double[aveValues.length];
            getEachIndexAveSd(baseVecList.toArray(new FeatureVector[baseVecList.size()]), aveValues, sdValues);
            doScaling(vecList, aveValues, sdValues, type);
        }
    }
}

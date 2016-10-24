package jp.mylib.science.common;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.util.ArrayList;
import java.util.List;

public class SparseFeatureVectorUtil {
    public static final String COMMENT_OUT = "//";
    public static final String NORMALIZATION = "Normalization";
    public static final String STANDARDIZATION = "Standardization";

    /*
    [id] is optional
    Delimiter: [\t], [,] or [ ]
    Comment out: [//]
    Sparse vector type
    [id]\t[label]\t[index1:value1]\t[index2:value2]\t[index3:value3]...
    [id]\t[label]\t[index3:value3]\t[index4:value4]\t[index7:value7]...
    [id]\t[label]\t[index2:value2]\t[index5:value5]\t[index7:value7]...
    ...
     */
    public static SparseFeatureVector[] generateSparseFeatureVectors(String inputFilePath, boolean hasId) {
        File inputFile = new File(inputFilePath);
        List<SparseFeatureVector> vecList = new ArrayList<>();
        int startIndex = (hasId) ? 2 : 1;
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

                List<Integer> indexList = new ArrayList<>();
                List<Double> valueList = new ArrayList<>();
                String id = (hasId) ? params[0] : String.valueOf(vecCount);
                String label = (hasId) ? params[1] : params[0];
                SparseFeatureVector SparseFeatureVector = new SparseFeatureVector(id, label, params.length - startIndex);
                for (int i = startIndex; i < params.length; i++) {
                    String[] keyValue = params[i].split(":");
                    if (keyValue.length == 2) {
                        indexList.add(Integer.parseInt(keyValue[0]));
                        valueList.add(Double.parseDouble(keyValue[1]));
                    } else {
                        valueList.add(Double.parseDouble(params[i]));
                    }
                }

                SparseFeatureVector.setValues(valueList, indexList);
                vecList.add(SparseFeatureVector);
                vecCount++;
            }
            br.close();
        } catch (Exception e) {
            System.err.println("Invalid file for SparseFeatureVector class : " + inputFile.getName());
        }
        return vecList.toArray(new SparseFeatureVector[vecList.size()]);
    }

    public static FeatureVector[] convertToFeatureVectors(SparseFeatureVector[] vectors) {
        int maxIndex = Integer.MIN_VALUE;
        for (int i = 0; i < vectors.length; i++) {
            int[] indices = vectors[i].getAllIndices();
            if (indices[indices.length - 1] > maxIndex) {
                maxIndex = indices[indices.length - 1];
            }
        }

        FeatureVector[] featureVectors = new FeatureVector[vectors.length];
        for (int i = 0; i < vectors.length; i++) {
            double[] orgValues = vectors[i].getAllValues();
            int[] orgIndices = vectors[i].getAllIndices();
            for (int j = 0; j < orgIndices.length; j++) {
                orgIndices[j]--;
            }

            double[] values = new double[maxIndex];
            for (int j = 0; j < orgIndices[0]; j++) {
                values[j] = 0.0d;
            }

            int nextIndex = orgIndices[0];
            for (int j = 0; j < orgIndices.length; j++) {
                if (orgIndices[j] > nextIndex) {
                    for (int k = nextIndex; k < orgIndices[j]; k++) {
                        values[k] = 0.0d;
                    }
                }

                values[orgIndices[j]] = orgValues[j];
                nextIndex = orgIndices[j] + 1;
            }

            for (int j = orgIndices[orgIndices.length - 1]; j < maxIndex; j++) {
                values[j] = 0.0d;
            }

            featureVectors[i] = new FeatureVector(vectors[i].getId(), vectors[i].getLabel(), maxIndex);
            featureVectors[i].setValues(values);
        }
        return featureVectors;
    }

    public static SparseFeatureVector[] getTargetVectors(SparseFeatureVector[] SparseFeatureVectors, String targetLabel) {
        List<SparseFeatureVector> vectorList = new ArrayList<>();
        for (SparseFeatureVector SparseFeatureVector : SparseFeatureVectors) {
            if (SparseFeatureVector.getLabel().equals(targetLabel)) {
                vectorList.add(SparseFeatureVector);
            }
        }

        SparseFeatureVector[] targetVectors = new SparseFeatureVector[vectorList.size()];
        for (int i = 0; i < targetVectors.length; i++) {
            targetVectors[i] = vectorList.get(i);
        }
        return targetVectors;
    }

    public static List<SparseFeatureVector> getTargetVectorList(List<SparseFeatureVector> SparseFeatureVectorList, String targetLabel) {
        List<SparseFeatureVector> vectorList = new ArrayList<>();
        for (SparseFeatureVector SparseFeatureVector : SparseFeatureVectorList) {
            if (SparseFeatureVector.getLabel().equals(targetLabel)) {
                vectorList.add(SparseFeatureVector);
            }
        }
        return vectorList;
    }

    public static void getEachIndexMinMax(SparseFeatureVector[] SparseFeatureVectors, double[] minValues, double[] maxValues) {
        for (int i = 0; i < minValues.length; i++) {
            minValues[i] = SparseFeatureVectors[0].getValue(i);
            maxValues[i] = SparseFeatureVectors[0].getValue(i);
        }

        for (int i = 1; i < SparseFeatureVectors.length; i++) {
            for (int j = 0; j < minValues.length; j++) {
                double value = SparseFeatureVectors[i].getValue(j);
                if (value < minValues[j]) {
                    minValues[j] = value;
                }

                if (value > maxValues[j]) {
                    maxValues[j] = value;
                }
            }
        }
    }

    public static void getEachIndexAveSd(SparseFeatureVector[] SparseFeatureVectors, double[] aveValues, double[] sdValues) {
        double[][] matrix = new double[aveValues.length][SparseFeatureVectors.length];
        for (int i = 0; i < SparseFeatureVectors.length; i++) {
            for (int j = 0; j < aveValues.length; j++) {
                matrix[j][i] = SparseFeatureVectors[i].getValue(j);
            }
        }

        for (int i = 0; i < SparseFeatureVectors.length; i++) {
            aveValues[i] = BasicMath.calcAverage(matrix[i]);
            sdValues[i] = BasicMath.calcStandardDeviation(matrix[i], aveValues[i]);
        }
    }

    public static void doScaling(SparseFeatureVector[] SparseFeatureVectors, SparseFeatureVector[] baseVectors, String type) {
        if (type.equals(NORMALIZATION)) {
            double[] minValues = new double[SparseFeatureVectors[0].getSize()];
            double[] maxValues = new double[SparseFeatureVectors[0].getSize()];
            getEachIndexMinMax(baseVectors, minValues, maxValues);
            for (int i = 0; i < SparseFeatureVectors.length; i++) {
                double[] scaledValues = new double[SparseFeatureVectors[i].getSize()];
                for (int j = 0; j < scaledValues.length; j++) {
                    scaledValues[j] = DataProcessor.normalize(SparseFeatureVectors[i].getValue(j), minValues[j], maxValues[j]);
                }
                SparseFeatureVectors[i].replaceAllValues(scaledValues);
            }
        } else if (type.equals(STANDARDIZATION)) {
            double[] aveValues = new double[SparseFeatureVectors[0].getSize()];
            double[] sdValues = new double[SparseFeatureVectors[0].getSize()];
            getEachIndexAveSd(baseVectors, aveValues, sdValues);
            for (int i = 0; i < SparseFeatureVectors.length; i++) {
                double[] scaledValues = new double[SparseFeatureVectors[i].getSize()];
                for (int j = 0; j < scaledValues.length; j++) {
                    scaledValues[j] = DataProcessor.standardize(SparseFeatureVectors[i].getValue(j), aveValues[j], sdValues[j]);
                }
                SparseFeatureVectors[i].replaceAllValues(scaledValues);
            }
        }
    }

    public static void doScaling(SparseFeatureVector[] SparseFeatureVectors, String type) {
        doScaling(SparseFeatureVectors, SparseFeatureVectors, type);
    }

    public static void doScaling(SparseFeatureVector[] SparseFeatureVectors) {
        doScaling(SparseFeatureVectors, NORMALIZATION);
    }
}

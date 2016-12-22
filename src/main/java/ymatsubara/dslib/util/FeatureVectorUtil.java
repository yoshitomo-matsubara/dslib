package ymatsubara.dslib.util;

import ymatsubara.dslib.common.BasicMath;
import ymatsubara.dslib.structure.FeatureVector;

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
        List<FeatureVector> vecList = new ArrayList<>();
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

                List<Integer> indexList = new ArrayList<>();
                List<Double> valueList = new ArrayList<>();
                String id = (hasId) ? params[0] : String.valueOf(vecCount);
                String label = (hasId) ? params[1] : params[0];
                FeatureVector vec = new FeatureVector(id, label, params.length - startIndex);
                for (int i = startIndex; i < params.length; i++) {
                    String[] keyValue = params[i].split(":");
                    if (keyValue.length == 2) {
                        indexList.add(Integer.parseInt(keyValue[0]));
                        valueList.add(Double.parseDouble(keyValue[1]));
                    } else {
                        valueList.add(Double.parseDouble(params[i]));
                    }
                }

                vec.setValues(valueList);
                vecList.add(vec);
                vecCount++;
            }
            br.close();
        } catch (Exception e) {
            System.err.println("Invalid file for FeatureVector class : " + inputFile.getName());
        }
        return vecList.toArray(new FeatureVector[vecList.size()]);
    }

    public static FeatureVector[] getTargetVectors(FeatureVector[] vecs, String targetLabel) {
        List<FeatureVector> vecList = new ArrayList<>();
        for (FeatureVector vec : vecs) {
            if (vec.getLabel().equals(targetLabel)) {
                vecList.add(vec);
            }
        }

        FeatureVector[] targetVecs = new FeatureVector[vecList.size()];
        for (int i = 0; i < targetVecs.length; i++) {
            targetVecs[i] = vecList.get(i);
        }
        return targetVecs;
    }

    public static List<FeatureVector> getTargetVectorList(List<FeatureVector> vecList, String targetLabel) {
        List<FeatureVector> targetVecList = new ArrayList<>();
        for (FeatureVector vec : vecList) {
            if (vec.getLabel().equals(targetLabel)) {
                targetVecList.add(vec);
            }
        }
        return targetVecList;
    }

    public static void getEachIndexMinMax(FeatureVector[] vecs, double[] minValues, double[] maxValues) {
        for (int i = 0; i < minValues.length; i++) {
            minValues[i] = vecs[0].getValue(i);
            maxValues[i] = vecs[0].getValue(i);
        }

        for (int i = 1; i < vecs.length; i++) {
            for (int j = 0; j < minValues.length; j++) {
                double value = vecs[i].getValue(j);
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

    public static void getEachIndexAveSd(FeatureVector[] vecs, double[] aveValues, double[] sdValues) {
        double[][] matrix = new double[aveValues.length][vecs.length];
        for (int i = 0; i < vecs.length; i++) {
            for (int j = 0; j < aveValues.length; j++) {
                matrix[j][i] = vecs[i].getValue(j);
            }
        }

        for (int i = 0; i < vecs.length; i++) {
            aveValues[i] = BasicMath.calcAverage(matrix[i]);
            sdValues[i] = BasicMath.calcStandardDeviation(matrix[i], aveValues[i]);
        }
    }

    public static void getEachIndexAveSd(List<FeatureVector> vecList, double[] aveValues, double[] sdValues) {
        getEachIndexAveSd(vecList.toArray(new FeatureVector[vecList.size()]), aveValues, sdValues);
    }

    public static void doScaling(FeatureVector vec, double[] valuesX, double[] valuesY, String type) {
        if (type.equals(NORMALIZATION)) {
            double[] scaledValues = new double[vec.getSize()];
            for (int j = 0; j < scaledValues.length; j++) {
                scaledValues[j] = DataProcessUtil.normalize(vec.getValue(j), valuesX[j], valuesY[j]);
            }
            vec.replaceAllValues(scaledValues);
        } else if (type.equals(STANDARDIZATION)) {
            double[] scaledValues = new double[vec.getSize()];
            for (int j = 0; j < scaledValues.length; j++) {
                scaledValues[j] = DataProcessUtil.standardize(vec.getValue(j), valuesX[j], valuesY[j]);
            }
            vec.replaceAllValues(scaledValues);
        }
    }

    public static void doScaling(FeatureVector[] vecs, double[] valuesX, double[] valuesY, String type) {
        if (type.equals(NORMALIZATION)) {
            for (int i = 0; i < vecs.length; i++) {
                double[] scaledValues = new double[vecs[i].getSize()];
                for (int j = 0; j < scaledValues.length; j++) {
                    scaledValues[j] = DataProcessUtil.normalize(vecs[i].getValue(j), valuesX[j], valuesY[j]);
                }
                vecs[i].replaceAllValues(scaledValues);
            }
        } else if (type.equals(STANDARDIZATION)) {
            for (int i = 0; i < vecs.length; i++) {
                double[] scaledValues = new double[vecs[i].getSize()];
                for (int j = 0; j < scaledValues.length; j++) {
                    scaledValues[j] = DataProcessUtil.standardize(vecs[i].getValue(j), valuesX[j], valuesY[j]);
                }
                vecs[i].replaceAllValues(scaledValues);
            }
        }
    }

    public static void doScaling(List<FeatureVector> vecList, double[] valuesX, double[] valuesY, String type) {
        int vecSize = vecList.size();
        if (type.equals(NORMALIZATION)) {
            for (int i = 0; i < vecSize; i++) {
                double[] scaledValues = new double[vecList.get(i).getSize()];
                for (int j = 0; j < scaledValues.length; j++) {
                    scaledValues[j] = DataProcessUtil.normalize(vecList.get(i).getValue(j), valuesX[j], valuesY[j]);
                }
                vecList.get(i).replaceAllValues(scaledValues);
            }
        } else if (type.equals(STANDARDIZATION)) {
            for (int i = 0; i < vecSize; i++) {
                double[] scaledValues = new double[vecList.get(i).getSize()];
                for (int j = 0; j < scaledValues.length; j++) {
                    scaledValues[j] = DataProcessUtil.standardize(vecList.get(i).getValue(j), valuesX[j], valuesY[j]);
                }
                vecList.get(i).replaceAllValues(scaledValues);
            }
        }
    }

    public static void doScaling(FeatureVector[] vecs, FeatureVector[] baseVecs, String type) {
        if (type.equals(NORMALIZATION)) {
            double[] minValues = new double[vecs[0].getSize()];
            double[] maxValues = new double[minValues.length];
            getEachIndexMinMax(baseVecs, minValues, maxValues);
            doScaling(vecs, minValues, maxValues, type);
        } else if (type.equals(STANDARDIZATION)) {
            double[] aveValues = new double[vecs[0].getSize()];
            double[] sdValues = new double[aveValues.length];
            getEachIndexAveSd(baseVecs, aveValues, sdValues);
            doScaling(vecs, aveValues, sdValues, type);
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

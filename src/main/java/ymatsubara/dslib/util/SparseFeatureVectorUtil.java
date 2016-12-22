package ymatsubara.dslib.util;

import ymatsubara.dslib.common.BasicMath;
import ymatsubara.dslib.structure.FeatureVector;
import ymatsubara.dslib.structure.SparseFeatureVector;

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

    public static FeatureVector[] convertToFeatureVectors(SparseFeatureVector[] sparseVecs) {
        int maxIndex = Integer.MIN_VALUE;
        for (int i = 0; i < sparseVecs.length; i++) {
            int[] indices = sparseVecs[i].getAllIndices();
            if (indices[indices.length - 1] > maxIndex) {
                maxIndex = indices[indices.length - 1];
            }
        }

        FeatureVector[] featureVecs = new FeatureVector[sparseVecs.length];
        for (int i = 0; i < sparseVecs.length; i++) {
            double[] orgValues = sparseVecs[i].getAllValues();
            int[] orgIndices = sparseVecs[i].getAllIndices();
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

            featureVecs[i] = new FeatureVector(sparseVecs[i].getId(), sparseVecs[i].getLabel(), maxIndex);
            featureVecs[i].setValues(values);
        }
        return featureVecs;
    }

    public static SparseFeatureVector[] getTargetVectors(SparseFeatureVector[] vecs, String targetLabel) {
        List<SparseFeatureVector> vecList = new ArrayList<>();
        for (SparseFeatureVector vec : vecs) {
            if (vec.getLabel().equals(targetLabel)) {
                vecList.add(vec);
            }
        }

        SparseFeatureVector[] targetVecs = new SparseFeatureVector[vecList.size()];
        for (int i = 0; i < targetVecs.length; i++) {
            targetVecs[i] = vecList.get(i);
        }
        return targetVecs;
    }

    public static List<SparseFeatureVector> getTargetVectorList(List<SparseFeatureVector> vecList, String targetLabel) {
        List<SparseFeatureVector> vectorList = new ArrayList<>();
        for (SparseFeatureVector vec : vecList) {
            if (vec.getLabel().equals(targetLabel)) {
                vectorList.add(vec);
            }
        }
        return vectorList;
    }

    public static void getEachIndexMinMax(SparseFeatureVector[] vecs, double[] minValues, double[] maxValues) {
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

    public static void getEachIndexAveSd(SparseFeatureVector[] vecs, double[] aveValues, double[] sdValues) {
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

    public static void doScaling(SparseFeatureVector[] vecs, SparseFeatureVector[] baseVecs, String type) {
        if (type.equals(NORMALIZATION)) {
            double[] minValues = new double[vecs[0].getSize()];
            double[] maxValues = new double[vecs[0].getSize()];
            getEachIndexMinMax(baseVecs, minValues, maxValues);
            for (int i = 0; i < vecs.length; i++) {
                double[] scaledValues = new double[vecs[i].getSize()];
                for (int j = 0; j < scaledValues.length; j++) {
                    scaledValues[j] = DataProcessUtil.normalize(vecs[i].getValue(j), minValues[j], maxValues[j]);
                }
                vecs[i].replaceAllValues(scaledValues);
            }
        } else if (type.equals(STANDARDIZATION)) {
            double[] aveValues = new double[vecs[0].getSize()];
            double[] sdValues = new double[vecs[0].getSize()];
            getEachIndexAveSd(baseVecs, aveValues, sdValues);
            for (int i = 0; i < vecs.length; i++) {
                double[] scaledValues = new double[vecs[i].getSize()];
                for (int j = 0; j < scaledValues.length; j++) {
                    scaledValues[j] = DataProcessUtil.standardize(vecs[i].getValue(j), aveValues[j], sdValues[j]);
                }
                vecs[i].replaceAllValues(scaledValues);
            }
        }
    }

    public static void doScaling(SparseFeatureVector[] vecs, String type) {
        doScaling(vecs, vecs, type);
    }

    public static void doScaling(SparseFeatureVector[] vecs) {
        doScaling(vecs, NORMALIZATION);
    }
}

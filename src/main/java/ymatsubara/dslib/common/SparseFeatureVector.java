package ymatsubara.dslib.common;

import java.util.List;

public class SparseFeatureVector {
    public static final String NONE_LABEL = "None";
    private String id, label;
    private double[] values;
    private int[] indices;

    public SparseFeatureVector(String id, String label, int size) {
        this.id = id;
        this.label = label;
        this.values = new double[size];
        this.indices = new int[size];
    }

    public SparseFeatureVector(String id, int size) {
        this(id, NONE_LABEL, size);
    }

    public void setLabel(String label) {
        this.label = label;
    }

    public void setValues(double[] values) {
        for (int i = 0; i < this.values.length; i++) {
            this.values[i] = values[i];
        }
    }

    public void setValues(List<Double> valueList) {
        for (int i = 0; i < this.values.length; i++) {
            this.values[i] = valueList.get(i);
        }
    }

    public void setValues(double[] values, int[] indices) {
        setValues(values);
        for (int i = 0; i < indices.length; i++) {
            this.indices[i] = indices[i];
        }
    }

    public void setValues(List<Double> valueList, List<Integer> indexList) {
        setValues(valueList);
        int size = valueList.size();
        for (int i = 0; i < size; i++) {
            this.indices[i] = indexList.get(i);
        }
    }

    public void setValues(double[] values, List<Integer> indexList) {
        setValues(values);
        int size = indexList.size();
        for (int i = 0; i < size; i++) {
            this.indices[i] = indexList.get(i);
        }
    }

    public void setValues(List<Double> valueList, int[] indices) {
        setValues(valueList);
        for (int i = 0; i < this.indices.length; i++) {
            this.indices[i] = indices[i];
        }
    }

    public String getId() {
        return this.id;
    }

    public String getLabel() {
        return this.label;
    }

    public double getValue(int index) {
        return this.values[index];
    }

    public double getIndex(int index) {
        return this.indices[index];
    }

    public double[] getAllValues() {
        return this.values;
    }

    public int[] getAllIndices() {
        return this.indices;
    }

    public int getSize() {
        return this.values.length;
    }

    public void clear() {
        int size = this.values.length;
        this.values = new double[size];
    }

    public void reset(int size) {
        this.values = new double[size];
    }

    public void replaceValue(double value, int index) {
        this.values[index] = value;
    }

    public void replaceAllValues(double[] values) {
        reset(values.length);
        for (int i = 0; i < this.values.length; i++) {
            this.values[i] = values[i];
        }
    }

    public void replaceAllValues(List<Double> valueList) {
        int size = valueList.size();
        reset(size);
        for (int i = 0; i < size; i++) {
            this.values[i] = valueList.get(i);
        }
    }

    public void replaceAllValues(double[] values, int[] indices) {
        replaceAllValues(values);
        this.indices = new int[indices.length];
        for (int i = 0; i < indices.length; i++) {
            this.indices[i] = indices[i];
        }
    }

    public void replaceAllValues(List<Double> valueList, List<Integer> indexList) {
        replaceAllValues(valueList);
        int size = indexList.size();
        this.indices = new int[size];
        for (int i = 0; i < this.indices.length; i++) {
            this.indices[i] = indexList.get(i);
        }
    }

    public void replaceAllValues(double[] values, List<Integer> indexList) {
        replaceAllValues(values);
        int size = indexList.size();
        this.indices = new int[size];
        for (int i = 0; i < this.indices.length; i++) {
            this.indices[i] = indexList.get(i);
        }
    }

    public void replaceAllValues(List<Double> valueList, int[] indices) {
        replaceAllValues(valueList);
        this.indices = new int[indices.length];
        for (int i = 0; i < indices.length; i++) {
            this.indices[i] = indices[i];
        }
    }
}

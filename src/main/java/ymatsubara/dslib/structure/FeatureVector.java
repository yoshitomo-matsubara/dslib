package ymatsubara.dslib.structure;

import java.util.List;

public class FeatureVector {
    public static final String NONE_LABEL = "None";
    public final String id;
    private String label;
    private double[] values;

    public FeatureVector(String id, String label, int size) {
        this.id = id;
        this.label = label;
        this.values = new double[size];
    }

    public FeatureVector(String id, String label, double[] values) {
        this.id = id;
        this.label = label;
        this.values = new double[values.length];
        for (int i = 0; i < this.values.length; i++) {
            this.values[i] = values[i];
        }
    }

    public FeatureVector(String id, int size) {
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

    public String getLabel() {
        return this.label;
    }

    public double getValue(int index) {
        return this.values[index];
    }

    public double[] getAllValues() {
        return this.values;
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

    public FeatureVector deepCopy() {
        return new FeatureVector(this.id, this.label, this.values);
    }
}

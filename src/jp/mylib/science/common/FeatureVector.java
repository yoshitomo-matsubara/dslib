package jp.mylib.science.common;

import java.util.List;

public class FeatureVector
{
    public static final String NONE_LABEL = "None";
    private String id, label;
    private int[] indices;
    private double[] values;

    public FeatureVector(String id, int size)
    {
        this.id = id;
        this.label = NONE_LABEL;
        this.indices = new int[size];
        this.values = new double[size];
    }

    public FeatureVector(String id, String label, int size)
    {
        this.id = id;
        this.label = label;
        this.indices = new int[size];
        this.values = new double[size];
    }

    public void setLabel(String label)
    {
        this.label = label;
    }

    public void setValues(double[] values)
    {
        for(int i=0;i<this.values.length;i++)
        {
            this.values[i] = values[i];
            this.indices[i] = i;
        }
    }

    public void setValues(List<Double> valueList)
    {
        for(int i=0;i<this.values.length;i++)
        {
            this.values[i] = valueList.get(i);
            this.indices[i] = i;
        }
    }

    public void setValues(double[] values, int[] indices)
    {
        for(int i=0;i<this.values.length;i++)
        {
            this.values[i] = values[i];
            this.indices[i] = indices[i];
        }
    }

    public void setValues(List<Double> values, List<Integer> indices)
    {
        for(int i=0;i<this.values.length;i++)
        {
            this.values[i] = values.get(i);
            this.indices[i] = indices.get(i);
        }
    }

    public void setValues(double[] values, List<Integer> indexList)
    {
        for(int i=0;i<this.values.length;i++)
        {
            this.values[i] = values[i];
            this.indices[i] = indexList.get(i);
        }
    }

    public void setValues(List<Double> valueList, int[] indices)
    {
        for(int i=0;i<this.values.length;i++)
        {
            this.values[i] = valueList.get(i);
            this.indices[i] = indices[i];
        }
    }

    public String getId()
    {
        return this.id;
    }

    public String getLabel()
    {
        return this.label;
    }

    public double getValue(int index)
    {
        return this.values[index];
    }

    public double[] getAllValues()
    {
        return this.values;
    }

    public int getIndex(int index)
    {
        return this.indices[index];
    }

    public int[] getAllIndices()
    {
        return this.indices;
    }

    public int getSize()
    {
        return this.values.length;
    }

    public void clear()
    {
        int size = this.indices.length;
        this.indices = new int[size];
        this.values = new double[size];
    }

    public void reset(int size)
    {
        this.indices = new int[size];
        this.values = new double[size];
    }

    public void replaceValue(double value, int index)
    {
        this.values[index] = value;
    }

    public void replaceAllValues(double[] values)
    {
        reset(values.length);
        for(int i=0;i<this.values.length;i++)
        {
            this.values[i] = values[i];
            this.indices[i] = i;
        }
    }

    public void replaceAllValues(List<Double> valueList)
    {
        reset(valueList.size());
        for(int i=0;i<this.values.length;i++)
        {
            this.values[i] = valueList.get(i);
            this.indices[i] = i;
        }
    }

    public void replaceAllValues(double[] values, int[] indices)
    {
        reset(values.length);
        for(int i=0;i<this.values.length;i++)
        {
            this.values[i] = values[i];
            this.indices[i] = indices[i];
        }
    }

    public void replaceAllValues(List<Double> valueList, List<Integer> indexList)
    {
        reset(valueList.size());
        for(int i=0;i<this.values.length;i++)
        {
            this.values[i] = valueList.get(i);
            this.indices[i] = indexList.get(i);
        }
    }

    public void replaceAllValues(double[] values, List<Integer> indexList)
    {
        reset(values.length);
        for(int i=0;i<this.values.length;i++)
        {
            this.values[i] = values[i];
            this.indices[i] = indexList.get(i);
        }
    }

    public void replaceAllValues(List<Double> valueList, int[] indices)
    {
        reset(valueList.size());
        for(int i=0;i<this.values.length;i++)
        {
            this.values[i] = valueList.get(i);
            this.indices[i] = indices[i];
        }
    }
}

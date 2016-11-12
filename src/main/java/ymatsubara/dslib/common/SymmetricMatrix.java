package ymatsubara.dslib.common;

import java.util.ArrayList;
import java.util.List;

public class SymmetricMatrix {
    private List<Double> list;
    private int size;

    public SymmetricMatrix(double[][] matrix) {
        this.list = new ArrayList<>();
        if (matrix.length == 0 || matrix.length != matrix[0].length) {
            return;
        }

        this.size = matrix.length;
        for (int i = 0; i < matrix.length; i++) {
            for (int j = i; j < matrix[0].length; j++) {
                this.list.add(matrix[i][j]);
            }
        }
    }

    public void set(int i, int j, double value) {
        if (i > j) {
            int tmp = i;
            i = j;
            j = tmp;
        }

        int sum = (i + 1) * i / 2;
        int index = i * this.size - sum + j;
        this.list.add(index, value);
    }

    public double get(int i, int j) {
        if (i > j) {
            int tmp = i;
            i = j;
            j = tmp;
        }

        int sum = (i + 1) * i / 2;
        int index = i * this.size - sum + j;
        return this.list.get(index);
    }

    public double[] getRow(int row) {
        double[] array = new double[this.size];
        for (int i = 0; i < size; i++) {
            array[i] = get(row, i);
        }
        return array;
    }

    public double[] getColumn(int column) {
        double[] array = new double[this.size];
        for (int i = 0; i < size; i++) {
            array[i] = get(i, column);
        }
        return array;
    }

    public int getRowSize() {
        return this.size;
    }

    public int getColumnSize() {
        return this.size;
    }

    public double[][] toMatrix() {
        double[][] matrix = new double[this.size][this.size];
        for (int i = 0; i < this.size; i++) {
            for (int j = 0; j < this.size; j++) {
                if (i > j) {
                    int tmp = i;
                    i = j;
                    j = tmp;
                }

                int sum = (i + 1) * i / 2;
                int index = i * this.size - sum + j;
                matrix[i][j] = this.list.get(index);
            }
        }
        return matrix;
    }
}

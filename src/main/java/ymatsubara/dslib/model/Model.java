package ymatsubara.dslib.model;

import ymatsubara.dslib.structure.FeatureVector;
import ymatsubara.dslib.structure.Result;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public abstract class Model {
    public static final String TYPE = "Abstract";

    public abstract void train(FeatureVector[] vecs);

    public void train(List<FeatureVector> vecList) {
        train(vecList.toArray(new FeatureVector[vecList.size()]));
    }

    public abstract Result predict(FeatureVector vec);

    public Result[] predict(FeatureVector[] vecs) {
        Result[] results = new Result[vecs.length];
        for (int i = 0; i < vecs.length; i++) {
            results[i] = predict(vecs[i]);
        }
        return results;
    }

    public List<Result> predict(List<FeatureVector> vecList) {
        List<Result> resultList = new ArrayList<>();
        for (FeatureVector vec : vecList) {
            resultList.add(predict(vec));
        }
        return resultList;
    }

    public abstract void reset();

    public List<Result> doLeaveOneOutCrossValidation(List<FeatureVector> vecList) {
        List<Result> resultList = new ArrayList<>();
        int size = vecList.size();
        for (int i = 0; i < size; i++) {
            FeatureVector testVec = vecList.get(0);
            vecList.remove(0);
            train(vecList);
            resultList.add(predict(testVec));
            vecList.add(testVec);
            reset();
        }
        return  resultList;
    }

    public Result[] doLeaveOneOutCrossValidation(FeatureVector[] vecs) {
        List<Result> resultList = doLeaveOneOutCrossValidation(Arrays.asList(vecs));
        Result[] results = new Result[resultList.size()];
        for (int i = 0 ; i < results.length ; i++) {
            results[i] = resultList.get(i);
        }
        return results;
    }

    public abstract void inputModel(String modelFilePath);

    public abstract void outputModel(String modelFilePath);
}

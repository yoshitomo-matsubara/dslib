package test;

import jp.mylib.science.classification.OneClassSvm;
import jp.mylib.science.common.FeatureVector;
import jp.mylib.science.common.FeatureVectorUtils;

public class SvmTester
{
    public static void main(String[] args)
    {
        OneClassSvm ocsvm = new OneClassSvm("test", 0.1);
        FeatureVector[] trainingVecs = FeatureVectorUtils.generateFeatureVectors("/home/yoshitomo/IdeaProjects/ScientificLibrary/in/train.csv");
        System.out.println("Training...");
//        ocsvm.doParamsGridSearch(trainingVecs, new double[]{0.1, 0.8, 0.1}, new double[][]{{0.1, 0.8, 0.1}}, true);
        ocsvm.train(trainingVecs);
        FeatureVector[] testVecs = FeatureVectorUtils.generateFeatureVectors("/home/yoshitomo/IdeaProjects/ScientificLibrary/in/test.csv");
        System.out.println("Testing...");
        int success = 0;
        for(FeatureVector vec : testVecs)
        {
            int result = ocsvm.predict(vec);
            if(result == Integer.parseInt(vec.getLabel()))
                success++;
        }

        System.out.println("Accuracy: " + ((double)success / (double)testVecs.length));
        ocsvm.outputModel("/home/yoshitomo/IdeaProjects/ScientificLibrary/in/model.csv");
    }
}

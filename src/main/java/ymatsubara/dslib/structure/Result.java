package ymatsubara.dslib.structure;

public class Result {
    public final String id, trueLabel, predictedLabel;
    public final double score;
    private String desc;

    public Result(FeatureVector vec, double score, String predictedLabel, String desc) {
        this.id = vec.id;
        this.trueLabel = vec.getLabel();
        this.score = score;
        this.predictedLabel = predictedLabel;
        this.desc = desc;
    }

    public Result(FeatureVector vec, double score, String predictedLabel) {
        this(vec, score, predictedLabel,null);
    }

    public Result(FeatureVector vec, double score) {
        this(vec, score, null,null);
    }

    public void setDesc(String desc) {
        this.desc = desc;
    }

    public String getDesc() {
        if (this.desc != null) {
            return this.desc;
        }
        return "";
    }
}
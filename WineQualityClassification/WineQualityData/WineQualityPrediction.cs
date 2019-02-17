using Microsoft.ML.Data;

namespace WineQualityClassification.WineQualityData
{
    public class WineQualityPrediction
    {
        [ColumnName("PredictedLabel")]
        public string PredictedLabel;
    }
}

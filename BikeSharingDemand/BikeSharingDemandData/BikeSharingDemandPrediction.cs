using Microsoft.ML.Data;

namespace BikeSharingDemand.BikeSharingDemandData
{
    public class BikeSharingDemandPrediction
    {
        [ColumnName("Score")]
        public float PredictedCount;
    }
}

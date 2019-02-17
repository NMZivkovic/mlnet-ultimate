using BikeSharingDemand.BikeSharingDemandData;
using Microsoft.Data.DataView;
using Microsoft.ML;
using Microsoft.ML.Core.Data;
using Microsoft.ML.Data;
using Microsoft.ML.Training;
using System.IO;
using System.Linq;

namespace BikeSharingDemand.ModelNamespace
{
    public sealed class Model
    {
        private readonly MLContext _mlContext;
        private PredictionEngine<BikeSharingDemandSample, BikeSharingDemandPrediction> _predictionEngine;
        private ITransformer _trainedModel;
        private TextLoader _textLoader;
        private ITrainerEstimator<ISingleFeaturePredictionTransformer<IPredictor>, IPredictor> _algorythim;
        
        public string Name { get; private set; }

        public Model(MLContext mlContext, ITrainerEstimator<ISingleFeaturePredictionTransformer<IPredictor>, IPredictor> algorythm)
        {
            _mlContext = mlContext;
            _algorythim = algorythm;
            _textLoader = _mlContext.Data.CreateTextLoader(new TextLoader.Arguments()
            {
                Separators = new[] { ',' },
                HasHeader = true,
                Column = new[]
                {
                    new TextLoader.Column("Season", DataKind.R4, 2),
                    new TextLoader.Column("Year", DataKind.R4, 3),
                    new TextLoader.Column("Month", DataKind.R4, 4),
                    new TextLoader.Column("Hour", DataKind.R4, 5),
                    new TextLoader.Column("Holiday", DataKind.Bool, 6),
                    new TextLoader.Column("Weekday", DataKind.R4, 7),
                    new TextLoader.Column("Weather", DataKind.R4, 8),
                    new TextLoader.Column("Temperature", DataKind.R4, 9),
                    new TextLoader.Column("NormalizedTemperature", DataKind.R4, 10),
                    new TextLoader.Column("Humidity", DataKind.R4, 11),
                    new TextLoader.Column("Windspeed", DataKind.R4, 12),
                    new TextLoader.Column("Count", DataKind.R4, 16),
                }
            });

            Name = algorythm.GetType().ToString().Split('.').Last();
        }

        public void BuildAndFit(string trainingDataViewLocation)
        {
            IDataView trainingDataView = _textLoader.Read(trainingDataViewLocation);

            var pipeline = _mlContext.Transforms.CopyColumns(inputColumnName: "Count", outputColumnName: "Label")
                .Append(_mlContext.Transforms.Categorical.OneHotEncoding("Season"))
                .Append(_mlContext.Transforms.Categorical.OneHotEncoding("Year"))
                .Append(_mlContext.Transforms.Categorical.OneHotEncoding("Month"))
                .Append(_mlContext.Transforms.Categorical.OneHotEncoding("Hour"))
                .Append(_mlContext.Transforms.Categorical.OneHotEncoding("Holiday"))
                .Append(_mlContext.Transforms.Categorical.OneHotEncoding("Weather"))
                .Append(_mlContext.Transforms.Concatenate("Features",
                                                "Season",
                                                "Year",
                                                "Month",
                                                "Hour",
                                                "Weekday",
                                                "Weather",
                                                "Temperature",
                                                "NormalizedTemperature",
                                                "Humidity",
                                                "Windspeed"))
                .AppendCacheCheckpoint(_mlContext)
                .Append(_algorythim);

            _trainedModel = pipeline.Fit(trainingDataView);
            _predictionEngine = _trainedModel.CreatePredictionEngine<BikeSharingDemandSample, BikeSharingDemandPrediction>(_mlContext);
        }
           
        public BikeSharingDemandPrediction Predict(BikeSharingDemandSample sample)
        {
            return _predictionEngine.Predict(sample);
        }

        public RegressionMetrics Evaluate(string testDataLocation)
        {
            var testData = _textLoader.Read(testDataLocation);
            var predictions = _trainedModel.Transform(testData);
            return _mlContext.Regression.Evaluate(predictions, "Label", "Score");
        }

        public void SaveModel()
        {
            using (var fs = new FileStream("./BikeSharingDemandsModel.zip", FileMode.Create, FileAccess.Write, FileShare.Write))
                _mlContext.Model.Save(_trainedModel, fs);
        }
    }
}

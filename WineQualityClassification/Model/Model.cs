using Microsoft.Data.DataView;
using Microsoft.ML;
using Microsoft.ML.Core.Data;
using Microsoft.ML.Data;
using Microsoft.ML.Training;
using Microsoft.ML.Transforms;
using System.IO;
using System.Linq;
using WineQualityClassification.WineQualityData;

namespace WineQualityClassification.ModelNamespace
{
    public class Model
    {
        private readonly MLContext _mlContext;
        private PredictionEngine<WineQualitySample, WineQualityPrediction> _predictionEngine;
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
                Separators = new[] { ';' },
                HasHeader = true,
                Column = new[]
                {
                    new TextLoader.Column("FixedAcidity", DataKind.R4, 0),
                    new TextLoader.Column("VolatileAcidity", DataKind.R4, 1),
                    new TextLoader.Column("CitricAcid", DataKind.R4, 2),
                    new TextLoader.Column("ResidualSugar", DataKind.R4, 3),
                    new TextLoader.Column("Chlorides", DataKind.R4, 4),
                    new TextLoader.Column("FreeSulfurDioxide", DataKind.R4, 5),
                    new TextLoader.Column("TotalSulfurDioxide", DataKind.R4, 6),
                    new TextLoader.Column("Density", DataKind.R4, 7),
                    new TextLoader.Column("Ph", DataKind.R4, 8),
                    new TextLoader.Column("Sulphates", DataKind.R4, 9),
                    new TextLoader.Column("Alcohol", DataKind.R4, 10),
                    new TextLoader.Column("Label", DataKind.Text, 11)
                }
            });

            Name = algorythm.GetType().ToString().Split('.').Last();
        }

        public void BuildAndFit(string trainingDataViewLocation)
        {
            IDataView trainingDataView = _textLoader.Read(trainingDataViewLocation);

            var pipeline = _mlContext.Transforms.ReplaceMissingValues(outputColumnName: "FixedAcidity", replacementKind: MissingValueReplacingTransformer.ColumnInfo.ReplacementMode.Mean)
                .Append(_mlContext.Transforms.Conversion.MapValueToKey("Label"))
                .Append(_mlContext.Transforms.Normalize())
                .Append(_mlContext.Transforms.Concatenate("Features",
                                                 "FixedAcidity",
                                                 "VolatileAcidity",
                                                 "CitricAcid",
                                                 "ResidualSugar",
                                                 "Chlorides",
                                                 "FreeSulfurDioxide",
                                                 "TotalSulfurDioxide",
                                                 "Density",
                                                 "Ph",
                                                 "Sulphates",
                                                 "Alcohol"))
                .AppendCacheCheckpoint(_mlContext)
                .Append(_algorythim)
                .Append(_mlContext.Transforms.Conversion.MapKeyToValue("PredictedLabel"));

            _trainedModel = pipeline.Fit(trainingDataView);
            _predictionEngine = _trainedModel.CreatePredictionEngine<WineQualitySample, WineQualityPrediction>(_mlContext);
        }

        public WineQualityPrediction Predict(WineQualitySample sample)
        {
            return _predictionEngine.Predict(sample);
        }

        public MultiClassClassifierMetrics Evaluate(string testDataLocation)
        {
            var testData = _textLoader.Read(testDataLocation);
            var predictions = _trainedModel.Transform(testData);
            return _mlContext.MulticlassClassification.Evaluate(predictions);
        }

        public void SaveModel()
        {
            using (var fs = new FileStream("./WineQualityModel.zip", FileMode.Create, FileAccess.Write, FileShare.Write))
                _mlContext.Model.Save(_trainedModel, fs);
        }
    }
}

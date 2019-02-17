using Microsoft.ML;
using Microsoft.ML.Training;
using System;
using System.Collections.Generic;
using System.Linq;
using WineQualityClassification.Helpers;
using WineQualityClassification.ModelNamespace;

namespace WineQualityClassification
{
    class Program
    {
        private static MLContext _mlContext = new MLContext();
        private static Dictionary<Model, double> _stats = new Dictionary<Model, double>();
        private static string _trainingDataLocation = @"Data/winequality_white_train.csv";
        private static string _testDataLocation = @"Data/winequality_white_test.csv";

        static void Main(string[] args)
        {

            var classifiers = new List<ITrainerEstimator<ISingleFeaturePredictionTransformer<IPredictor>, IPredictor>>()
            {
                _mlContext.MulticlassClassification.Trainers.LogisticRegression(labelColumn: "Label", featureColumn: "Features"),
                _mlContext.MulticlassClassification.Trainers.NaiveBayes(labelColumn: "Label", featureColumn: "Features"),
                _mlContext.MulticlassClassification.Trainers.StochasticDualCoordinateAscent(labelColumn: "Label", featureColumn: "Features")
            };

            classifiers.ForEach(RunAlgorythm);

            var bestModel = _stats.Where(x => x.Value == _stats.Max(y => y.Value)).Single().Key;
            VisualizeTenPredictionsForTheModel(bestModel);
            bestModel.SaveModel();

            Console.ReadLine();
        }

        private static void RunAlgorythm(ITrainerEstimator<ISingleFeaturePredictionTransformer<IPredictor>, IPredictor> algorythm)
        {
            var model = new Model(_mlContext, algorythm);
            model.BuildAndFit(_trainingDataLocation);
            PrintAndStoreMetrics(model);
        }

        private static void PrintAndStoreMetrics(Model model)
        {
            var metrics = model.Evaluate(_testDataLocation);

            Console.WriteLine($"*************************************************");
            Console.WriteLine($"*       Metrics for {model.Name}          ");
            Console.WriteLine($"*------------------------------------------------");
            Console.WriteLine($"*       Accuracy Macro: {metrics.AccuracyMacro:0.##}");
            Console.WriteLine($"*       Accuracy Micro: {metrics.AccuracyMicro:0.##}");
            Console.WriteLine($"*       Log Loss: {metrics.LogLoss:#.##}");
            Console.WriteLine($"*       Log Loss Reduction: {metrics.LogLossReduction:#.##}");
            Console.WriteLine($"*************************************************");

            _stats.Add(model, metrics.AccuracyMacro);
        }

        private static void VisualizeTenPredictionsForTheModel(Model model)
        {
            Console.WriteLine($"*************************************************");
            Console.WriteLine($"* BEST MODEL IS: {model.Name}!");
            Console.WriteLine($"* Here are its predictions: ");
            var testData = new WineQualityCsvReader().GetDataFromCsv(_testDataLocation).ToList();

            for (int i = 0; i < 10; i++)
            {
                var prediction = model.Predict(testData[i]);
                Console.WriteLine($"*------------------------------------------------");
                Console.WriteLine($"* Predicted : {prediction.PredictedLabel}");
                Console.WriteLine($"* Actual:    {testData[i].Label}");
                Console.WriteLine($"*------------------------------------------------");
            }
            Console.WriteLine($"*************************************************");
        }
    }
}

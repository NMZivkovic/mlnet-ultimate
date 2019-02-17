using BikeSharingDemand.BikeSharingDemandData;
using BikeSharingDemand.Helpers;
using BikeSharingDemand.ModelNamespace;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Training;
using System;
using System.Collections.Generic;
using System.Linq;

namespace BikeSharingDemand
{
    class Program
    {
        private static MLContext _mlContext = new MLContext();
        private static Dictionary<Model, double> _stats = new Dictionary<Model, double>();
        private static string _trainingDataLocation = @"Data/hour_train.csv";
        private static string _testDataLocation = @"Data/hour_test.csv";

        static void Main(string[] args)
        {

            var regressors = new List<ITrainerEstimator<ISingleFeaturePredictionTransformer<IPredictor>, IPredictor>>()
            {
                _mlContext.Regression.Trainers.FastForest(labelColumn: "Count", featureColumn: "Features"),
                _mlContext.Regression.Trainers.FastTree(labelColumn: "Count", featureColumn: "Features"),
                _mlContext.Regression.Trainers.FastTreeTweedie(labelColumn: "Count", featureColumn: "Features"),
                _mlContext.Regression.Trainers.GeneralizedAdditiveModels(labelColumn: "Count", featureColumn: "Features"),
                _mlContext.Regression.Trainers.OnlineGradientDescent(labelColumn: "Count", featureColumn: "Features"),
                _mlContext.Regression.Trainers.PoissonRegression(labelColumn: "Count", featureColumn: "Features"),
                _mlContext.Regression.Trainers.StochasticDualCoordinateAscent(labelColumn: "Count", featureColumn: "Features")
            };

            regressors.ForEach(RunAlgorythm);

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
            Console.WriteLine($"*       R2 Score: {metrics.RSquared:0.##}");
            Console.WriteLine($"*       Absolute loss: {metrics.L1:#.##}");
            Console.WriteLine($"*       Squared loss: {metrics.L2:#.##}");
            Console.WriteLine($"*       RMS loss: {metrics.Rms:#.##}");
            Console.WriteLine($"*************************************************");

            _stats.Add(model, metrics.RSquared);
        }

        private static void VisualizeTenPredictionsForTheModel(Model model)
        {
            Console.WriteLine($"*************************************************");
            Console.WriteLine($"* BEST MODEL IS: {model.Name}!");
            Console.WriteLine($"* Here are its predictions: ");
            var testData = new BikeSharingDemandsCsvReader().GetDataFromCsv(_testDataLocation).ToList();
            for (int i = 0; i < 10; i++)
            {
                var prediction = model.Predict(testData[i]);
                Console.WriteLine($"*------------------------------------------------");
                Console.WriteLine($"* Predicted : {prediction.PredictedCount}");
                Console.WriteLine($"* Actual:    {testData[i].Count}");
                Console.WriteLine($"*------------------------------------------------");
            }
            Console.WriteLine($"*************************************************");
        }
    }
}

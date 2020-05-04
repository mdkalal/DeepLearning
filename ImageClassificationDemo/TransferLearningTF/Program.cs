
// This app is based off tutorial https://docs.microsoft.com/en-us/dotnet/machine-learning/tutorials/image-classification
// also the google one
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using Microsoft.ML;
using Microsoft.ML.Data;

namespace TransferLearningTF
{
    class Program
    {

        static readonly string _assetsPath = @"C:\dev\DLassets";
        static readonly string _modelPath = Path.Combine(_assetsPath, "imagemodel.zip");
        static readonly string _imagesFolder = Path.Combine(_assetsPath, "images");
        static readonly string _trainTagsTsv = Path.Combine(_imagesFolder, "tags.tsv");
        static readonly string _testTagsTsv = Path.Combine(_imagesFolder, "test-tags.tsv");
        static readonly string _inceptionTensorFlowModel = Path.Combine(_assetsPath, "inception", "tensorflow_inception_graph.pb");
        static void Main(string[] args)
        {

            // Create new ML.Net environment
            MLContext mlContext = new MLContext();

            if (args[0] != "test")
            {
                // Create the model
                ITransformer model = GenerateModel(mlContext);

            }

            // test harness for single images
            
            // load model
            ITransformer mlModel = mlContext.Model.Load(GetAbsolutePath(_modelPath), out DataViewSchema inputSchema);

            ClassifySingleImage(mlContext, mlModel, Path.Combine(_imagesFolder, "chewy_tst.jpg"));


        }

        public static string GetAbsolutePath(string relativePath)
        {
            FileInfo _dataRoot = new FileInfo(typeof(Program).Assembly.Location);
            string assemblyFolderPath = _dataRoot.Directory.FullName;

            string fullPath = Path.Combine(assemblyFolderPath, relativePath);

            return fullPath;
        }


        private static void SaveModel(MLContext mlContext, ITransformer mlModel, string modelRelativePath, DataViewSchema modelInputSchema)
        {
            // Save/persist the trained model to a .ZIP file
            // Console.WriteLine("=============== Saving the model  ===============");
            mlContext.Model.Save(mlModel, modelInputSchema, GetAbsolutePath(modelRelativePath));
            Console.WriteLine("The model is saved to {0}", GetAbsolutePath(modelRelativePath));
        }

      
        // Parameters required by the Tensorflow Inception model
        private struct InceptionSettings
        {
            public const int ImageHeight = 224;
            public const int ImageWidth = 224;
            public const float Mean = 117;
            public const float Scale = 1;
            public const bool ChannelsLast = true;
        }

        public static void ClassifySingleImage(MLContext mlContext, ITransformer model, string pImagePath)
        {
            ImageData imageData = new ImageData();
            imageData.ImagePath = pImagePath;


            var predictor = mlContext.Model.CreatePredictionEngine<ImageData, ImagePrediction>(model);
            var prediction = predictor.Predict(imageData);

            Console.WriteLine($"Image: {Path.GetFileName(imageData.ImagePath)} predicted as: {prediction.PredictedLabelValue} with score: {prediction.Score.Max()} ");


        }


        public static ITransformer GenerateModel(MLContext mlContext)
        {
            IEstimator<ITransformer> pipeline = mlContext.Transforms.LoadImages(outputColumnName: "input", imageFolder: _imagesFolder, inputColumnName: nameof(ImageData.ImagePath))

                // Transform the images into the model's expected format
                .Append(mlContext.Transforms.ResizeImages(outputColumnName: "input", imageWidth: InceptionSettings.ImageWidth, imageHeight: InceptionSettings.ImageHeight, inputColumnName: "input"))
                .Append(mlContext.Transforms.ExtractPixels(outputColumnName: "input", interleavePixelColors: InceptionSettings.ChannelsLast, offsetImage: InceptionSettings.Mean))

                // Load TensorFlow inception model (downloaded from TensorFlow)
                .Append(mlContext.Model.LoadTensorFlowModel(_inceptionTensorFlowModel).
                    ScoreTensorFlowModel(outputColumnNames: new[] { "softmax2_pre_activation" }, inputColumnNames: new[] { "input" }, addBatchDimensionInput: true))

                //Specify the image classification info
                .Append(mlContext.Transforms.Conversion.MapValueToKey(outputColumnName: "LabelKey", inputColumnName: "Label"))

                //Select training algorithm (multiclass traininer)
                .Append(mlContext.MulticlassClassification.Trainers.LbfgsMaximumEntropy(labelColumnName: "LabelKey", featureColumnName: "softmax2_pre_activation"))

                //Specify the classification prediction info
                .Append(mlContext.Transforms.Conversion.MapKeyToValue("PredictedLabelValue", "PredictedLabel"))

                //Use cached data
                // From MS docs- Cache checkpoints should be removed if disk thrashing or OutOfMemory exceptions are seen,
                // which can occur on when the featured dataset immediately prior to the checkpoint is larger than
                // available RAM.
                .AppendCacheCheckpoint(mlContext);

            // load training data - text file contains labels
            IDataView trainingData = mlContext.Data.LoadFromTextFile<ImageData>(path: _trainTagsTsv, hasHeader: false);

            // Train the model
            // Get comfy, this takes a while!
            ITransformer model = pipeline.Fit(trainingData);


            // Now, do a test and see how well the model performs
            // Load up some test data - text file contains labels
            IDataView testData = mlContext.Data.LoadFromTextFile<ImageData>(path: _testTagsTsv, hasHeader: false);

            // Classify test images
            IDataView predictions = model.Transform(testData);

            // Create an IEnumerable for the predictions for displaying results
            IEnumerable<ImagePrediction> imagePredictionData = mlContext.Data.CreateEnumerable<ImagePrediction>(predictions, true);

            // Show results
            DisplayResults(imagePredictionData);
            
            //evaluate model
            MulticlassClassificationMetrics metrics = mlContext.MulticlassClassification.Evaluate(predictions, labelColumnName: "LabelKey", predictedLabelColumnName: "PredictedLabel");

            // display model accuracy metrics
            // https://docs.microsoft.com/en-us/dotnet/machine-learning/resources/metrics
            Console.WriteLine($"LogLoss is: {metrics.LogLoss}");
            Console.WriteLine($"PerClassLogLoss is: {String.Join(" , ", metrics.PerClassLogLoss.Select(c => c.ToString()))}");

            // Save model as a .zip file -- make sure you have permissions to write to _modelPath
            SaveModel(mlContext, model, _modelPath, trainingData.Schema);
            
            return model;
        }

        private static void DisplayResults(IEnumerable<ImagePrediction> imagePredictionData)
        {
            foreach (ImagePrediction prediction in imagePredictionData)
            {
                Console.WriteLine($"Image: {Path.GetFileName(prediction.ImagePath)} predicted as: {prediction.PredictedLabelValue} with score: {prediction.Score.Max()} ");
            }
        }

        public static IEnumerable<ImageData> ReadFromTsv(string file, string folder)
        {
            return File.ReadAllLines(file)
            .Select(line => line.Split('\t'))
            .Select(line => new ImageData()
            {
                ImagePath = Path.Combine(folder, line[0])
            });
        }
    }

    public class ImageData
    {
        [LoadColumn(0)]
        public string ImagePath;

        [LoadColumn(1)]
        public string Label;
    }

    public class ImagePrediction : ImageData
    {
        public float[] Score;

        public string PredictedLabelValue;
    }

}

using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Threading.Tasks;
using Microsoft.AspNetCore.Mvc;
using Microsoft.Extensions.Logging;
using ImageClassificationDemo.Models;
using System.IO;
using Microsoft.ML;
using Microsoft.ML.Data;

namespace ImageClassificationDemo.Controllers
{
    public class HomeController : Controller
    {
        private readonly ILogger<HomeController> _logger;
        private Random rnd = new Random();

        static readonly string _assetsPath = @"C:\dev\DLassets";
        static readonly string _modelPath = Path.Combine(_assetsPath, "imagemodel.zip");  //@"C:\talks\DeepLearning\assets\imagemodel.zip";
        static readonly string _predictionFolder = @"C:\Dev\ImageClassificationDemo\ImageClassificationDemo\wwwroot\images\predict\";
        static readonly string _imagesFolder = "/images/predict/";
        

        public string DoClassify()
        {
            string retVal;

            // creates a random number between one and ten
            int imageNumber = rnd.Next(1, 11); 
            
            // use random number to select a random image
            string imageName = "image" + imageNumber.ToString() + ".jpg";

            // Create new ML.Net environment
            MLContext mlContext = new MLContext();

            // Load the model (saved as a .zip file)
            ITransformer mlModel = mlContext.Model.Load(_modelPath, out DataViewSchema inputSchema);
            
            // Create and populate image class
            ImageData imageData = new ImageData();
            imageData.ImagePath = _predictionFolder + imageName;

            // Create prediction engine based on model
            var predictor = mlContext.Model.CreatePredictionEngine<ImageData, ImagePrediction>(mlModel);
            
            // Perform prediction based on prediction engine
            var prediction = predictor.Predict(imageData);

            retVal = _imagesFolder + imageName + "," + prediction.PredictedLabelValue;

            return retVal;
        }

        // Class to pass to image classification
        public class ImageData
        {
            [LoadColumn(0)]
            public string ImagePath;

            [LoadColumn(1)]
            public string Label;
        }

        // Class to return classification
        public class ImagePrediction : ImageData
        {
            public float[] Score;

            public string PredictedLabelValue;
        }

        public HomeController(ILogger<HomeController> logger)
        {
            _logger = logger;
        }

        public IActionResult Index()
        {
            return View();
        }

        public IActionResult Privacy()
        {
            return View();
        }

        [ResponseCache(Duration = 0, Location = ResponseCacheLocation.None, NoStore = true)]
        public IActionResult Error()
        {
            return View(new ErrorViewModel { RequestId = Activity.Current?.Id ?? HttpContext.TraceIdentifier });
        }
    }
}

Deep Learning talk resources

Project descriptions:

ImageClassificationDemo - demo of image classification using ML.Net
In order to run this app- You need .net Core 3.1 (download available from Microsoft)
Following dependencies available from Nuget- 
	Microsoft.ML
	Microsoft.ML.ImageAnalytics
	Microsoft.ML.Tensorflow
	SciSharp.TensorFlow.Redist

The app is coded to look for files in certian locations.  You can change this by changing the locaton in 
HomeController:
	static readonly string _assetsPath = @"C:\dev\DLassets";
	static readonly string _predictionFolder = @"C:\dev\ImageClassificationDemo\ImageClassificationDemo\wwwroot\images\predict\";


TransferLearningTF - Demo of creating/training a model for image classification using ML.Net and Tensoflow Inception
In order to run this app- You need .net Core 3.1 (download available from Microsoft)

Following dependencies available from Nuget- 
	Microsoft.ML
	Microsoft.ML.ImageAnalytics
	Microsoft.ML.Tensorflow
	SciSharp.TensorFlow.Redist

The app is coded to look for files and dependencies, locations are outlined in Program.cs, or else change the locaton in 
	static readonly string ASSETS_PATH = @"C:\dev\DLassets";

DLAssets
	Contains the data required for the applications

DeepLearnning.pdf - Slide deck for talk
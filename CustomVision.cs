namespace CustomVisionCLI
{
    using PowerArgs;
    using System;
    using System.Diagnostics;
    using System.IO;
    using TensorFlow;
    using OpenCvSharp;
    using System.Threading.Tasks;
    using System.Threading;

    [ArgExceptionBehavior(ArgExceptionPolicy.StandardExceptionHandling)]
    [TabCompletion(HistoryToSave = 10)]
    [ArgExample("CustomVision-TensorFlow.exe -m Assets\\model.pb -l Assets\\labels.txt -t Assets\\test.jpg", "using arguments", Title = "Classify image using relative paths")]
    [ArgExample("CustomVision-TensorFlow.exe -m c:\\tensorflow\\model.pb -l c:\\tensorflow\\labels.txt -t c:\\tensorflow\\test.jpg", "using arguments", Title = "Classify image using full filepath")]
    public class CustomVision
    {
        [ArgRequired(PromptIfMissing = true)]
        [ArgDescription("CustomVision.ai TensorFlow exported model")]
        [ArgShortcut("-m")]
        public string TensorFlowModelFilePath { get; set; }

        [ArgRequired(PromptIfMissing = true)]
        [ArgDescription("CustomVision.ai TensorFlow exported labels")]
        [ArgShortcut("-l")]
        public string TensorFlowLabelsFilePath { get; set; }

        [ArgRequired(PromptIfMissing = true)]
        [ArgDescription("Image to classify (jpg)")]
        [ArgShortcut("-t")]
        public string TestImageFilePath { get; set; }
     
        [HelpHook]
        public bool Help { get; set; }

        public void Main()
        {
            Stopwatch stopwatch = new Stopwatch();
            stopwatch.Start();

            var graph = new TFGraph();
            var model = File.ReadAllBytes(TensorFlowModelFilePath);
            var labels = File.ReadAllLines(TensorFlowLabelsFilePath);
            graph.Import(model);

            var bestIdx = 0;
            float best = 0;

            Mat image = new Mat();
            VideoCapture capture = new VideoCapture(0);
            capture.Grab();
            int count = 0, check = 0;
            string res = "hello";
            while (capture.IsOpened())
            {
                capture.Read(image);
                Mat temp = new Mat();
                Cv2.CvtColor(image, temp, ColorConversionCodes.RGB2GRAY);

                count++;
                Point p1 = new Point(100, 100);
                Rect rect = new Rect(p1, new Size(200, 200));
                Scalar c1 = new Scalar(0, 0, 100);

                Console.WriteLine(count);
                Cv2.Rectangle(image, rect, c1);
                
                Mat tr = new Mat(temp, rect);
                
                

                if(count==100)
                {
                    tr.SaveImage("Assets\\test.jpg");
                    new Thread(() => {

                        using (var session = new TFSession(graph))
                        {
                            var tensor = ImageUtil.CreateTensorFromImageFile(TestImageFilePath);
                            var runner = session.GetRunner();
                            runner.AddInput(graph["Placeholder"][0], tensor).Fetch(graph["loss"][0]);
                            var output = runner.Run();
                            var result = output[0];

                            var probabilities = ((float[][])result.GetValue(jagged: true))[0];
                            for (int i = 0; i < probabilities.Length; i++)
                            {
                                if (probabilities[i] > best)
                                {
                                    bestIdx = i;
                                    best = probabilities[i];
                                }
                            }
                        }

                        // fin
                        stopwatch.Stop();
                        Console.WriteLine($"{TestImageFilePath} = {labels[bestIdx]} ({best * 100.0}%)");
                        Console.WriteLine($"Total time: {stopwatch.Elapsed}");
                        res = labels[bestIdx];
                        Console.ReadKey();
                        
                        Console.WriteLine(labels[bestIdx]);
                    }).Start();
                    count = 0;
                }

                   MatOfByte3 mat3 = new MatOfByte3(image); // cv::Mat_<cv::Vec3b>
                    var indexer = mat3.GetIndexer();
                    for (int x = 100; x < 250; x++)
                    {
                        for (int y = 40; y < 90; y++)
                        {
                            Vec3b color = new Vec3b(161,101,0);

                            indexer[y,x] = color;
                        }
                    }
                    Cv2.PutText(image, res, new Point(150, 80), HersheyFonts.HersheyPlain, 2, Scalar.White);
               // Cv2.PutText(image, res, new Point(30, 30), HersheyFonts.HersheySimplex, 3, new Scalar(0, 0, 100));
                Cv2.ImShow("Frame1", image);
                Cv2.ImShow("Frame2", tr);
                Cv2.WaitKey(1);
            }
        

        }
    }
}

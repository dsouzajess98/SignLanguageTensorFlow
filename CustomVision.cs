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
            int count = 0;
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
                    count = 50;
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
                
                Cv2.ImShow("Original", image);
                Cv2.ImShow("Hand", tr);
                Mat sign = new Mat();
                sign=Cv2.ImRead("Assets\\India_SL.jpg");

                //Identifying sign in chart
                if (res == "A")
                    Cv2.Rectangle(sign, new Rect(new Point(0, 0), new Size(75, 75)), new Scalar(0, 0, 100));
                else if(res=="B")
                    Cv2.Rectangle(sign, new Rect(new Point(75,0), new Size(75, 75)), new Scalar(0, 0, 100));
                else if (res == "C")
                    Cv2.Rectangle(sign, new Rect(new Point(75*2, 0), new Size(75, 75)), new Scalar(0, 0, 100));
                else if (res == "D")
                    Cv2.Rectangle(sign, new Rect(new Point(75*3, 0), new Size(75, 75)), new Scalar(0, 0, 100));
                else if (res == "E")
                    Cv2.Rectangle(sign, new Rect(new Point(75*4, 0), new Size(75, 75)), new Scalar(0, 0, 100));
                else if (res == "F")
                    Cv2.Rectangle(sign, new Rect(new Point(75*5, 0), new Size(75, 75)), new Scalar(0, 0, 100));
                else if (res == "G")
                    Cv2.Rectangle(sign, new Rect(new Point(75*6, 0), new Size(75, 75)), new Scalar(0, 0, 100));
                else if (res == "H")
                    Cv2.Rectangle(sign, new Rect(new Point(0, 75), new Size(75, 75)), new Scalar(0, 0, 100));
                else if (res == "I")
                    Cv2.Rectangle(sign, new Rect(new Point(75, 75), new Size(75, 75)), new Scalar(0, 0, 100));
                else if (res == "J")
                    Cv2.Rectangle(sign, new Rect(new Point(75 * 2, 75), new Size(75, 75)), new Scalar(0, 0, 100));
                else if (res == "K")
                    Cv2.Rectangle(sign, new Rect(new Point(75 * 3, 75), new Size(75, 75)), new Scalar(0, 0, 100));
                else if (res == "L")
                    Cv2.Rectangle(sign, new Rect(new Point(75 * 4, 75), new Size(75, 75)), new Scalar(0, 0, 100));
                else if (res == "M")
                    Cv2.Rectangle(sign, new Rect(new Point(75 * 5, 75), new Size(75, 75)), new Scalar(0, 0, 100));
                else if (res == "N")
                    Cv2.Rectangle(sign, new Rect(new Point(75 * 6, 75), new Size(75, 75)), new Scalar(0, 0, 100));
                else if (res == "O")
                    Cv2.Rectangle(sign, new Rect(new Point(0, 75*2), new Size(75, 75)), new Scalar(0, 0, 100));
                else if (res == "P")
                    Cv2.Rectangle(sign, new Rect(new Point(75, 75 * 2), new Size(75, 75)), new Scalar(0, 0, 100));
                else if (res == "Q")
                    Cv2.Rectangle(sign, new Rect(new Point(75 * 2, 75 * 2), new Size(75, 75)), new Scalar(0, 0, 100));
                else if (res == "R")
                    Cv2.Rectangle(sign, new Rect(new Point(75 * 3, 75 * 2), new Size(75, 75)), new Scalar(0, 0, 100));
                else if (res == "S")
                    Cv2.Rectangle(sign, new Rect(new Point(75 * 4, 75 * 2), new Size(75, 75)), new Scalar(0, 0, 100));
                else if (res == "T")
                    Cv2.Rectangle(sign, new Rect(new Point(75 * 5, 75 * 2), new Size(75, 75)), new Scalar(0, 0, 100));
                else if (res == "U")
                    Cv2.Rectangle(sign, new Rect(new Point(75 * 6, 75 * 2), new Size(75, 75)), new Scalar(0, 0, 100));
                else if (res == "V")
                    Cv2.Rectangle(sign, new Rect(new Point(75, 75 * 3), new Size(75, 75)), new Scalar(0, 0, 100));
                else if (res == "W")
                    Cv2.Rectangle(sign, new Rect(new Point(75 * 2, 75 * 3), new Size(75, 75)), new Scalar(0, 0, 100));
                else if (res == "X")
                    Cv2.Rectangle(sign, new Rect(new Point(75 * 3, 75 * 3), new Size(75, 75)), new Scalar(0, 0, 100));
                else if (res == "Y")
                    Cv2.Rectangle(sign, new Rect(new Point(75 * 4, 75 * 3), new Size(75, 75)), new Scalar(0, 0, 100));
                else if (res == "Z")
                    Cv2.Rectangle(sign, new Rect(new Point(75 * 5, 75 * 3), new Size(75, 75)), new Scalar(0, 0, 100));


                Cv2.ImShow("Signs", sign);
                Cv2.WaitKey(1);
            }
        

        }
    }
}

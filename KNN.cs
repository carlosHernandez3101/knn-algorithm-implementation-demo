using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace knn_algorithm_implementation_demo
{
    internal class KNN
    {
        private List<Point> trainingData;
        private List<Point> testData;
        private int K;

        public KNN(int K)
        {
            this.K = K;
            trainingData = new List<Point>();
            testData = new List<Point>();
        }

        public void LoadAndSplitDatasetCSV(string filePath, double trainPercentage)
        {
            List<Point> dataset = new List<Point>();

            using (var reader = new StreamReader(filePath))
            {
                bool isFirstLine = true;  // Indicador para la primera línea
                while (!reader.EndOfStream)
                {
                    var line = reader.ReadLine();

                    // Ignorar la primera fila (encabezado)
                    if (isFirstLine)
                    {
                        isFirstLine = false;
                        continue;  // Saltar esta iteración para no procesar la primera línea
                    }

                    var values = line.Split(',');

                    // Los últimos valores son las etiquetas, los primeros son las características
                    List<double> features = values.Take(values.Length - 1).Select(double.Parse).ToList();
                    string label = values.Last();  // Último valor es la etiqueta

                    dataset.Add(new Point(features, label));
                }
            }

            // Mezclar aleatoriamente el dataset
            Random rng = new Random();
            dataset = dataset.OrderBy(a => rng.Next()).ToList();

            // Dividir el dataset en entrenamiento y prueba
            int trainSize = (int)(trainPercentage * dataset.Count);
            trainingData = dataset.Take(trainSize).ToList();
            testData = dataset.Skip(trainSize).ToList();
        }

        // Método para normalizar los datos numéricos
        public void NormalizeData()
        {
            // Unimos los datos de entrenamiento y prueba para normalizar todos los datos a la vez
            var allData = trainingData.Concat(testData).ToList();

            // Número de características
            int numFeatures = allData[0].features.Count;

            // Encontrar los valores mínimos y máximos para cada característica
            double[] minValues = new double[numFeatures];
            double[] maxValues = new double[numFeatures];

            // Inicializamos minValues y maxValues con los primeros valores de los datos
            for (int i = 0; i < numFeatures; i++)
            {
                minValues[i] = allData.Min(p => p.features[i]);
                maxValues[i] = allData.Max(p => p.features[i]);
            }

            // Aplicar la normalización Min-Max a cada conjunto de características
            foreach (var point in allData)
            {
                for (int i = 0; i < point.features.Count; i++)
                {
                    double minValue = minValues[i];
                    double maxValue = maxValues[i];
                    point.features[i] = (point.features[i] - minValue) / (maxValue - minValue);
                }
            }

            // Separar nuevamente en conjuntos de entrenamiento y prueba
            trainingData = allData.Take(trainingData.Count).ToList();
            testData = allData.Skip(trainingData.Count).ToList();
        }

        public string Predict(List<double> newFeatures)
        {
            Point newPoint = new Point(newFeatures, "");
            var distances = trainingData
                .Select(p => new { Point = p, Distance = p.DistanceTo(newPoint) })
                .OrderBy(p => p.Distance)
                .Take(K);

            var predictedLabel = distances
                .GroupBy(p => p.Point.label)
                .OrderByDescending(g => g.Count())
                .First().Key;

            return predictedLabel;
        }

        public (double accuracy, Dictionary<string, Dictionary<string, int>> confusionMatrix) EvaluateTestData()
        {
            int correctPredictions = 0;
            Dictionary<string, Dictionary<string, int>> confusionMatrix = new Dictionary<string, Dictionary<string, int>>();

            foreach (var point in testData)
            {
                // Predicción para cada punto en testData
                string predictedLabel = Predict(point.features);
                string actualLabel = point.label;

                // Verificar si la predicción es correcta
                if (predictedLabel == actualLabel)
                {
                    correctPredictions++;
                }

                // Actualizar la matriz de confusión
                if (!confusionMatrix.ContainsKey(actualLabel))
                {
                    confusionMatrix[actualLabel] = new Dictionary<string, int>();
                }
                if (!confusionMatrix[actualLabel].ContainsKey(predictedLabel))
                {
                    confusionMatrix[actualLabel][predictedLabel] = 0;
                }
                confusionMatrix[actualLabel][predictedLabel]++;
            }

            // Calcular el accuracy
            double accuracy = (double)correctPredictions / testData.Count;
            return (accuracy, confusionMatrix);
        }
    }
}

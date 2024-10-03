using knn_algorithm_implementation_demo;

public class Program
{
    public static void Main(string[] args)
    {
        // Ruta del archivo CSV
        string filePath = "C:\\drugs-norm.csv";  // Asegúrate de colocar el archivo en la ubicación correcta

        // Creamos el clasificador KNN con K = 3
        KNN knn = new KNN(3);

        // Cargar y dividir el dataset (70% entrenamiento, 30% prueba)
        knn.LoadAndSplitDatasetCSV(filePath, 0.7);

        //Normalizamos los datos del dataset
        //knn.NormalizeData();

        // Evaluar el modelo sobre el testData
        var (accuracy, confusionMatrix) = knn.EvaluateTestData();

        // Mostrar el accuracy
        Console.WriteLine($"Precisión del modelo: {accuracy * 100}%");

        // Mostrar la matriz de confusión
        Console.WriteLine("Matriz de confusión:");
        foreach (var actualLabel in confusionMatrix.Keys)
        {
            foreach (var predictedLabel in confusionMatrix[actualLabel].Keys)
            {
                Console.WriteLine($"Real: {actualLabel}, Predicho: {predictedLabel}, Cantidad: {confusionMatrix[actualLabel][predictedLabel]}");
            }
        }
    }
}
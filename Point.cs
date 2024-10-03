using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace knn_algorithm_implementation_demo
{
    internal class Point
    {
        public List<double> features {get; set;}
        public string label { get; set;}

        public Point(List<double> features, string label)
        {
            this.features = features;
            this.label = label;
        }

        public double DistanceTo(Point pointPredict)
        {
            double sum = 0.0;
            for (int i = 0; i < features.Count; i++)
            {
                sum += Math.Pow(features[i] - pointPredict.features[i], 2);                
            }
            return Math.Sqrt(sum);
        }
    }
}

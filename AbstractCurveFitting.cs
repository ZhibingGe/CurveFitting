using System;
using System.Linq;

namespace Common.CurveFitting
{
    public abstract class AbstractCurveFitting
    {
        public abstract string Formula { get; }
        private protected abstract int MinimumDataCount { get; }
        private protected abstract int ParamCount { get; }

        public abstract double[] Fit(double[] x, double[] y, out double r2);
        public abstract double CalculateX(double[] parms, double y);
        public abstract double CalculateY(double[] parms, double x);

        protected double CalcR2(double[] parms, double[] x, double[] y)
        {
            double sum1 = 0, sum2 = 0;
            var avg = y.Average();
            for (var i = 0; i < y.Length; i++)
            {
                var value = CalculateY(parms, x[i]);
                sum1 += (y[i] - value) * (y[i] - value);
                sum2 += (y[i] - avg) * (y[i] - avg);
            }

            return 1 - sum1 / sum2;
        }

        protected void CheckData(double[] x, double[] y)
        {
            if (x.Length != y.Length)
            {
                throw new ArgumentException("The length of two array should be equal");
            }

            if (x.Length < MinimumDataCount)
            {
                throw new ArgumentException($"The length of two array should be more than {MinimumDataCount}");
            }

            var newX = x.Select((value, index) => new {value, index}).OrderBy(t => t.value).ToArray();
            var isIncreasing = y[newX.Last().index] > y[newX.First().index];
            for (var i = 1; i < x.Length; i++)
            {
                if (isIncreasing && y[newX[i].index] < y[newX[i - 1].index])
                {
                    throw new ArgumentException("The data should be monotone increasing or decreasing");
                }

                if (!isIncreasing && y[newX[i].index] > y[newX[i - 1].index])
                {
                    throw new ArgumentException("The data should be monotone increasing or decreasing");
                }
            }
        }

        protected void CheckParam(double[] parms)
        {
            if (parms.Length != ParamCount)
            {
                throw new ArgumentException($"The length of parameters should be {ParamCount}");
            }
        }

        protected void AdjustZeroElement(double[] x)
        {
            for (var i = 0; i < x.Length; i++)
            {
                if (x[i] == 0)
                {
                    x[i] = x.Where(t => t != 0).Min() / 100;
                }
            }
        }
    }
}

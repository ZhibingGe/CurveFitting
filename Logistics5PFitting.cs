using System;
using System.Linq;
using MathNet.Numerics.LinearAlgebra.Double;

namespace Common.CurveFitting
{
    public class Logistics5PFitting : AbstractCurveFitting
    {
        public override string Formula => "y=(a-d)/(1+(x/c)^b)^g+d";
        private protected override int MinimumDataCount => 6;
        private protected override int ParamCount => 5;

        public override double[] Fit(double[] x, double[] y, out double r2)
        {
            CheckData(x, y);
            AdjustZeroElement(x);
            var a = y.Max() + 1;
            var d = y.Min() - 1;
            var g = 1d;
            var x2 = x.Select(t => Math.Log(t)).ToArray();
            var y2 = y.Select(t => Math.Log((t - a) / (d - t))).ToArray();
            var parms = new LinearFitting().Fit(x2, y2, out _);
            var b = parms[0];
            var c = Math.Exp(-parms[1] / b);
            var cnt = 1000;
            var factor = 0.03;
            r2 = 0;
            while (cnt-- > 0)
            {
                r2 = CalcR2(new[] {a, b, c, d, g}, x, y);
                if (r2 > 0.997)
                {
                    break;
                }

                var jacobi = new DenseMatrix(x.Length, 5);
                var rawMatrix = new DenseMatrix(x.Length, 1);
                for (var i = 0; i < x.Length; i++)
                {
                    jacobi[i, 0] = 1 / Math.Pow(1 + Math.Pow(x[i] / c, b), g);
                    jacobi[i, 1] = -(g * Math.Log(x[i] / c) * (a - d) * Math.Pow(x[i] / c, b)) /
                                   Math.Pow(1 + Math.Pow(x[i] / c, b), g + 1);
                    jacobi[i, 2] = b * g * x[i] * (a - d) * Math.Pow(x[i] / c, b - 1) / c / c /
                                   Math.Pow(1 + Math.Pow(x[i] / c, b), g + 1);
                    jacobi[i, 3] = 1 - 1 / Math.Pow(1 + Math.Pow(x[i] / c, b), g);
                    jacobi[i, 4] = -Math.Log(1 + Math.Pow(x[i] / c, b)) * (a - d) /
                                   Math.Pow(1 + Math.Pow(x[i] / c, b), g);
                    rawMatrix[i, 0] = y[i] - CalculateY(new[] {a, b, c, d, g}, x[i]);
                }

                var diff = (jacobi.Transpose() * jacobi).Inverse() * jacobi.Transpose() * rawMatrix;
                if (double.IsNaN(diff.L2Norm()) || diff.L2Norm() < 1e-5)
                {
                    break;
                }

                a += factor * diff[0, 0];
                b += factor * diff[1, 0];
                c += factor * diff[2, 0];
                d += factor * diff[3, 0];
                g += factor * diff[4, 0];
                if (double.IsNaN(a + b + c + d + g))
                {
                    throw new Exception("Cannot fit by given data");
                }
            }

            return new[] {a, b, c, d, g};
        }

        public override double CalculateX(double[] parms, double y)
        {
            CheckParam(parms);
            return parms[2] * Math.Pow(Math.Pow((parms[0] - parms[3]) / (y - parms[3]), 1 / parms[4]) - 1,
                1 / parms[1]);
        }

        public override double CalculateY(double[] parms, double x)
        {
            return (parms[0] - parms[3]) / Math.Pow(1 + Math.Pow(x / parms[2], parms[1]), parms[4]) + parms[3];
        }
    }
}

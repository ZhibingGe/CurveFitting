using System;
using System.Linq;
using MathNet.Numerics.LinearAlgebra.Double;

namespace Common.CurveFitting
{
    public class Logistics4PFitting : AbstractCurveFitting
    {
        public override string Formula => "y=(a-d)/(1+(x/c)^b)+d";
        private protected override int MinimumDataCount => 5;
        private protected override int ParamCount => 4;

        public override double[] Fit(double[] x, double[] y, out double r2)
        {
            CheckData(x, y);
            AdjustZeroElement(x);
            var a = y.Max() + 1;
            var d = y.Min() - 1;
            var x2 = x.Select(t => Math.Log(t)).ToArray();
            var y2 = y.Select(t => Math.Log((t - a) / (d - t))).ToArray();
            var parms = new LinearFitting().Fit(x2, y2, out _);
            var b = parms[0];
            var c = Math.Exp(-parms[1] / b);
            var cnt = 1000;
            var factor = 0.3;
            r2 = 0;
            while (cnt-- > 0)
            {
                r2 = CalcR2(new[] {a, b, c, d}, x, y);
                if (r2 > 0.997)
                {
                    break;
                }

                var jacobi = new DenseMatrix(x.Length, 4);
                var rawMatrix = new DenseMatrix(x.Length, 1);
                for (var i = 0; i < x.Length; i++)
                {
                    var temp = 1 + Math.Pow(x[i] / c, b);
                    jacobi[i, 0] = 1 / temp;
                    jacobi[i, 1] = (d - a) / temp / temp * (temp - 1) * Math.Log(x[i] / c);
                    jacobi[i, 2] = (a - d) / temp / temp * (temp - 1) / c * b;
                    jacobi[i, 3] = 1 - 1 / temp;
                    rawMatrix[i, 0] = y[i] - (a - d) / temp - d;
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
                if (double.IsNaN(a + b + c + d))
                {
                    throw new Exception("Cannot fit by given data");
                }
            }

            return new[] {a, b, c, d};
        }

        public override double CalculateX(double[] parms, double y)
        {
            CheckParam(parms);
            return parms[2] * Math.Pow((parms[0] - y) / (y - parms[3]), 1 / parms[1]);
        }

        public override double CalculateY(double[] parms, double x)
        {
            CheckParam(parms);
            return (parms[0] - parms[3]) / (1 + Math.Pow(x / parms[2], parms[1])) + parms[3];
        }
    }
}

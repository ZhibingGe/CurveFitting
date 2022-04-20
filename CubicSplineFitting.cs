using System;
using System.Linq;
using MathNet.Numerics.LinearAlgebra.Double;

namespace Common.CurveFitting
{
    public class CubicSplineFitting : AbstractCurveFitting
    {
        public override string Formula => "y=a*x^3+b*x^2+c*x+d";
        private protected override int MinimumDataCount => 2;
        private protected override int ParamCount => throw new NotSupportedException();

        public override double[] Fit(double[] x, double[] y, out double r2)
        {
            CheckData(x, y);
            var matrixL = new DenseMatrix(x.Length, x.Length);
            var matrixR = new DenseMatrix(x.Length, 1);
            r2 = 1;
            matrixL[0, 0] = 1;
            matrixL[x.Length - 1, x.Length - 1] = 1;
            matrixR[0, 0] = 0;
            matrixR[x.Length - 1, 0] = 0;
            for (var i = 1; i < x.Length - 1; i++)
            {
                matrixL[i, i - 1] = x[i] - x[i - 1];
                matrixL[i, i] = 2 * (x[i + 1] - x[i - 1]);
                matrixL[i, i + 1] = x[i + 1] - x[i];
                matrixR[i, 0] = 6 * (y[i + 1] - y[i]) / (x[i + 1] - x[i]) - 6 * (y[i] - y[i - 1]) / (x[i] - x[i - 1]);
            }

            var result = matrixL.Inverse() * matrixR;

            //y = a + b(x-x0)+c(x-x0)^2+d(x-x0)^3 -> y=ax^3+bx^2+cx+d;
            var parms = new double[x.Length * 8 - 8];
            var isIncreasing = y.Last() >= y.First();
            for (var i = 0; i < x.Length - 1; i++)
            {
                var h = x[i + 1] - x[i];
                var a = y[i];
                var b = (y[i + 1] - y[i]) / h - h / 2 * result[i, 0] - h / 6 * (result[i + 1, 0] - result[i, 0]);
                var c = result[i, 0] / 2;
                var d = (result[i + 1, 0] - result[i, 0]) / 6 / h;
                parms[i * 8 + 0] = d;
                parms[i * 8 + 1] = -3 * d * x[i] + c;
                parms[i * 8 + 2] = 3 * d * x[i] * x[i] - 2 * c * x[i] + b;
                parms[i * 8 + 3] = -d * x[i] * x[i] * x[i] + c * x[i] * x[i] - b * x[i] + a;
                parms[i * 8 + 4] = x[i];
                parms[i * 8 + 5] = x[i + 1];
                parms[i * 8 + 6] = y[i];
                parms[i * 8 + 7] = y[i + 1];
                if (!CheckIncreasing(parms[i * 8 + 0], parms[i * 8 + 1], parms[i * 8 + 2], x[i], x[i + 1],
                        isIncreasing))
                {
                    throw new Exception("The cubic line was not monotonous");
                }

                if (i == 0)
                {
                    var range = CheckRange(parms[i * 8 + 0], parms[i * 8 + 0], parms[i * 8 + 0], x[i], true);
                    parms[i * 8 + 4] = Math.Max(0, range);
                    parms[i * 8 + 6] = parms[i * 8 + 0] * Math.Pow(parms[i * 8 + 4], 3) +
                                       parms[i * 8 + 1] * Math.Pow(parms[i * 8 + 4], 2) +
                                       parms[i * 8 + 2] * parms[i * 8 + 4] + parms[i * 8 + 3];
                }
                else if (i == x.Length - 2)
                {
                    var range = CheckRange(parms[i * 8 + 0], parms[i * 8 + 0], parms[i * 8 + 0], x[i + 1], false);
                    parms[i * 8 + 5] = Math.Min(2 * x[i + 1], range);
                    parms[i * 8 + 7] = parms[i * 8 + 0] * Math.Pow(parms[i * 8 + 5], 3) +
                                       parms[i * 8 + 1] * Math.Pow(parms[i * 8 + 5], 2) +
                                       parms[i * 8 + 2] * parms[i * 8 + 5] + parms[i * 8 + 3];
                }
            }

            return parms;
        }

        private double CheckRange(double a, double b, double c, double x, bool isLeft)
        {
            // Derivative: y=3ax^2+2bx+c
            var value = 4 * b * b - 12 * a * c;
            double limitX;
            if (value <= 0)
            {
                if (isLeft)
                {
                    limitX = double.MinValue;
                }
                else
                {
                    limitX = double.MaxValue;
                }

                return limitX;
            }

            var zero0 = Math.Min(-(2 * b - Math.Sqrt(value)) / 6 / a, -(2 * b + Math.Sqrt(value)) / 6 / a);
            var zero1 = Math.Max(-(2 * b - Math.Sqrt(value)) / 6 / a, -(2 * b + Math.Sqrt(value)) / 6 / a);
            if (isLeft)
            {
                if (x < zero0)
                {
                    limitX = double.MinValue;
                }
                else if (zero0 <= x && x < zero1)
                {
                    limitX = zero0;
                }
                else
                {
                    limitX = zero1;
                }
            }
            else
            {
                if (x < zero0)
                {
                    limitX = zero0;
                }
                else if (zero0 <= x && x < zero1)
                {
                    limitX = zero1;
                }
                else
                {
                    limitX = double.MaxValue;
                }
            }

            return limitX;
        }

        private bool CheckIncreasing(double a,
            double b,
            double c,
            double x1,
            double x2,
            bool isIncreasing)
        {
            var derivative1 = 3 * a * x1 * x1 + 2 * b * x1 + c;
            var derivative2 = 3 * a * x2 * x2 + 2 * b * x2 + c;
            var derivative3 = 0d;
            var peakX = -b / 3 / a;
            if (x1 <= peakX && peakX <= x2)
            {
                derivative3 = 3 * a * peakX * peakX + 2 * b * peakX + c;
            }

            if (isIncreasing)
            {
                return derivative1 >= 0 && derivative2 >= 0 && derivative3 >= 0;
            }

            return derivative1 <= 0 && derivative2 <= 0 && derivative3 <= 0;
        }

        public override double CalculateX(double[] parms, double y)
        {
            var isIncreasing = parms.Last() >= parms[6];
            if (isIncreasing && y <= parms[6])
            {
                return parms[4];
            }

            if (isIncreasing && y >= parms.Last())
            {
                return parms[^3];
            }

            if (!isIncreasing && y >= parms[6])
            {
                return parms[4];
            }

            if (!isIncreasing && y <= parms.Last())
            {
                return parms[^3];
            }

            for (var i = 6; i < parms.Length; i += 8)
            {
                if (parms[i] <= y && y <= parms[i + 1])
                {
                    var leftX = parms[i - 2];
                    var rightX = parms[i - 1];
                    while (rightX - leftX > 1e-5)
                    {
                        var leftY = GetCubicPoly(parms[i - 6], parms[i - 5], parms[i - 4], parms[i - 3], leftX);
                        var rightY = GetCubicPoly(parms[i - 6], parms[i - 5], parms[i - 4], parms[i - 3], rightX);
                        if (Math.Abs(leftY - y) < 1e-5)
                        {
                            return leftX;
                        }

                        if (Math.Abs(rightY - y) < 1e-5)
                        {
                            return rightX;
                        }

                        var midX = leftX / 2 + rightX / 2;
                        var midY = GetCubicPoly(parms[i - 6], parms[i - 5], parms[i - 4], parms[i - 3], midX);
                        if ((y - midY) * (y - leftY) < 0)
                        {
                            rightX = midX;
                        }
                        else
                        {
                            leftX = midX;
                        }
                    }

                    return leftX / 2 + rightX / 2;
                }
            }

            return 0;
        }

        public override double CalculateY(double[] parms, double x)
        {
            if (x <= parms[4])
            {
                return parms[6];
            }

            if (x >= parms[^3])
            {
                return parms.Last();
            }

            for (var i = 4; i < parms.Length; i += 8)
            {
                if (parms[i] <= x && x <= parms[i + 1])
                {
                    return GetCubicPoly(parms[i - 4], parms[i - 3], parms[i - 2], parms[i - 1], x);
                }
            }

            return 0;
        }

        private double GetCubicPoly(double a, double b, double c, double d, double x)
        {
            return a * x * x * x + b * x * x + c * x + d;
        }
    }
}

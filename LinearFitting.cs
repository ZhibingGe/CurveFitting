namespace Common.CurveFitting
{
    public sealed class LinearFitting : AbstractCurveFitting
    {
        public override string Formula => "y=a*x+b";
        private protected override int MinimumDataCount => 2;
        private protected override int ParamCount => 2;

        public override double[] Fit(double[] x, double[] y, out double r2)
        {
            CheckData(x, y);
            double sumX = 0, sumY = 0, sumXx = 0, sumXy = 0;
            for (var i = 0; i < x.Length; i++)
            {
                sumX += x[i];
                sumXx += x[i] * x[i];
                sumY += y[i];
                sumXy += x[i] * y[i];
            }

            var a = (x.Length * sumXy - sumX * sumY) / (x.Length * sumXx - sumX * sumX);
            var b = (sumY - a * sumX) / x.Length;
            var parms = new[] {a, b};
            r2 = CalcR2(parms, x, y);
            return parms;
        }

        public override double CalculateX(double[] parms, double y)
        {
            CheckParam(parms);
            return (y - parms[1]) / parms[0];
        }

        public override double CalculateY(double[] parms, double x)
        {
            CheckParam(parms);
            return x * parms[0] + parms[1];
        }
    }
}

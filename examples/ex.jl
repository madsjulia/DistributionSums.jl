import DistributionSums
import Distributions
import PyPlot

u = Distributions.Uniform(0, 1)
upu = DistributionSums.DistributionSum(1001, [-1, 2], u, u)
@show Distributions.pdf(upu, 1.)
ts = collect(linspace(-2, 3, 51))
pdfs = map(t->Distributions.pdf(upu, t), ts)
@show pdfs
PyPlot.clf()
PyPlot.plot(ts, pdfs)

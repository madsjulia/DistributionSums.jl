using Base.Test
import DistributionSums
import Distributions

function basictest(N, u, bounds, truepdf)
	upu = DistributionSums.DistributionSum(N, bounds, u, u)
	xs = collect(linspace((2 * bounds)..., 6 * N))
	pdfs = map(x->Distributions.pdf(upu, x), xs)
	truepdfs = map(truepdf, xs)
	for i = 1:length(xs)
		@test pdfs[i] >= 0
		@test_approx_eq_eps pdfs[i] truepdfs[i] 4 / N
	end
end

function testuniform(N)
	u = Distributions.Uniform(0, 1)
	bounds = [-0.1, 1.1]
	truepdf(x) = x < 0 ? 0. : x < 1 ? x : x < 2 ? 2 - x : 0.
	basictest(N, u, bounds, truepdf)
end

function testnormal(N)
	u = Distributions.Normal(1, 2)
	bounds = [-6, 8]
	trueupu = Distributions.Normal(2, sqrt(8))
	truepdf(x) = Distributions.pdf(trueupu, x)
	basictest(N, u, bounds, truepdf)
end

function testgamma(N)
	u = Distributions.Gamma(1, 2)
	bounds = [-0.1, 16]
	trueupu = Distributions.Gamma(2, 2)
	truepdf(x) = Distributions.pdf(trueupu, x)
	basictest(N, u, bounds, truepdf)
end

testuniform(100)
testuniform(1000)
testuniform(10000)
testnormal(100)
testnormal(1000)
testnormal(10000)
testgamma(100)
testgamma(1000)
testgamma(10000)

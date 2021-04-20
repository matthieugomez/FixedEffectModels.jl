using DataFrames, FixedEffectModels, Random, CategoricalArrays

# Very simple setup
N = 10000000
K = 100
id1 = rand(1:div(N, K), N)
id2 = rand(1:K, N)
x1 = 5 * cos.(id1) + 5 * sin.(id2) + randn(N)
x2 =  cos.(id1) +  sin.(id2) + randn(N)
y= 3 .* x1 .+ 5 .* x2 .+ cos.(id1) .+ cos.(id2).^2 .+ randn(N)
df = DataFrame(id1 = id1, id2 = id2, x1 = x1, x2 = x2, y = y)
# first time
using SnoopCompileCore
inf_timing = @snoopi tmin=0.01 reg(df, @formula(y ~ x1 + x2))
using SnoopCompile
pc = SnoopCompile.parcel(inf_timing)
pc[:FixedEffectModels]
0.01
@time reg(df, @formula(y ~ x1 + x2))
# 14s
@time reg(df, @formula(y ~ x1 + x2))
# 0.582029 seconds (852 allocations: 535.311 MiB, 18.28% gc time)
@time reg(df, @formula(y ~ x1 + x2),  Vcov.cluster(:id2))
# 0.759032 seconds (631 allocations: 728.437 MiB, 4.89% gc time)
@time reg(df, @formula(y ~ x1 + x2 + fe(id1)))
# 1.316560 seconds (230.66 k allocations: 908.386 MiB, 4.28% gc time, 11.84% compilation time)
@time reg(df, @formula(y ~ x1 + x2 + fe(id1)), Vcov.cluster(:id1))
# 1.501165 seconds (230.94 k allocations: 952.029 MiB, 0.84% gc time, 10.66% compilation time)
@time reg(df, @formula(y ~ x1 + x2 + fe(id1) + fe(id2)))
# 3.058090 seconds (331.56 k allocations: 1.005 GiB, 2.08% gc time, 5.89% compilation time)


using Random, FixedEffects, BenchmarkTools
N = 10000000
K = 100
gid1 = FixedEffects.group(rand(1:div(N, K), N))
fecoefs = zeros(gid1.n)
refs = gid1.refs
y = rand(N)
cache = rand(N)
@btime FixedEffects.scatter!(y, 1, fecoefs, refs, cache, 1:N)
@btime FixedEffects.scatter!(y, 1, fecoefs, refs, cache, 4)


@btime FixedEffects.gather!(fecoefs, refs, 1.0, y, cache, 1:N)
@btime FixedEffects.gather!(fecoefs, refs, 1.0, y, cache, 4)



# More complicated setup
N = 800000 # number of observations
M = 40000 # number of workers
O = 5000 # number of firms
id1 = rand(1:M, N)
id2 = [rand(max(1, div(x, 8)-10):min(O, div(x, 8)+10)) for x in id1]
x_1 = randn(N)
x1 = 5 * cos.(id1) + 5 * sin.(id2) + x_1
x_2 = x + 0.1 .* randn(N)
x2 = 5 * cos.(id1) + 5 * sin.(id2) + x_2
y= 3 .* x1 .+ 5 .* x2 .+ cos.(id1) .+ cos.(id2).^2 .+ randn(N)
df = DataFrame(id1 = id1, id2 = id2, x1 = x1, x2 = x2, y = y, x_1 = x_1, x_2 = x_2)
@time reg(df, @formula(y ~ x1 + x2 + fe(id1) + fe(id2)))
#   3.190288 seconds (393.89 k allocations: 109.614 MiB, 3.63% gc time, 9.75% compilation time)
@time reg(df, @formula(y ~ x1 + x2 + fe(id1) + fe(id2)), tol = 1e-8)
@time reg(df, @formula(y ~ x_1 + x_2))

+# fixest
n = 10_000_000
nb_dum = [div(n,20), floor(Int, sqrt(n)), floor(Int, n^.33)]
N = nb_dum.^3
id1 = categorical(rand(1:nb_dum[1], n))
id2 = categorical(rand(1:nb_dum[2], n))
id3 = categorical(rand(1:nb_dum[3], n))
X1 = rand(n)
ln_y = 3 .* X1 .+ rand(n) 
df = DataFrame(X1 = X1, ln_y = ln_y, id1 = id1, id2 = id2, id3 = id3)
@time reg(df, @formula(ln_y ~ X1 + fe(id1)), Vcov.cluster(:id1))
# 1.058386 seconds (219.24 k allocations: 751.826 MiB, 13.05% compilation time)
@time reg(df, @formula(ln_y ~ X1 + fe(id1) + fe(id2)), Vcov.cluster(:id1))
# 2.895223 seconds (284.28 k allocations: 870.782 MiB, 1.73% gc time, 5.97% compilation time)
@time reg(df, @formula(ln_y ~ X1 + fe(id1) + fe(id2) + fe(id3)), Vcov.cluster(:id1))
# 3.191483 seconds (380.10 k allocations: 991.134 MiB, 6.88% compilation time)

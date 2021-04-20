"""
Estimate a linear model with high dimensional categorical variables / instrumental variables

### Arguments
* `df`: a Table
* `FormulaTerm`: A formula created using [`@formula`](@ref)
* `CovarianceEstimator`: A method to compute the variance-covariance matrix

### Keyword arguments
* `contrasts::Dict = Dict()` An optional Dict of contrast codings for each categorical variable in the `formula`.  Any unspecified variables will have `DummyCoding`.
* `weights::Union{Nothing, Symbol}` A symbol to refer to a columns for weights
* `save::Symbol`: Should residuals and eventual estimated fixed effects saved in a dataframe? Default to `:none` Use `save = :residuals` to only save residuals, `save = :fe` to only save fixed effects, `save = :all` for both. Once saved, they can then be accessed using `residuals()` or `fe()`. The returned DataFrame is automatically aligned with the original DataFrame.
* `method::Symbol`: A symbol for the method. Default is :cpu. Alternatively,  :gpu requires `CuArrays`. In this case, use the option `double_precision = false` to use `Float32`.
* `nthreads::Integer` Number of threads to use in the estimation. If `method = :cpu`, defaults to `Threads.nthreads()`. If `method = :gpu`, defaults to 256.
* `double_precision::Bool`: Should the demeaning operation use Float64 rather than Float32? Default to true.
* `tol::Real` Tolerance. Default to 1e-6.
* `maxiter::Integer = 10000`: Maximum number of iterations
* `drop_singletons::Bool = true`: Should singletons be dropped?
* `progress_bar::Bool = true`: Should the regression show a progressbar
* `first_stage::Bool = true`: Should the first-stage F-stat and p-value be computed?
* `dof_add::Integer = 0`: 
* `subset::Union{Nothing, AbstractVector} = nothing`: select specific rows 


### Details
Models with instruments variables are estimated using 2SLS. `reg` tests for weak instruments by computing the Kleibergen-Paap rk Wald F statistic, a generalization of the Cragg-Donald Wald F statistic for non i.i.d. errors. The statistic is similar to the one returned by the Stata command `ivreg2`.

### Examples
```julia
using RDatasets, FixedEffectModels
df = dataset("plm", "Cigar")
reg(df, @formula(Sales ~ NDI + fe(State) + fe(State)&Year))
reg(df, @formula(Sales ~ NDI + fe(State)*Year))
reg(df, @formula(Sales ~ (Price ~ Pimin)))
reg(df, @formula(Sales ~ NDI), Vcov.robust())
reg(df, @formula(Sales ~ NDI), Vcov.cluster(:State))
reg(df, @formula(Sales ~ NDI), Vcov.cluster(:State , :Year))
df.YearC = categorical(df.Year)
reg(df, @formula(Sales ~ YearC), contrasts = Dict(:YearC => DummyCoding(base = 80)))
```
"""

function reg(df, @nospecialize(formula::FormulaTerm), @nospecialize(vcov::CovarianceEstimator = Vcov.simple());
    contrasts::Dict = Dict{Symbol, Any}(),
    weights::Union{Symbol, Nothing} = nothing,
    save::Symbol = :none,
    method::Symbol = :cpu,
    nthreads::Integer = method == :cpu ? Threads.nthreads() : 256,
    double_precision::Bool = true,
    tol::Real = 1e-6,
    maxiter::Integer = 10000,
    drop_singletons::Bool = true,
    progress_bar::Bool = true,
    dof_add::Integer = 0,
    subset::Union{AbstractVector, Nothing} = nothing,
    first_stage::Bool = true)

    df = DataFrame(df; copycols = false)
    N = size(df, 1)

    ##############################################################################
    ##
    ## Parse formula
    ##
    ##############################################################################

    formula_origin = formula
    if  !omitsintercept(formula) & !hasintercept(formula)
        formula = FormulaTerm(formula.lhs, InterceptTerm{true}() + formula.rhs)
    end
    formula, formula_endo, formula_iv = parse_iv(formula)

    ##############################################################################
    ##
    ## Save keyword argument
    ##
    ##############################################################################
    if save ∉ (:all, :residuals, :fe, :none)
            throw("the save keyword argument must be a Symbol equal to :all, :none, :residuals or :fe")
    end
    save_residuals = (save == :residuals) | (save == :all)

    if method == :cpu && nthreads > Threads.nthreads()
        @warn "Keyword argument nthreads = $(nthreads) is ignored (Julia was started with only $(Threads.nthreads()) threads)."
        nthreads = Threads.nthreads()
    end
    ##############################################################################
    ##
    ## Construct new dataframe after removing missing values
    ##
    ##############################################################################

    # create a dataframe without missing values & negative weights
    vars = StatsModels.termvars(formula)
    iv_vars = Symbol[]
    endo_vars = Symbol[]
    if formula_iv !== nothing
        iv_vars = StatsModels.termvars(formula_iv)
        endo_vars = StatsModels.termvars(formula_endo)
    end
    # create a dataframe without missing values & negative weights
    all_vars = unique(vcat(vars, endo_vars, iv_vars))

    esample = completecases(df, all_vars)
    if weights !== nothing
        esample .&= BitArray(!ismissing(x) && (x > 0) for x in df[!, weights])
    end
    if subset !== nothing
        if length(subset) != N
            throw("df has $(N) rows but the subset vector has $(length(subset)) elements")
        end
        esample .&= BitArray(!ismissing(x) && x for x in subset)
    end
    esample .&= Vcov.completecases(df, vcov)
    fes, ids, fekeys, formula = parse_fixedeffect(df, formula)
    if drop_singletons
        for fe in fes
            drop_singletons!(esample, fe)
        end
    end
    save_fe = (save == :fe) | ((save == :all) &  !isempty(fes))
    nobs = sum(esample)
    (nobs > 0) || throw("sample is empty")

    if nobs == N
        esample = Colon()
    end

    # Compute weights
    if weights !== nothing
        weights = Weights(convert(Vector{Float64}, view(df, esample, weights)))
        all(isfinite, weights) || throw("Weights are not finite")
    else
        weights = uweights(nobs)
    end

    # Compute feM, an AbstractFixedEffectSolver
    has_intercept = hasintercept(formula)
    has_fe_intercept = false
    feM = nothing
    if !isempty(fes)
        has_fe_intercept = any(fe.interaction isa UnitWeights for fe in fes)
        fes = FixedEffect[fe[esample] for fe in fes]
        feM = AbstractFixedEffectSolver{double_precision ? Float64 : Float32}(fes, weights, Val{method}, nthreads)
    end
    # Compute data for std errors
    vcov_method = Vcov.materialize(view(df, esample, :), vcov)
    ##############################################################################
    ##
    ## Dataframe --> Matrix
    ##
    ##############################################################################
    exo_vars = unique(StatsModels.termvars(formula))
    subdf = Tables.columntable((; (x => disallowmissing(view(df[!, x], esample)) for x in exo_vars)...))
    s = schema(formula, subdf, contrasts)
    formula_schema = apply_schema(formula, s, FixedEffectModel, has_fe_intercept)

    # Obtain y
    # for a Vector{Float64}, conver(Vector{Float64}, y) aliases y
    y = convert(Vector{Float64}, response(formula_schema, subdf))
    all(isfinite, y) || throw("Some observations for the dependent variable are infinite")

    # Obtain X
    Xexo = convert(Matrix{Float64}, modelmatrix(formula_schema, subdf))
    all(isfinite, Xexo) || throw("Some observations for the exogeneous variables are infinite")

    response_name, coef_names = coefnames(formula_schema)
    if !(coef_names isa Vector)
        coef_names = typeof(coef_names)[coef_names]
    end

    coefendo_names = Symbol[]
    Xendo = Matrix{Float64}(undef, nobs, 0)
    Z = Matrix{Float64}(undef, nobs, 0)
    if formula_iv !== nothing
        Xendo, coefendo_names = compute_matrix_names(df, endo_vars, esample, formula_endo, contrasts)
        append!(coef_names, coefendo_names)
        Z, iv_names = compute_matrix_names(df, iv_vars, esample, formula_iv, contrasts)
        formula = FormulaTerm(formula.lhs, (tuple(eachterm(formula.rhs)..., (term for term in eachterm(formula_endo.rhs) if term != ConstantTerm(0))...)))
    end

    # compute tss now before potentially demeaning y
    tss_total = tss(y, has_intercept | has_fe_intercept, weights)
    # create unitilaized
    iterations, converged, tss_partial = 0, true, NaN
    oldy = y
    oldX = Xexo
    if !isempty(fes)
        # used to compute tss even without save_fe
        if save_fe
            oldy = deepcopy(y)
            oldX = hcat(Xexo, Xendo)
        end
        _, iterationss, convergeds = solve_residuals!(Combination(y, Xexo, Xendo, Z), feM; maxiter = maxiter, tol = tol, progress_bar = progress_bar)
        iterations = maximum(iterationss)
        converged = all(convergeds)
        if converged == false
            @info "Convergence not achieved in $(iterations) iterations; try increasing maxiter or decreasing tol."
        end
        tss_partial = tss(y, has_intercept | has_fe_intercept, weights)
    end

    _scale!(y, weights)
    _scale!(Xexo, weights)
    _scale!(Xendo, weights)
    _scale!(Z, weights)
    
    ##############################################################################
    ##
    ## Do the regression
    ##
    ##############################################################################
    # Compute linearly independent columns + create the Xhat matrix
    perm, basis_coef, Xexo, Xhat, X, Pi, Xendo_res, Pi2, Z_res = compute_linearly_independent(Xexo, Xendo, Z, coefendo_names)
    crossx = cholesky!(Symmetric(Xhat' * Xhat))
    coef = crossx \ (Xhat' * y)
    residuals = y - X * coef

    _helper(esample, vcov, vcov_method, formula_origin, formula, residuals, weights, save_residuals, N, oldy, oldX, X, Xhat, Pi, Xendo_res, Z_res, basis_coef, coef, tol, maxiter, fekeys, fes, feM, dof_add, nobs, crossx, has_intercept, has_fe_intercept, save_fe, first_stage, tss_partial, tss_total, perm, coef_names, response_name, contrasts, iterations, converged)::FixedEffectModel
end

function _helper(@nospecialize(esample), @nospecialize(vcov::CovarianceEstimator), @nospecialize(vcov_method::CovarianceEstimator), @nospecialize(formula_origin::FormulaTerm), @nospecialize(formula::FormulaTerm), residuals, weights, save_residuals, N, oldy, oldX, X, Xhat, Pi, Xendo_res, Z_res, basis_coef, coef, tol, maxiter, fekeys, fes, feM, dof_add, nobs, crossx, has_intercept, has_fe_intercept, save_fe, first_stage, tss_partial, tss_total, perm, coef_names, response_name, contrasts, iterations, converged)
    ##############################################################################
    ##
    ## Optionally save objects in a new dataframe
    ##
    ##############################################################################
    residuals2 = compute_residuals(residuals, esample, weights, save_residuals, N)

    df_fes = DataFrame()
    if save_fe
        newfes, b, c = solve_coefficients!(oldy - getcols(oldX[:, perm], basis_coef) * coef, feM; tol = tol, maxiter = maxiter)
        for fekey in fekeys
            df_fes[!, fekey] = df[:, fekey]
        end
        for j in eachindex(fes)
            df_fes[!, ids[j]] = Vector{Union{Float64, Missing}}(missing, N)
            df_fes[esample, ids[j]] = newfes[j]
        end
    end

    ##############################################################################
    ##
    ## Test Statistics
    ##
    ##############################################################################
    # Compute degrees of freedom
    dof_fes = compute_dof_fes(fes, vcov, vcov_method)
    dof_residual_ = max(1, nobs - size(X, 2) - dof_fes - dof_add)

    # Compute standard error
    nclusters = Vcov.nclusters(vcov_method)
    vcov_data = Vcov.VcovData(Xhat, crossx, residuals, dof_residual_)
    matrix_vcov = StatsBase.vcov(vcov_data, vcov_method)

    # Compute Fstat
    F = Fstat(coef, matrix_vcov, has_intercept)
    df_FStat_ = max(1, Vcov.df_FStat(vcov_data, vcov_method, has_intercept | has_fe_intercept))
    p = fdistccdf(max(length(coef) - (has_intercept | has_fe_intercept), 1), df_FStat_, F)
    # Compute Fstat of First Stage
    p_kp, F_kp = compute_Fkp_pkp(Pi, Xendo_res, Z_res, size(X, 2), dof_fes, vcov_method, first_stage)

    # Compute rss, tss, r2, r2 adjusted
    rss = sum(abs2, residuals)
    mss = tss_total - rss
    r2 = 1 - rss / tss_total
    adjr2 = 1 - rss / tss_total * (nobs - (has_intercept | has_fe_intercept)) / dof_residual_
    r2_within = 1 - rss / tss_partial

    ##############################################################################
    ##
    ## Return regression result
    ##
    ##############################################################################
    # add omitted variables
    coef, matrix_vcov = add_omitted_variables(coef, matrix_vcov, basis_coef)
    coef, matrix_vcov = undo_perm(coef, matrix_vcov, perm)

    return FixedEffectModel(coef, matrix_vcov, vcov, nclusters, esample == Colon() ? trues(N) : esample, residuals2, df_fes, fekeys, coef_names, response_name, formula_origin, formula, contrasts, nobs, dof_residual_,  rss, tss_total, r2, adjr2, F, p, iterations, converged, r2_within, F_kp, p_kp)
end



function compute_matrix_names(df, vars, esample, formula, contrasts = Dict{Symbol, Any}())
    subdf = Tables.columntable((; (x => disallowmissing(view(df[!, x], esample)) for x in vars)...))
    formula_schema = apply_schema(formula, schema(formula, subdf, contrasts), StatisticalModel)
    X = convert(Matrix{Float64}, modelmatrix(formula_schema, subdf))
    all(isfinite, X) || throw("Some observations are infinite")
    _, _coefnames = coefnames(formula_schema)
    if !(_coefnames isa Vector)
        _coefnames = typeof(_coefnames)[_coefnames]
    end
    return X, _coefnames
end


function compute_linearly_independent(Xexo, Xendo, Z, coefendo_names)
    if !isempty(Z)
        perm = 1:(size(Xexo, 2) + size(Xendo, 2))
        # first pass: remove colinear variables in Xendo
        basis_endo = basis(eachcol(Xendo)...)
        Xendo = getcols(Xendo, basis_endo)

        # second pass: remove colinear variable in Xexo, Z, and Xendo
        basis_all = basis(eachcol(Xexo)..., eachcol(Z)..., eachcol(Xendo)...)
        basis_Xexo = basis_all[1:size(Xexo, 2)]
        basis_Z = basis_all[(size(Xexo, 2) +1):(size(Xexo, 2) + size(Z, 2))]
        basis_endo_small = basis_all[(size(Xexo, 2) + size(Z, 2) + 1):end]
        if !all(basis_endo_small)
            # if adding Xexo and Z makes Xendo collinar, consider these variables are exogeneous
            Xexo = hcat(Xexo, getcols(Xendo, .!basis_endo_small))
            Xendo = getcols(Xendo, basis_endo_small)

            # out returns false for endo collinear with instruments
            basis_endo2 = trues(length(basis_endo))
            basis_endo2[basis_endo] = basis_endo_small

            # Change coef_names and oldX
            # TODO: I should probably also change formula in this case so that predict still works 
            ans = 1:length(basis_endo)
            ans = vcat(ans[.!basis_endo2], ans[basis_endo2])
            perm = vcat(1:length(basis_Xexo), length(basis_Xexo) .+ ans)

            out = join(coefendo_names[.!basis_endo2], " ")
            @info "Endogenous vars collinear with ivs. Recategorized as exogenous: $(out)"
                                    
            # third pass
            basis_all = basis(eachcol(Xexo)..., eachcol(Z)..., eachcol(Xendo)...)
            basis_Xexo = basis_all[1:size(Xexo, 2)]
            basis_Z = basis_all[(size(Xexo, 2) +1):(size(Xexo, 2) + size(Z, 2))]
        end

        Xexo = getcols(Xexo, basis_Xexo)
        Z = getcols(Z, basis_Z)
        size(Z, 2) >= size(Xendo, 2) || throw("Model not identified. There must be at least as many ivs as endogeneous variables")
        basis_coef = vcat(basis_Xexo, basis_endo[basis_endo_small])

        # Build
        newZ = hcat(Xexo, Z)
        Pi = cholesky!(Symmetric(newZ' * newZ)) \ (newZ' * Xendo)
        Xhat = hcat(Xexo, newZ * Pi)
        X = hcat(Xexo, Xendo)

        # prepare residuals used for first stage F statistic
        ## partial out Xendo in place wrt (Xexo, Z)
        Xendo_res = BLAS.gemm!('N', 'N', -1.0, newZ, Pi, 1.0, Xendo)
        ## partial out Z in place wrt Xexo
        Pi2 = cholesky!(Symmetric(Xexo' * Xexo)) \ (Xexo' * Z)
        Z_res = BLAS.gemm!('N', 'N', -1.0, Xexo, Pi2, 1.0, Z)
        return perm, basis_coef, Xexo, Xhat, X, Pi, Xendo_res, Pi2, Z_res
    else
        # get linearly independent columns
        basis_Xexo = basis(eachcol(Xexo)...)
        Xexo = getcols(Xexo, basis_Xexo)
        return collect(1:size(Xexo, 2)), basis_Xexo, Xexo, Xexo, Xexo, Matrix{Float64}[], Matrix{Float64}[], Matrix{Float64}[], Matrix{Float64}[]
    end
end

function compute_dof_fes(fes, vcov, vcov_method)
    out = 0
    for fe in fes
        # adjust degree of freedom only if fe is not fully nested in a cluster variable:
        if (vcov isa Vcov.ClusterCovariance) && any(isnested(fe, v.refs) for v in values(vcov_method.clusters))
            out += 1 # if fe is nested you still lose 1 degree of freedom
        else
            #only count groups that exists
            out += nunique(fe)
        end
    end
    return out
end

function compute_Fkp_pkp(Pi, Xendo_res, Z_res, n_X, dof_fes, vcov_method, first_stage)
    if first_stage & !isempty(Pi)
        Pip = Pi[(size(Pi, 1) - size(Z_res, 2) + 1):end, :]
        try 
            r_kp = Vcov.ranktest!(Xendo_res, Z_res, Pip, vcov_method, n_X, dof_fes)
            p_kp = chisqccdf(size(Z_res, 2) - size(Xendo_res, 2) + 1, r_kp)
            F_kp = r_kp / size(Z_res, 2)
            return p_kp, F_kp
        catch
            @info "ranktest failed ; first-stage statistics not estimated"
            return NaN, NaN
        end
    else
        return NaN, NaN
    end
end

function add_omitted_variables(coef, matrix_vcov, basis_coef)
    if all(basis_coef)
        return coef, matrix_vcov
    else
        newcoef = zeros(length(basis_coef))
        newmatrix_vcov = fill(NaN, (length(basis_coef), length(basis_coef)))
        newindex = [searchsortedfirst(cumsum(basis_coef), i) for i in 1:length(coef)]
        for i in eachindex(newindex)
            newcoef[newindex[i]] = coef[i]
            for j in eachindex(newindex)
                newmatrix_vcov[newindex[i], newindex[j]] = matrix_vcov[i, j]
            end
        end
        return newcoef, Symmetric(newmatrix_vcov)
    end
end

function undo_perm(coef, matrix_vcov, perm)
    if all(perm[i] == i for i in perm)
        return coef, matrix_vcov
    else
        _invperm = invperm(perm)
        newcoef = coef[_invperm]
        newmatrix_vcov = zeros(size(matrix_vcov))
        for i in 1:size(newmatrix_vcov, 1)
            for j in 1:size(newmatrix_vcov, 1)
                newmatrix_vcov[i, j] = matrix_vcov[_invperm[i], _invperm[j]]
            end
        end
        return newcoef, Symmetric(newmatrix_vcov)
    end
end

_scale!(y, weights::UnitWeights) = y
function _scale!(y, weights)
    y .= y .* sqrt.(weights)
end

_unscale!(y, weights::UnitWeights) = y
function _unscale!(y, weights)
    y .= y ./ sqrt.(weights)
end

function compute_residuals(residuals, esample, weights, save_residuals, N)
    if save_residuals
        out = Vector{Union{Float64, Missing}}(missing, N)
        out[esample] = _unscale!(residuals, weights)
        return out
    else
        return nothing
    end
end

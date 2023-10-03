def steepest_decent_backtracking():
    # %INITIALISATION
    # 1 Set a, g % set parameters of the objective function
    a = [1, 1]
    g = [1, 1]
    # 2 Set αinit, 0 < c < 1, 0 < r < 1, nreset, u % set parameters of the algorithm
    # %CONJUGATE GRADIENT ALGORITHM
    # 3 Compute mnew = m(u)
    # 4 α = αinit, mold = 10100, f new = 0, hnew = 0, cnt = 0 % dummy values and counter
    # 5 while mnew < mold
    # 6 mold = mnew , f old = f new , hold = hnew
    # 7 Compute f new = ∂m/∂u ∣u
    # %Determine search direction
    # 8 if mod(cnt, nreset) = 0
    # 9 hnew = −f new
    # 10 else
    # 11 β according to Eq. (4.6), (4.7), (4.8) or (4.9)
    # 12 hnew = −f new + max([0, β]) hold
    # 13 end
    # %Backtracking line search
    # 14 α = 1
    # r αinit
    # 15 mx = 10100 % dummy value
    # 16 while mx > mnew + c αhnewT f new
    # 17 α = r α % decrease stepsize by factor r
    # 18 ux = u + α hnew
    # 19 Compute mx = m(ux)
    # 20 end
    # 21 mnew = mx, u = ux, cnt = cnt + 1
    # 22 end
    # %COMPLETION
    # 23 m∗ = mold, u∗ = u − αhnew

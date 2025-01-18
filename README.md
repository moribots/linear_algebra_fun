# linear_algebra_fun
A  place to brush up on practical applications of linear algebra.

# Project 1 - Linear Regression
Predicting y from x. 
y = mx + b + ϵ
where ϵ is random noise

## 1. Matrix Form:
y = Xβ + ϵ

where:
y is the target vector.

X is the design matrix (including a column of ones for the intercept).

β is the vector of coefficients to solve for.

## 2. Closed-form solution (Normal Equation):
β = (𝑋.T 𝑋).inv() * 𝑋.T y


## 3. Implement gradient descent to minimize the mean squared error:
J(β) = (1/2n) ∥Xβ − y∥^2


Update rule:
βk + 1 = βk −n∇J(βk)

where:

∇J(β) = (1/n) X.T(Xβ − y)


## 4. Compare Solutions

Compare the coefficients β = [m, b] from 2. and 3.

Plot above to visualize.


## 5. Analyze Performance
Compute MSE.

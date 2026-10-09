# Deep Learning Mathematics Foundations

## Purpose

These notes cover the mathematical foundations required to understand Deep Learning deeply.

The goal is **not** to learn all university-level mathematics.

The goal is to learn the mathematics that actually appears in:

* Neural networks
* Backpropagation
* Optimization
* Loss functions
* Computer Vision
* Transformers
* Embeddings
* Training stability
* Probability-based models

The target is to understand what happens mathematically rather than simply using PyTorch APIs.

---

# 1. Mathematics Roadmap

```text
Mathematics
│
├── 1. Algebra
│
├── 2. Functions
│
├── 3. Linear Algebra
│   ├── Scalars
│   ├── Vectors
│   ├── Matrices
│   ├── Tensors
│   ├── Dot Product
│   ├── Matrix Multiplication
│   ├── Transpose
│   ├── Norms
│   ├── Eigenvalues
│   └── Eigenvectors
│
├── 4. Calculus
│   ├── Derivatives
│   ├── Partial Derivatives
│   ├── Chain Rule
│   ├── Gradients
│   ├── Jacobians
│   └── Hessians
│
├── 5. Probability
│   ├── Probability
│   ├── Random Variables
│   ├── Distributions
│   ├── Expectation
│   ├── Variance
│   ├── Covariance
│   └── Conditional Probability
│
├── 6. Statistics
│   ├── Mean
│   ├── Variance
│   ├── Standard Deviation
│   ├── Sampling
│   ├── Estimation
│   └── Maximum Likelihood
│
└── 7. Optimization
    ├── Objective Functions
    ├── Gradient Descent
    ├── SGD
    ├── Momentum
    ├── Adam
    ├── AdamW
    ├── Learning Rate
    └── Convexity
```

---

# 2. Algebra

Algebra is the foundation for the rest of the mathematics.

## 2.1 Variables

A variable represents a value that can change.

$$
x=5
$$

---

## 2.2 Constants

A constant is a value that does not change.

$$
a=10
$$

---

## 2.3 Basic Operations

Addition:

$$
a+b
$$

Subtraction:

$$
a-b
$$

Multiplication:

$$
ab
$$

Division:

$$
\frac{a}{b}
$$

Power:

$$
x^2
$$

Square root:

$$
\sqrt{x}
$$

---

# 3. Algebraic Manipulation

Be comfortable with:

$$
(a+b)^2
$$

$$
(a-b)^2
$$

$$
(a+b)(a-b)
$$

Factorization:

$$
x^2-9=(x-3)(x+3)
$$

These operations appear frequently when deriving mathematical expressions.

---

# 4. Exponents

Important rules:

$$
x^a x^b=x^{a+b}
$$

$$
\frac{x^a}{x^b}=x^{a-b}
$$

$$
(x^a)^b=x^{ab}
$$

Exponents are especially important because neural networks frequently use exponential functions.

---

# 5. Logarithms

Logarithms are extremely important in Deep Learning.

They appear in:

* Cross Entropy
* Maximum Likelihood
* Softmax
* Probability
* Numerical stability

Important rules:

$$
\log(ab)=\log a+\log b
$$

$$
\log\left(\frac{a}{b}\right)
=
\log a-\log b
$$

$$
\log(a^b)=b\log a
$$

---

# 6. Natural Logarithm

You will frequently encounter:

$$
\ln(x)
$$

Its derivative is:

$$
\frac{d}{dx}\ln(x)=\frac{1}{x}
$$

---

# 7. Functions

A function maps an input to an output.

$$
y=f(x)
$$

Example:

$$
f(x)=x^2
$$

If:

$$
x=3
$$

then:

$$
f(3)=9
$$

---

# 8. Composition of Functions

Functions can be nested:

$$
f(g(x))
$$

For example:

$$
g(x)=x^2
$$

and:

$$
f(x)=3x+1
$$

Then:

$$
f(g(x))=3x^2+1
$$

This concept is fundamental to backpropagation.

---

# 9. Linear Algebra

Linear Algebra is one of the most important mathematical foundations for Deep Learning.

A neural network contains operations such as:

$$
Wx+b
$$

---

# 10. Scalar

A scalar is a single number.

$$
x=5
$$

Examples:

```text
5
0.25
-3
```

---

# 11. Vector

A vector is an ordered collection of numbers.

$$
x=
\begin{bmatrix}
2\\
4\\
6
\end{bmatrix}
$$

This vector belongs to:

$$
\mathbb{R}^3
$$

because it contains three elements.

---

# 12. Matrix

A matrix is a rectangular collection of numbers.

$$
A=
\begin{bmatrix}
1&2&3\\
4&5&6
\end{bmatrix}
$$

Its shape is:

$$
2\times3
$$

Meaning:

```text
2 rows
3 columns
```

---

# 13. Tensor

A tensor generalizes scalars, vectors, and matrices to higher dimensions.

```text
Scalar → 0D
Vector → 1D
Matrix → 2D
Tensor → 3D+
```

For example, an RGB image can be represented as:

$$
H\times W\times3
$$

A batch of RGB images might be:

$$
N\times H\times W\times C
$$

---

# 14. Vector Addition

If:

$$
a=
\begin{bmatrix}
1\\
2
\end{bmatrix}
$$

and:

$$
b=
\begin{bmatrix}
3\\
4
\end{bmatrix}
$$

then:

$$
a+b=
\begin{bmatrix}
4\\
6
\end{bmatrix}
$$

---

# 15. Scalar Multiplication

$$
3
\begin{bmatrix}
1\\
2
\end{bmatrix}
=
\begin{bmatrix}
3\\
6
\end{bmatrix}
$$

---

# 16. Dot Product

Given:

$$
a=
\begin{bmatrix}
1\\
2\\
3
\end{bmatrix}
$$

and:

$$
b=
\begin{bmatrix}
4\\
5\\
6
\end{bmatrix}
$$

the dot product is:

$$
a^Tb
=
1(4)+2(5)+3(6)
$$

$$
=32
$$

Dot products are fundamental to:

* Linear layers
* Similarity
* Embeddings
* Attention

---

# 17. Matrix Multiplication

If:

$$
A\in\mathbb{R}^{m\times n}
$$

and:

$$
B\in\mathbb{R}^{n\times p}
$$

then:

$$
AB\in\mathbb{R}^{m\times p}
$$

The inner dimensions must match:

```text
(m × n) × (n × p)
       ↓
     m × p
```

---

# 18. Transpose

Given:

$$
A=
\begin{bmatrix}
1&2&3\\
4&5&6
\end{bmatrix}
$$

the transpose is:

$$
A^T=
\begin{bmatrix}
1&4\\
2&5\\
3&6
\end{bmatrix}
$$

The shape changes:

$$
m\times n
\rightarrow
n\times m
$$

---

# 19. Identity Matrix

The identity matrix is:

$$
I=
\begin{bmatrix}
1&0\\
0&1
\end{bmatrix}
$$

It satisfies:

$$
AI=A
$$

---

# 20. Inverse Matrix

If a matrix is invertible:

$$
AA^{-1}=I
$$

You should understand the concept, but do not make matrix inversion a major focus for Deep Learning.

In practical numerical computing, we often solve systems without explicitly computing the inverse.

---

# 21. Norms

A norm measures the size or magnitude of a vector.

## L1 Norm

$$
||x||_1
=
\sum_i|x_i|
$$

## L2 Norm

$$
||x||_2
=
\sqrt{\sum_i x_i^2}
$$

Norms appear in:

* Regularization
* Distance calculations
* Gradient clipping
* Optimization

---

# 22. Calculus

Calculus is central to understanding neural-network training.

---

# 23. Derivative

The derivative measures the rate of change.

If:

$$
y=f(x)
$$

then:

$$
\frac{dy}{dx}
$$

measures how the output changes when the input changes.

---

# 24. Partial Derivative

For a function with multiple variables:

$$
f(x,y)
$$

we can calculate:

$$
\frac{\partial f}{\partial x}
$$

and:

$$
\frac{\partial f}{\partial y}
$$

When calculating the derivative with respect to \(x\), treat \(y\) as constant.

---

# 25. Gradient

For:

$$
f(x_1,x_2,\dots,x_n)
$$

the gradient is:

$$
\nabla f=
\begin{bmatrix}
\frac{\partial f}{\partial x_1}\\
\frac{\partial f}{\partial x_2}\\
\vdots\\
\frac{\partial f}{\partial x_n}
\end{bmatrix}
$$

The gradient tells us the direction of greatest local increase.

In Deep Learning:

$$
\nabla_\theta L
$$

represents the gradient of the loss with respect to the model parameters.

---

# 26. Chain Rule

If:

$$
y=f(g(x))
$$

then:

$$
\frac{dy}{dx}
=
\frac{dy}{dg}
\frac{dg}{dx}
$$

The Chain Rule is the mathematical foundation of backpropagation.

---

# 27. Chain Rule Example

Suppose:

$$
z=x^2
$$

and:

$$
y=3z
$$

Then:

$$
\frac{dy}{dz}=3
$$

and:

$$
\frac{dz}{dx}=2x
$$

Therefore:

$$
\frac{dy}{dx}
=
\frac{dy}{dz}
\frac{dz}{dx}
$$

$$
=3(2x)
$$

$$
=6x
$$

---

# 28. Computational Graphs

A computational graph represents mathematical operations as connected nodes.

```text
x
↓
g(x)
↓
f(g(x))
↓
Loss
```

During the forward pass, we calculate values.

During the backward pass, we calculate gradients.

---

# 29. Backpropagation

Backpropagation is the systematic application of the Chain Rule through a computational graph.

```text
Forward:

Input
 ↓
Layer
 ↓
Layer
 ↓
Loss


Backward:

Loss
 ↓
Gradient
 ↓
Layer
 ↓
Gradient
 ↓
Parameters
```

The key idea is:

> Backpropagation tells us how much each parameter contributed to the final loss.

---

# 30. Jacobian

For:

$$
f:\mathbb{R}^n\rightarrow\mathbb{R}^m
$$

the Jacobian is the matrix of partial derivatives:

$$
J=
\begin{bmatrix}
\frac{\partial f_1}{\partial x_1}&
\cdots&
\frac{\partial f_1}{\partial x_n}\\
\vdots&
\ddots&
\vdots\\
\frac{\partial f_m}{\partial x_1}&
\cdots&
\frac{\partial f_m}{\partial x_n}
\end{bmatrix}
$$

Jacobians are useful for understanding automatic differentiation and vector-valued functions.

---

# 31. Hessian

The Hessian contains second-order derivatives.

$$
H=\nabla^2 f
$$

with:

$$
H_{ij}
=
\frac{\partial^2f}
{\partial x_i\partial x_j}
$$

The Hessian helps describe the curvature of a function.

It is relevant to:

* Optimization
* Loss landscapes
* Second-order methods
* Curvature

You do not need to use Hessians every day, but you should understand the concept.

---

# 32. Taylor Expansion

A first-order Taylor approximation is:

$$
f(x+\Delta x)
\approx
f(x)+\nabla f(x)^T\Delta x
$$

This helps explain:

* Gradient descent
* Local approximation
* Optimization

---

# 33. Optimization

Training a model means finding parameters that minimize a loss function.

The general problem is:

$$
\min_\theta L(\theta)
$$

where:

* \(\theta\) = parameters
* \(L\) = loss function

---

# 34. Gradient Descent

The basic update rule is:

$$
\theta_{t+1}
=
\theta_t-\eta\nabla_\theta L
$$

where:

* \(\theta\) = parameters
* \(\eta\) = learning rate
* \(\nabla_\theta L\) = gradient

The negative sign is important because the gradient points toward increasing loss.

We move in the opposite direction to reduce the loss.

---

# 35. SGD

Stochastic Gradient Descent uses a subset of the training data to estimate the gradient.

Typical training flow:

```text
Dataset
 ↓
Mini-batch
 ↓
Forward
 ↓
Loss
 ↓
Backward
 ↓
Gradient
 ↓
Parameter Update
```

---

# 36. Momentum

Momentum keeps information about previous gradients.

A simplified formulation is:

$$
v_t=\beta v_{t-1}+g_t
$$

and:

$$
\theta_t
=
\theta_{t-1}-\eta v_t
$$

Momentum can help:

* Reduce oscillation
* Accelerate movement in consistent directions

---

# 37. Adam

Adam combines ideas related to:

* Momentum
* Adaptive learning rates

It tracks estimates of:

* First moment of the gradient
* Second moment of the gradient

Conceptually:

```text
Gradient
   │
   ├──→ First Moment
   │
   └──→ Second Moment
              │
              ↓
       Adaptive Update
```

---

# 38. AdamW

AdamW separates weight decay from the adaptive gradient update.

This is important because:

```text
Adam + L2 regularization
```

is not equivalent to:

```text
AdamW
```

when using adaptive optimization.

---

# 39. Convexity

A convex function has a structure that makes optimization easier to analyze.

Deep neural-network loss landscapes are generally non-convex.

Important concepts include:

* Global minimum
* Local minimum
* Saddle point
* Flat region
* Steep region

You should understand these concepts, but do not obsess over proving advanced convex-analysis theorems.

---

# 40. Probability

Probability is important for:

* Classification
* Generative models
* Bayesian methods
* Maximum likelihood
* Cross entropy
* Uncertainty

---

# 41. Probability Basics

For an event \(A\):

$$
0\le P(A)\le1
$$

---

# 42. Conditional Probability

$$
P(A|B)
=
\frac{P(A\cap B)}
{P(B)}
$$

This means:

> The probability of A given that B has occurred.

---

# 43. Bayes' Theorem

$$
P(A|B)
=
\frac{P(B|A)P(A)}
{P(B)}
$$

Important for probabilistic machine learning.

---

# 44. Random Variables

A random variable represents the outcome of a random process.

For example:

$$
X
$$

could represent:

* A measurement
* A class
* A continuous value
* A discrete outcome

---

# 45. Probability Distributions

A probability distribution describes how probability is assigned to possible values.

Important distributions include:

* Bernoulli
* Binomial
* Gaussian
* Uniform
* Poisson

You do not need to memorize every distribution.

Focus on understanding what each distribution represents and where it is used.

---

# 46. Expected Value

The expected value is:

$$
E[X]
$$

It represents the theoretical average value of a random variable.

---

# 47. Variance

$$
Var(X)
=
E[(X-E[X])^2]
$$

Variance measures how spread out values are around the mean.

---

# 48. Standard Deviation

$$
\sigma
=
\sqrt{Var(X)}
$$

Standard deviation measures spread in the same units as the original variable.

---

# 49. Covariance

Covariance measures how two variables change together.

$$
Cov(X,Y)
=
E[(X-E[X])(Y-E[Y])]
$$

It is important for:

* Feature relationships
* Covariance matrices
* PCA

---

# 50. Correlation

Correlation normalizes covariance:

$$
\rho_{XY}
=
\frac{Cov(X,Y)}
{\sigma_X\sigma_Y}
$$

and:

$$
-1\le\rho\le1
$$

---

# 51. Statistics

Statistics is important for understanding:

* Data
* Sampling
* Model evaluation
* Generalization
* Experiments

---

# 52. Mean

$$
\bar{x}
=
\frac{1}{n}
\sum_{i=1}^{n}x_i
$$

---

# 53. Median

The median is the middle value after sorting the data.

---

# 54. Sample Variance

$$
s^2
=
\frac{1}{n-1}
\sum_{i=1}^{n}
(x_i-\bar{x})^2
$$

---

# 55. Standardization

A common transformation is:

$$
z=
\frac{x-\mu}{\sigma}
$$

This produces data with approximately:

$$
mean=0
$$

and:

$$
std=1
$$

---

# 56. Maximum Likelihood

Maximum Likelihood Estimation searches for parameters that make the observed data most likely.

$$
\hat{\theta}
=
\arg\max_\theta
P(D|\theta)
$$

In practice, we often maximize the log-likelihood instead.

---

# 57. Loss Functions

A loss function measures how wrong the model prediction is.

---

## Mean Squared Error

$$
MSE
=
\frac{1}{n}
\sum_i
(y_i-\hat y_i)^2
$$

Commonly used for regression.

---

## Mean Absolute Error

$$
MAE
=
\frac{1}{n}
\sum_i
|y_i-\hat y_i|
$$

---

# 58. Binary Cross Entropy

For binary classification:

$$
L=
-
[y\log(p)+(1-y)\log(1-p)]
$$

This connects:

```text
Probability
   ↓
Logarithm
   ↓
Likelihood
   ↓
Loss
   ↓
Optimization
```

---

# 59. Cross Entropy

Cross entropy is fundamental to classification.

It connects directly to:

* Probability
* Logarithms
* Maximum likelihood
* Softmax

---

# 60. Softmax

Softmax converts scores into probabilities:

$$
softmax(z_i)
=
\frac{e^{z_i}}
{\sum_j e^{z_j}}
$$

The resulting probabilities sum to:

$$
\sum_i p_i=1
$$

---

# 61. Numerical Stability

Correct mathematics is not enough.

Deep Learning also requires numerical stability.

Common problems:

```text
Overflow
Underflow
NaN
Inf
```

For example:

$$
e^{1000}
$$

can overflow numerically.

A common stable softmax formulation is:

$$
softmax(z)
=
softmax(z-\max(z))
$$

because subtracting the same constant does not change the resulting probabilities.

---

# 62. Activation Functions

You should understand the mathematical behavior of common activation functions.

---

## Sigmoid

$$
\sigma(x)
=
\frac{1}{1+e^{-x}}
$$

---

## Tanh

$$
tanh(x)
=
\frac{e^x-e^{-x}}
{e^x+e^{-x}}
$$

---

## ReLU

$$
ReLU(x)=\max(0,x)
$$

---

## GELU

GELU is widely used in modern neural-network architectures.

You should understand its behavior and motivation before worrying about memorizing its exact approximation.

---

# 63. Activation Derivatives

Backpropagation requires derivatives of activation functions.

For ReLU:

$$
ReLU'(x)
=
\begin{cases}
1 & x>0\\
0 & x<0
\end{cases}
$$

This eventually connects to:

* Vanishing gradients
* Dead neurons
* Gradient flow
* Training stability

---

# 64. Eigenvalues and Eigenvectors

If:

$$
Av=\lambda v
$$

then:

* \(v\) is an eigenvector.
* \(\lambda\) is its eigenvalue.

These concepts are relevant to:

* PCA
* Covariance matrices
* Optimization
* Stability
* Spectral analysis

---

# 65. Singular Value Decomposition

SVD decomposes a matrix as:

$$
A=U\Sigma V^T
$$

It is useful for:

* Dimensionality reduction
* Matrix approximation
* PCA
* Low-rank representations

---

# 66. PCA

Principal Component Analysis uses:

* Linear algebra
* Covariance
* Eigenvectors
* Eigenvalues

The main idea is:

> Find directions that capture the largest amount of variance in the data.

---

# 67. Distance Metrics

Distance and similarity measures are important for:

* Clustering
* Embeddings
* Retrieval
* Computer Vision
* Representation learning

---

## Euclidean Distance

$$
d(x,y)
=
\sqrt{\sum_i(x_i-y_i)^2}
$$

---

## Manhattan Distance

$$
d(x,y)
=
\sum_i|x_i-y_i|
$$

---

## Cosine Similarity

$$
\cos(\theta)
=
\frac{x^Ty}
{||x||||y||}
$$

Cosine similarity is particularly important for embeddings and similarity search.

---

# 68. Broadcasting

Broadcasting is important when working with NumPy and PyTorch.

For example:

```python
X.shape = (32, 128)
b.shape = (128,)
```

Then:

```python
X + b
```

is valid because `b` is broadcast across the batch dimension.

You should understand why this works rather than simply relying on it.

---

# 69. Tensor Shape Mathematics

In Deep Learning, tensor shapes should become part of your thinking.

Example:

```text
X: (batch, features)
W: (features, hidden)
b: (hidden,)
```

Then:

```text
X @ W
```

produces:

```text
(batch, hidden)
```

And:

```text
X @ W + b
```

also produces:

```text
(batch, hidden)
```

Understanding shapes is one of the most useful skills for debugging neural networks.

---

# 70. Priority Levels

Do not study every topic with equal depth.

## Must Know Deeply

```text
✓ Functions
✓ Algebra
✓ Vectors
✓ Matrices
✓ Matrix multiplication
✓ Dot product
✓ Transpose
✓ Broadcasting
✓ Derivatives
✓ Partial derivatives
✓ Chain rule
✓ Gradients
✓ Computational graphs
✓ Backpropagation
✓ Probability basics
✓ Expectation
✓ Variance
✓ Loss functions
✓ Gradient descent
✓ SGD
✓ Momentum
✓ Adam
✓ AdamW
```

---

## Should Understand

```text
✓ Jacobians
✓ Hessians
✓ Eigenvalues
✓ Eigenvectors
✓ SVD
✓ PCA
✓ Taylor expansion
✓ Convexity
✓ Maximum likelihood
✓ Covariance
✓ Numerical stability
```

---

## Learn Later or Only When Needed

```text
○ Measure theory
○ Real analysis
○ Abstract algebra
○ Advanced differential geometry
○ Advanced topology
○ Functional analysis
```

These are not prerequisites for becoming a practical Deep Learning Engineer.

---

# 71. Correct Study Order

Do not randomly jump between topics.

Use this dependency order:

```text
Algebra
   ↓
Functions
   ↓
Linear Algebra
   ↓
Derivatives
   ↓
Partial Derivatives
   ↓
Chain Rule
   ↓
Gradients
   ↓
Jacobians
   ↓
Computational Graphs
   ↓
Backpropagation
   ↓
Probability
   ↓
Statistics
   ↓
Loss Functions
   ↓
Optimization
   ↓
Numerical Stability
```

---

# 72. Mathematics → Deep Learning Connection

The goal is to connect every mathematical concept to something inside a neural network.

## Linear Algebra

```text
Linear Algebra
      ↓
Matrix multiplication
      ↓
Linear layer
      ↓
Neural network
```

For example:

$$
Wx+b
$$

is not just a matrix equation.

It is the mathematical operation behind a neural-network linear layer.

---

## Calculus

```text
Calculus
   ↓
Derivative
   ↓
Gradient
   ↓
Chain Rule
   ↓
Backpropagation
```

---

## Optimization

```text
Gradient
   ↓
Learning rate
   ↓
Parameter update
   ↓
Better model
```

---

## Probability

```text
Probability
   ↓
Likelihood
   ↓
Cross Entropy
   ↓
Classification
```

---

# 73. The Complete Deep Learning Mathematical Picture

You should eventually see training like this:

```text
Input
  ↓
Linear Algebra
  ↓
Forward Pass
  ↓
Prediction
  ↓
Loss Function
  ↓
Calculus
  ↓
Chain Rule
  ↓
Backpropagation
  ↓
Gradient
  ↓
Optimization
  ↓
Parameter Update
  ↓
Better Model
```

This is the mathematical foundation of neural-network training.

---

# 74. Definition of Mastery

Do not consider a mathematical topic learned simply because you can solve one textbook exercise.

For each important concept, you should be able to:

1. Explain the intuition.
2. Write the mathematical definition.
3. Solve a numerical example.
4. Implement it with NumPy.
5. Recognize it inside PyTorch.
6. Explain why it is used in Deep Learning.
7. Identify common mistakes.

For example, for a gradient:

```text
Explain it
    ↓
Write it
    ↓
Calculate it
    ↓
Implement it
    ↓
Use it in a neural network
    ↓
Understand its role in training
```

---

# 75. Final Mathematics Roadmap

```text
FOUNDATION
│
├── Algebra
├── Functions
└── Logarithms
        ↓
LINEAR ALGEBRA
│
├── Scalars
├── Vectors
├── Matrices
├── Dot Product
├── Matrix Multiplication
├── Transpose
├── Norms
└── Tensors
        ↓
CALCULUS
│
├── Derivatives
├── Partial Derivatives
├── Chain Rule
├── Gradients
├── Jacobians
└── Hessians
        ↓
PROBABILITY & STATISTICS
│
├── Probability
├── Random Variables
├── Distributions
├── Expectation
├── Variance
├── Covariance
└── Maximum Likelihood
        ↓
OPTIMIZATION
│
├── Objective Functions
├── Gradient Descent
├── SGD
├── Momentum
├── Adam
└── AdamW
        ↓
DEEP LEARNING
│
├── Forward Propagation
├── Backpropagation
├── Training
├── Regularization
├── Normalization
└── Modern Neural Networks
```

# Final Rule

Do not memorize Deep Learning mathematics as isolated formulas.

For every equation, ask:

> **Where does this appear inside a neural network?**

For example:

$$
Wx+b
$$

means:

```text
Matrix multiplication
        ↓
Linear layer
        ↓
Neural network
```

And:

$$
\nabla_\theta L
$$

means:

```text
Derivative
    ↓
Gradient
    ↓
Backpropagation
    ↓
Parameter update
```

And:

$$
\theta_{t+1}
=
\theta_t-\eta\nabla_\theta L
$$

means:

```text
Gradient
   ↓
Learning rate
   ↓
Parameter update
   ↓
Model training
```

The objective is not to become a mathematician.

The objective is to become capable of looking at a Deep Learning equation and understanding **what it means, where it comes from, and what it does during training**.

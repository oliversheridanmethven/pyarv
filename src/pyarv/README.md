# PyARV

Approximate random variables in Python. 

## Description

The goal of this project is to make available to the Python community
a new source of high speed random numbers. These random numbers should
be considerably faster than those which come out-of-the-box from the
likes of NumPy, SciPy, etc., by an order of magnitude (or more).

### How is this achieved?

The source of the improved speed comes from using approximations
in a popular algorithm for producing random variables, and thus
produces _approximate random variables_. The algorithm
is the inverse transform method, which can generate random variables
from any distribution of interest (and thus is generalisable), and the
approximation is utilising piecewise polynomials to approximate inverse
cumulative distribution functions. These approximations are mathematically
simple and extremely well suited to SIMD execution, so run incredibly fast
on the latest CPUs and GPUs which have vectorised hardware. The exact
details are spelled out in the ACM TOMS article:

>Michael B. Giles and Oliver Sheridan-Methven.
> _Approximating
inverse cumulative distribution functions to produce
approximate random variables._
> ACM Transactions
on Mathematical Software, 49(3), Article 26, September 2023, 29 pages.
> [https://doi.org/10.1145/3604935](https://doi.org/10.1145/3604935)

### Are approximate random numbers appropriate for my application?

It is important to note, that the approximate random variables produced
for a given distribution, are as their name suggests, **approximate**. 
For users who believe they need numbers exactly following the desired
distribution, the random numbers produced using this package 
will not meet this need. Instead, what they will give you are random 
numbers which follow a distribution with very similar statistics. 
How closely the approximate distribution matches the exact one
depends on the quality of the approximation, and the statistics we
use to measure the difference between the two. The quality of the 
approximation can be controlled to strike a good balance between
extremely fast approximations which are very low fidelity, or 
alternatively higher fidelity approximations which are a little
slower. 

For users asking themselves: "will I still get the right
answer if I use these", the answer is "yes", if you use them 
properly. The authors can't anticipate every use case, but
if you are using these in Monte Carlo simulations, then the authors 
have provided a rigorous mathematical analysis detailing how
to use them correctly in Monte Carlo applications:

> Michael B. Giles and Oliver Sheridan-Methven. 
> _Analysis of nested multilevel Monte Carlo using approximate
normal random variables_. 
> SIAM/ASA Journal on Uncertainty
Quantification, 10(1):200–226, 2022.
> [https://doi.org/10.1137/21M1399385](https://doi.org/10.1137/21M1399385)

### Should you chase this level or performance or not?

Lots of applications use random numbers, and many of these need
lots of random numbers very quickly, such as e.g. Monte Carlo
simulations in finance. For anyone who faces this problem and
needs to use Python, this package is for you.

Many critics will say that using Python and worrying about
performance is an oxymoron, and if you want high performance
code that needs random numbers quickly, you should switch to a
lower level compiled language such as e.g. C, C++, Fortran, Rust, etc.
Somewhat ironically this is an opinion the authors of this package share.
Nonetheless, the routines offered in this package use low
level implementations under the hood, so there should not be
any major loss in performance from switching to Python.
That being said, the authors can't anticipate the exact hardware
being run on, the compilers available, etc. For the ultimate in
performance, don't use Python, but instead use (or take
inspiration from) our underlying implementations in the underlying
ARV package. 
# The PyARV package

**_The fastest random numbers in Python._** 

## What is the package's mission?

The PyARV package was designed to:

1. Provide a clean C library implementation of 
approximate random variables, to either be used directly 
in C applications, indirectly through wrappers and bindings, 
or purely as a reference implementation for porting to 
other languages.
2. Make most of this performance accessible for 
Python applications. 

## Should I be using the package?

You **should** be using the package if:

- You have an optimised application spending most of its time generating random numbers.
- You are a scientist or a researcher investigating applications which make extensive
use of random numbers. 

You probably **should not** be using the package if:

- You are spending a negligible amount of time generating random numbers. 
- You require cryptographically secure random numbers. 
- You don't care about performance and haven't optimised your application already. 

## Where does the package originate from?

Generating random numbers from desired distributions
can be a bottleneck in many applications (e.g. in Monte Carlo 
simulations). Approximate random variables (the "ARV" in PyARV) 
were developed and analysed by Prof. Mike Giles and Dr Oliver 
Sheridan-Methven as a method of alleviating this problem,
offering considerable speed improvements. (They further showed that using
nested multilevel Monte Carlo can allow full accuracy to be all-the-while maintained).
These were implemented in C by Dr Oliver Sheridan-Methven with the 
assistance of Prof. Mike Giles and the Performance Libraries team of Arm
led by Dr Christopher Goodyer. 

The code was originally developed as part of Dr Oliver Sheridan-Methven's 
doctoral thesis. After a few iterations of cleaning the code for various 
publications and hand-overs, the code has been pruned and improved, and has
transitioned from a proof-of-concept, messy, and "rough around the edges" codebase, 
into a much cleaner, better documented, better tested code base. There is still 
much room for improvement, but the code has certainly come a long way. 

The repository preceeding PyArv was: [`approximate_random_variables`](https://github.com/oliversheridanmethven/approximate_random_variables),
and this contains various "how-to" and "getting started" guides, alongside 
implementations specialised for Intel AVX-512, Nvidia, and Arm ThunderX2 VLA SVE hardwares.
The core C libraries in PyARV are lifted from this repo. 

## Why does the package exist?

This package is a selection of cherry-picked implementations which are suitable for 
a broad range of hardwares. (There are other hardware specific specialisations which 
are even faster, targetting Arm and Intel, which are not included in this package).

The underlying C implementations are the core of the package, and arguably the most valuable
part of the package. However, to promote the use of approximate random numbers more
broadly, 
we created this package to expose the core functionality to the Python community.
While this won't of course make Python a HPC language, it will bring considerable
HPC capabilities into the reach of Python applications, at least in so far as random
number generation is concerned. 



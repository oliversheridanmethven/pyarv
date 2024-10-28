# Upcoming features

Here we sketch out some of the milestones we would like to achieve
and a collection of our higher level ambitions from this project.

## Motivation 

This project is currently constructed primarily as a learning exercise. While we
hope it picks up traction and becomes wildly popular, that is not our
primary goal. Our aim is to use this as a development testbed for
documentation generation, repository hosting, testing frameworks,
PyPI deployment, build system integration with `pip` and CMake
(or e.g. Bazel), type hinting, sanitisers, etc.   

## Versions

### Release 0

#### 0.1

- A first working draft of a Python interface that can be installed using 
pip from PyPI or from a git clone of the repo.  
- The Gaussian distribution:
  - Linear and cubic approximations for appropriate pre-specified table sizes. 

#### 0.2

- The non-central \( \chi^2 \) distribution. 
- Better documentation including a user guide, developer guide, examples, etc. 

#### 0.3

- Other distributions such as: Poisson, Beta, \( \chi^2 \), etc. 

### Release 1

Once these checkpoint have been achieved we will be ready to 
release this as version 1.0. 

### Release 2

For version 2.0, we want to split out the C implementation into its 
own repo and have it as a submodule (or similar) in this repo. 
Thereafter, the two repos will be two separate projects. 

# Documentation conventions

The documentation is split into two halves, aimed at users
and developers separately. We currently only document the 
Python interface, and the underlying C interface is not 
documented (we hope to change this should the mkdocstring
C handler be open sourced). 

Following the defaults used by mkdocstring-Python, 
the additional convention we follow is that _everything_ is assumed 
to be user facing unless it matches any of the following 
conditions:

* A parent directory contains a leading or training underscore, e.g. 
`_foo/`, `foo_/`, or `_foo_/`.
* A parent directory is called `tests` or `demos`. 
* A file contains a leading or trailing undercore (except `__init__.py`). 

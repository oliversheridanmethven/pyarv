# Binding

A simple hello world example showing how to bind our C code 
for use in Python. 
 

## Conventions

For simplicity (and ease of consistency) we will adopt the following
principles when constructing our C libraries and possible Python bindings. 

### Error checking and handling

We will have a mix of C and Python libraries, designed to
interoperate and also still function independently. Consequently, 
there is a grey area in the responsibility for error handling,
with the following possibilities:

- Have the C libraries try to check and recover from errors.
(Made tricky by the limited error handling in C). 

- Have the C libraries make no attempt at error handling.

- Have the C extension wrappers try and check 
the Python input after it has been passed to the C library.

- Have the Python library check all the input before it gets
passed to the C library. 

- No error checking anywhere.

- etc.

Clearly there are various options. 

### Type errors

For type checking we will largely rely on the type system of
C and the compilation rules. For Python, where necessary, 
we will rely on the Python code to assert the validity of
any data types before they are passed to the C libraries.
For the boundary of these two languages, we will rely on the 
error handling of the python argument and keyword parser.

There is a grey area in between where one language must 
respect or acknowledge the rules of the other, such as 
a Python integer being represented as a `long`, but the 
corresponding Python function expecting an `int`, where
these two might be of different sizes. Here, we will largely
put the emphasis on Python to ensure C is happy, as this 
is typically easier to code and enforce.  

### Runtime errors

Sometimes codes fail for internal problems. An example
might be a quadratic solver which finds the real roots 
of polynomials and presented with one with complex roots. 
Another might be a request for more memory by `malloc` 
failing, not finding a file, a write to some output 
failing (`printf` can fail), or a matrix inversion
not being possible. In most of these cases, we generally
want to avoid the responsibility of error handling, and 
we adopt the mentality "if something untoward has happened, 
then let the program crash and fail in a fast and loud 
way..." (we may try and also do this gracefully 
where possible). In C this will generally mean an immediate
call to `exit`.

## Splitting into two libraries

We follow the convention that for some library, we split this into
its core functionality, which contains all the core C functionality
and no Python, and a second library which only implements the Python
interface. 

### Failing to link to Python C extension libraries on Mac OSX

One of the reasons we split the functionality into two libraries,
(aside from a more modularity), is because the Python build 
proceedure typically produces bundles, whereas our 
C libraries are typically producing dynamic libraries. This can 
lead to linking errors. 

### Putting module libraries in their own directory

To keep a nice modular structure where the C extensions match 
the style of Python modules, put any C extensions in their own
subdirectory which would be the equivalent of a single python file.
# Dependencies

## List of dependencies

The project relies on a few dependencies, the most notable of
which include:

### The source code

- GCC (with C23 and C++23 support).
- CMake (for building).
- GLIB (for logging).
- Argp (for the CLI).
- Criterion (for testing).

### Python bindings

- scikit-build (for Python bindings).

### The documentation

- mkdocs (for documentation).
    - mkdocs-same-dir.
    - mkdocs-awesome-pages-plugin.
    - mkdocs-exclude
    - pillow cairosvg
    - mkdocs-git-revision-date-localized-plugin
    - mkdocs-git-committers-plugin-2
- github command line tools `gh`

## Installation

Most of these can be installed either through `pip` or `brew`
or similar. 

### Virtual environments

In our setup, we use a Python-3.11 based virtual environment
based in the root directory of the project. 

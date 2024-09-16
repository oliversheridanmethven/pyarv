from skbuild import setup
from setuptools import find_packages

packages_to_expose = ['pyarv']
# ^ These typically contain the C modules
package_dirs = {package: f"src/{package.replace('.', '/')}" for package in find_packages("src/") if any([package_base in package for package_base in packages_to_expose])}

setup(
    name="PyARV",
    description="Approximate random variables",
    author="Dr Oliver Sheridan-Methven",
    license="MIT",
    setuptools_git_versioning={"enabled": True},
    setup_requires=["setuptools-git-versioning<2"],
    install_requires=["numpy"],
    tests_require=["varname"],
    extras_require={
        'dev': [
            "mkdocs",
            "mkdocs-material",
            "mkdocs-exclude",
            "mkdocs-same-dir",
            "mkdocs-awesome-pages-plugin",
            "mkdocs-git-revision-date-localized-plugin",
            "mkdocs-material[imaging]",
            "mkdocstrings",
            "mkdocstrings-python",
            "mkdocs-literate-nav",
            "mkdocs-section-index",
            "mkdocs-gen-files",
            "python-markdown-math",
            "mike",
            "matplotlib",
            "pandas",
            "scipy",
            "build"
        ]
    },
    packages=package_dirs.keys(),
    package_dir=package_dirs,
    python_requires=">=3.10",
)

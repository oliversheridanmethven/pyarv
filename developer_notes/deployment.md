# Deployment

To upload this package to PyPI, we use
`twine` and follow the instructions detailed
in online tutorials such as e.g.:  

* [https://www.freecodecamp.org/news/how-to-create-and-upload-your-first-python-package-to-pypi/](https://www.freecodecamp.org/news/how-to-create-and-upload-your-first-python-package-to-pypi/)     
* [https://packaging.python.org/en/latest/tutorials/packaging-projects/#creating-readme-md](https://packaging.python.org/en/latest/tutorials/packaging-projects/#creating-readme-md)  

Mostly it is a combination of 
```bash
python3 -m build 
```
and then
```bash
python3 -m twine upload --repository testpypi dist/*
```
and
```bash
python3 -m twine upload dist/*
```
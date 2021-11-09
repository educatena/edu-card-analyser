# Educatena Card Analyser Library

## About
Contains computer vision models to extract and manipulate answer sheet card information.

## Models

* **SheetV1**
    * Sheet first vision as described/inputed on the initial meetings of the development of the Card Analyser using an example answer card layout found on the internet.

## Debugging
There is a file called "tester.py" on the src folder, you can run that whilst the working directory points to the repository root folder.

And do not forget to Configure your IDE to use the python that pipfile builds up for you upon using pipenv install whilst on the repository root folder.

## Compiling the Library and Testing
On the virtual environment:

    pipenv shell
    python setup.py bdist_wheel

Then on your target machine get whatever is inside the dist folder, like 'edu_card_analyser-0.0.1-py3-none-any.whl' and run.

    pip install 'edu_card_analyser-0.0.1-py3-none-any.whl'

After that, the python environment in which you install that whl file will have access to a package called "edu_card_models", on which you can import the related modules and classes like

    from edu_card_models.SheetV1 import SheetV1

## Get Started Quickly
    pipenv install
    pipenv shell
    python src/tester.py

## Upload the build to pypi
    python3 -m twine upload dist/*
twine check dist/*
twine upload --repository-url https://upload.pypi.org/legacy/ dist/*

twine upload dist/*



python setup.py build
python setup.py bdist_wheel
python setup.py sdist
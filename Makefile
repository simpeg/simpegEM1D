.PHONY: build tests clean deploy

build:
    python setup.py build_ext --inplace

tests:
    nosetests --logging-level=INFO

clean:
    find . -name "*.pyc" | xargs -I {} rm -v "{}"

deploy:
    python setup.py sdist bdist_wheel upload


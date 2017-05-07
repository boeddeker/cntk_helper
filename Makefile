.PHONY: clean lint install

################################################################################
# COMMANDS                                                                     #
################################################################################

install:
	pip install --user -e .

clean:
	find . -name "*.pyc" -exec rm {} \;

lint:
	flake8 --ignore=F841 --exclude=lib/,bin/,docs/conf.py .


SHELL := /bin/bash

MODE ?= full   # full | lora

.PHONY: all setup run clean

all: setup run

setup:
	module purge && \
	module load JupyterLab/3.5.0-GCCcore-11.3.0 && \
	source /fp/projects01/ec517/venvs/in5310/bin/activate && \
	pip install -r requirements.txt --user

run:
	module purge && \
	module load JupyterLab/3.5.0-GCCcore-11.3.0 && \
	source /fp/projects01/ec517/venvs/in5310/bin/activate && \
	python src/train.py --mode $(MODE)

clean:
	rm -f output/*.pt

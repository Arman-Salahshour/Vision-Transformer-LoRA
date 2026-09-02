SHELL := /bin/bash

MODE ?= full   # full | lora
DATA_URL = https://s3.amazonaws.com/fast-ai-imageclas/imagewoof2-320.tgz
DATA_DIR = data/imagewoof2

.PHONY: all setup data run clean

all: setup data run

setup:
	module purge && \
	module load JupyterLab/4.2.5-GCCcore-13.3.0 && \
	source /projects/ec674/venvs/in5310h26/bin/activate && \
	pip install -r requirements.txt --user

data:
	@if [ ! -d "$(DATA_DIR)/train" ]; then \
		mkdir -p data; \
		curl -L $(DATA_URL) -o data/imagewoof2-320.tgz; \
		tar -xzf data/imagewoof2-320.tgz -C data; \
		mv data/imagewoof2-320 $(DATA_DIR); \
		rm data/imagewoof2-320.tgz; \
	fi

run:
	module purge && \
	module load JupyterLab/4.2.5-GCCcore-13.3.0 && \
	source /projects/ec674/venvs/in5310h26/bin/activate && \
	python src/train.py --mode $(MODE)

clean:
	rm -f output/*.pt output/*_results.json
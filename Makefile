.PHONY: all test clean situations emotions

all: train

clean: clean-binary clean-dataset clean-emotions clean-situations clean-models

build:
	python build_dataset.py

emotions:
	python emotions.py

situations:
	python situations.py

train: build
	./train.sh fast

###############################################################################
# Detailed cleanup
###############################################################################

clean-binary:
	@find . -name "*.pyc tmp.prototxt" -delete
	@find . -name "tmp.prototxt" -delete

clean-dataset:
	@find data/Ria_Gurtow/jpg/ -maxdepth 1 -name "*.jpg" -delete
	@rm -rf data/Ria_Gurtow/jpg
	@rm -f data/Ria_Gurtow/train.txt data/Ria_Gurtow/test.txt

clean-emotions:
	@find emotions/data/frames/ -maxdepth 1 -name "*.jpg" -delete

clean-situations:
	@find situations/data/frames/ -maxdepth 1 -name "*.jpg" -delete

clean-models:
	@find models/Ria_Gurtow/generations/ -maxdepth 1 -name "*.caffemodel" -delete
	@find models/Ria_Gurtow/generations/ -maxdepth 1 -name "*.solverstate" -delete

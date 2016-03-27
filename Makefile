.PHONY: all test clean situations emotions

all: clean dataset train

clean: clean-binary clean-dataset clean-emotions clean-situations clean-models

dataset:
	@echo "Creating dataset for Ria Gurtow"
	@python build_dataset.py

emotions:
	@echo "Rendering Ria Gurtow emotions"
	@python emotions.py

situations:
	@echo "Rendering Ria Gurtow situations"
	@python situations.py

train:
	@echo "Training Ria Gurtow situations model"
	@./train.sh fast

###############################################################################
# Detailed cleanup
###############################################################################

clean-binary:
	@echo "Cleaning binaty temporary files"
	@find . -name "*.pyc tmp.prototxt" -delete
	@find . -name "tmp.prototxt" -delete

clean-dataset:
	@echo "Cleaning dataset"
	@find data/Ria_Gurtow/jpg/ -maxdepth 1 -name "*.jpg" -delete
	@rm -f data/Ria_Gurtow/train.txt data/Ria_Gurtow/test.txt
	@rm -rf data/Ria_Gurtow/*.lmdb/ @rm -rf data/Ria_Gurtow/*.leveldb/
	@rm -rf models/Ria_Gurtow/*.binaryproto

clean-emotions:
	@echo "Cleaning emotions model frames"
	@find emotions/data/frames/ -maxdepth 1 -name "*.jpg" -delete

clean-situations:
	@echo "Cleaning situations model frames"
	@find situations/data/frames/ -maxdepth 1 -name "*.jpg" -delete

clean-models:
	@echo "Cleaning situations model generations"
	@find models/Ria_Gurtow/generations/ -maxdepth 1 -name "*.caffemodel" -delete
	@find models/Ria_Gurtow/generations/ -maxdepth 1 -name "*.solverstate" -delete

.PHONY: all test clean situations emotions

MP4_FRAMERATE:=-framerate 30
MP4_ENCODING:=-c:v libx264 -vf fps=30 -pix_fmt yuv420p

all: clean dataset train

clean: clean-binary clean-dataset clean-emotions clean-situations clean-models clean-solvers

dataset:
	@echo "Creating dataset for Ria Gurtow"
	@python build_dataset.py

emotions:
	@echo "Rendering Ria Gurtow emotions"
	@python emotions.py

situations:
	@echo "Rendering Ria Gurtow situations"
	@python situations.py \
		--image images/athena_louvre_700px.jpg \
		--mask images/athena_louvre_700px_face_mask.png \
		--stages conv4 conv3 conv5 \
		--resize_in 256 256 \
		--resize_out 700 700 \
		--dest situations/data/frames \
		--verbose \
		--start_from 0 \
		# --max_gen 10 \

solvers:
	@echo "Render Ria Gurtow solvers"
	@python render_solvers.py

train: solvers
	@echo "Training Ria Gurtow situations model"
	@./train.sh fast

gif:
	@echo "Create gif for learning progress"
	@convert -delay 0.05 -loop 0 situations/data/frames/gen-*.jpg situations/data/progress.gif

mp4:
	@echo "Create .mp4 movie for learning progress"
	@ffmpeg $(MP4_FRAMERATE) -pattern_type glob -i 'situations/data/frames/gen-*.jpg' $(MP4_ENCODING) situations/data/progress.mp4

sepia:
	@echo "Create sepia for learning progress frames"
	@python convert_to_sepia.py --files situations/data/frames/'gen-*.jpg' --dest situations/data/frames/

sepia-gif:
	@echo "Create gif for learning progress in sepia"
	@convert -delay 0.05 -loop 0 situations/data/frames/sepia-gen-*.jpg situations/data/sepia-progress.gif

sepia-mp4:
	@echo "Create .mp4 movie for learning progress in sepia"
	@ffmpeg $(MP4_FRAMERATE) -pattern_type glob -i situations/data/frames/'sepia-gen-*.jpg' $(MP4_ENCODING) situations/data/sepia-progress.mp4

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
	@rm -f situations/data/*.gif
	@rm -f situations/data/*.mp4

clean-models:
	@echo "Cleaning situations model generations"
	@find models/Ria_Gurtow/generations/ -maxdepth 1 -name "*.caffemodel" -delete
	@find models/Ria_Gurtow/generations/ -maxdepth 1 -name "*.solverstate" -delete

clean-solvers:
	@echo "Cleaning rendered solver settings"
	@rm -f models/Ria_Gurtow/solver.prototxt
	@rm -f models/Ria_Gurtow/*solver.prototxt

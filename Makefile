.PHONY: all test clean situations emotions

MP4_FRAMERATE:=-framerate 15
MP4_ENCODING:=-c:v libx264 -vf fps=30 -pix_fmt yuv420p

MP4_SLOW_FRAMERATE:=-framerate 5
MP4_SLOW_ENCODING:=-c:v libx264 -vf fps=5 -pix_fmt yuv420p

all: clean dataset train

clean: clean-binary clean-dataset clean-emotions clean-situations clean-models clean-solvers

dataset:
	@echo "Creating dataset for Ria Gurtow"
	@python build_dataset.py --subdirs

emotions:
	@echo "Rendering Ria Gurtow emotions"
	@python emotions.py

render-lisa:
	@echo "Rendering Ria Gurtow situations for Lisa Gotfrik"
	@python situations.py \
		--image images/base/lisa-08.jpg \
		--mask images/base/lisa-black-mask.png \
		--stages conv4 conv3 conv5 \
		--resize_in 0 256 \
		--resize_out 0 1080 \
		--dest frames/lisa \
		--verbose \
		--start_from 0 \
		--save_all \
		--num_rendered 5 \
		--max_gen 3500 \
		# --test \

render-unknown:
	@echo "Rendering Ria Gurtow situations for Jane Dow"
	@python situations.py \
		--image images/base/unknown-06.jpg \
		--mask images/base/unknown-black-mask.png \
		--stages conv4 conv3 conv5 \
		--resize_in 0 256 \
		--resize_out 0 1080 \
		--dest frames/unknown \
		--verbose \
		--start_from 0 \
		--save_all \
		--num_rendered 5\
		--max_gen 0 \

render-athena:
	@echo "Rendering Ria Gurtow situations for Athena"
	@python situations.py \
		--image images/athena_louvre_700px.jpg \
		--mask images/athena_louvre_700px_face_mask.png \
		--stages conv4 conv3 conv5 \
		--resize_in 256 256 \
		--resize_out 700 700 \
		--dest situations/data/frames \
		--verbose \
		--start_from 0 \
		# --max_gen 3500 \

solvers: clean-solvers
	@echo "Render Ria Gurtow solvers"
	@python render_solvers.py

train: solvers
	@echo "Training Ria Gurtow situations model"
	@./train.py

vizualize:
	@draw_net.py --rankdir TB models/Ria_Gurtow/deploy.prototxt images/Ria_Gurtow_model.png

###############################################################################
# Media tools
###############################################################################

gif:
	@echo "Create gif for learning progress"
	@convert -delay 0.05 -loop 0 situations/data/frames/gen-*.jpg situations/data/progress.gif

mp4:
	@echo "Create .mp4 movie for learning progress"
	@ffmpeg -y $(MP4_FRAMERATE) -pattern_type glob -i 'situations/data/frames/gen-*.jpg' $(MP4_SLOW_ENCODING) situations/data/progress.mp4

mp4-lisa:
	@echo "Create .mp4 movie for learning progress and Lisa Gotfrik"
	@ffmpeg -y $(MP4_SLOW_FRAMERATE) -pattern_type glob -i 'frames/lisa/verbose-gen-*.jpg' $(MP4_SLOW_ENCODING) frames/lisa.mp4

mp4-unknown:
	@echo "Create .mp4 movie for learning progress and Jane Dow"
	@ffmpeg -y $(MP4_SLOW_FRAMERATE) -pattern_type glob -i 'frames/unknown/verbose-gen-*.jpg' $(MP4_ENCODING) frames/unknown.mp4

mask:
	@echo "Create masked versions for progress frames"
	@python apply_mask.py --files situations/data/frames/'gen-*.jpg' --mask images/athena_louvre_1024px_face_mask.png --dest situations/data/frames/ --img_pos 162 0

mask-gif: mask
	@echo "Create gif for learning progress with mask"
	@convert -delay 0.05 -loop 0 situations/data/frames/mask-gen-*.jpg situations/data/mask-progress.gif

mask-mp4: mask
	@echo "Create .mp4 movie for learning progress with mask"
	@ffmpeg -y $(MP4_FRAMERATE) -pattern_type glob -i situations/data/frames/'mask-gen-*.jpg' $(MP4_ENCODING) situations/data/mask-progress.mp4

sepia:
	@echo "Create sepia for learning progress frames"
	@python convert_to_sepia.py --files situations/data/frames/'gen-*.jpg' --dest situations/data/frames/

sepia-gif: sepia
	@echo "Create gif for learning progress in sepia"
	@convert -delay 0.05 -loop 0 situations/data/frames/sepia-gen-*.jpg situations/data/sepia-progress.gif

sepia-mp4: sepia
	@echo "Create .mp4 movie for learning progress in sepia"
	@ffmpeg -y $(MP4_FRAMERATE) -pattern_type glob -i situations/data/frames/'sepia-gen-*.jpg' $(MP4_ENCODING) situations/data/sepia-progress.mp4

###############################################################################
# Detailed cleanup
###############################################################################

clean-binary:
	@echo "Cleaning binaty temporary files"
	@find . -name "*.pyc tmp.prototxt" -delete
	@find . -name "tmp.prototxt" -delete

clean-dataset:
	@echo "Cleaning dataset"
	@rm -rf data/Ria_Gurtow/situations/
	@rm -f data/Ria_Gurtow/*.txt
	@rm -rf data/Ria_Gurtow/*.lmdb/ @rm -rf data/Ria_Gurtow/*.leveldb/
	@rm -rf models/Ria_Gurtow/*.binaryproto

clean-emotions:
	@echo "Cleaning emotions model frames"
	@find emotions/data/frames/ -maxdepth 1 -name "*.jpg" -delete

clean-sepia:
	@find situations/data/frames/ -maxdepth 1 -name "sepia-*.jpg" -delete

clean-mask:
	@find situations/data/frames/ -maxdepth 1 -name "mask-*.jpg" -delete

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

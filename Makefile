all: run

run:
	dreamer.py

clean:
	@rm -f *.pyc tmp.prototxt
	@rm -rf emotions/data/frames/*.jpg
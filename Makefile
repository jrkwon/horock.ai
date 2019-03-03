.PHONY: all clean clean-all
.PHONY: git-clone setup-condaenv remove-condaenv setup
.PHONY: extract-taeri
.PHONY: step1 step2

ANACONDAURL=https://repo.anaconda.com/archive/Anaconda3-2018.12-Linux-x86_64.sh
ANACONDAINST=download/anaconda-install.sh

FPS=30
DATA_LOC=datasets

all: help

help:
	@echo "Usage:"
	@echo "    make install    : install anaconda"
	@echo "    make data       : extract data for training"
	@echo "    make train      : train data"
	@echo "    make test       : evaluate model"

## Install Toolchains

install: setup-condaenv

setup-condaenv: stamps/condainst stamps/condaenv

remove-condaenv:
	conda env remove -n horock
	rm -f stamps/conda-env

stamps/condaenv: environment.yml
	conda env create -f environment.yml || conda env update -f environment.yml
	touch $@
	@echo ""
	@echo "Now you can use conda. with"
	@echo ""
	@echo " . conda-horock"
	@echo ""

stamps/condainst: $(ANACONDAINST)
	@if which conda; then \
		echo 'Already conda installed' \
	else \
		 bash +x $(ANACONDAINST) && exit 1; \
	fi
	touch $@

$(ANACONDAINST):
	mkdir -p `dirname $(ANACONDAINST)`
	@if which curl 2>/dev/null >&2; then exit 0; else echo "You need curl, (e.g. ubuntu, sudo apt install curl)" && exit 1; fi
	curl $(ANACONDAURL) > $@

## Setup Data

.PHONY: data

data:
	make extract AB=A NAME=taeri
	make extract AB=B NAME=iu

extract: setup-condaenv stamps/extract-$(NAME)

stamps/extract-$(NAME): setup-condaenv stamps/splitscenes-$(NAME)
	touch $@
	
astamps/extract: stamps/crop-$(NAME)
	python faceswap/faceswap.py extract -i $(DATA_LOC)/data$(AB)/$(NAME) -o $(DATA_LOC)/data$(AB)/$(NAME)/extracted -D mtcnn -r 45 -ae -mp

stamps/splitscenes-$(NAME): stamps/scenes-$(NAME)
	bash ./scripts/splitscenes.sh $(DATA_LOC)/$(NAME) stamps/scenes-$(NAME)
	touch $@

stamps/scenes-$(NAME): stamps/genimages-$(NAME)
	ffprobe -hide_banner -show_frames -of compact=p=0 -f lavfi 'movie=$(DATA_LOC)/$(NAME)/%05d.png,select=gt(scene\,.2)' | tee $@

stamps/genimages-$(NAME): $(DATA_LOC)/$(NAME).mp4
	mkdir -p $(DATA_LOC)/$(NAME)
	ffmpeg -hide_banner -i $(DATA_LOC)/$(NAME).mp4 -vf fps=$(FPS) $(DATA_LOC)/$(NAME)/%05d.png
	touch $@

stamps/crop-$(NAME): stamps/genimages-$(NAME)
	mkdir -p $(DATA_LOC)/data$(AB)/$(NAME)-crop-resize
	mogrify -path $(DATA_LOC)/data$(AB)/$(NAME)-crop-resize \
		-crop $(CROPAREA) \
		-resize 256x256^ \
		$(DATA_LOC)/data$(AB)/$(NAME)/*.png
	touch $@

$(DATA_LOC)/$(NAME).mp4:
	@if test ! -f $(DATA_LOC)/$(NAME).mp4; then echo "\nYou need $(DATA_LOC)/$(NAME).mp4\n"; exit 1; fi

## Cleaning

clean:
	rm -rf stamps/*

clean-all: clean
	rm -rf faceswap face2face

clean-data:
	rm -f stamps/splitscenes-* stamps/genimages-*
	rm -rf datasets/dataA/* datasets/dataB/*

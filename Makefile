.PHONY: all clean clean-all
.PHONY: git-clone setup-condaenv remove-condaenv setup
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

install: setup-ffmpeg-cmake setup-condaenv

setup-ffmpeg-cmake:
	@if ! which ffmpeg 2>/dev/null >&2; then \
		sudo apt install ffmpeg; \ 
	fi 
	@if ! which cmake 2>/dev/null >&2; then \
		sudo apt install cmake; \
	fi

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
		echo 'Already conda installed'; \
	else \
		 bash +x $(ANACONDAINST) && exit 1; \
	fi
	touch $@

$(ANACONDAINST):
	mkdir -p `dirname $(ANACONDAINST)`
	@if which curl 2>/dev/null >&2; then exit 0; else echo "You need curl, (e.g. ubuntu, sudo apt install curl)" && exit 1; fi
	curl $(ANACONDAURL) > $@

## Setup Data

.PHONY: data extract

data:
	make extract NAME=jaein
	make extract NAME=muhyeon

extract: stamps/arrange-$(NAME)

stamps/arrange-$(NAME): stamps/extract-$(NAME) ./scripts/arrange_training_set.sh
	bash ./scripts/arrange_training_set.sh $(NAME) $(DATA_LOC) 1000
	touch $@

stamps/extract-$(NAME): stamps/facetag-$(NAME)
	bash ./scripts/facecrop.sh $(DATA_LOC)/$(NAME)-scenes $(DATA_LOC)/$(NAME)-faces stamps/facetag-$(NAME)
	touch $@
	
stamps/facetag-$(NAME): stamps/splitscenes-$(NAME) scripts/facetag.sh
	bash ./scripts/facetag.sh $(NAME) $(DATA_LOC) $@

stamps/splitscenes-$(NAME): stamps/scenes-$(NAME)
	bash ./scripts/splitscenes.sh $(DATA_LOC)/$(NAME) $(DATA_LOC)/$(NAME)-scenes stamps/scenes-$(NAME)
	touch $@

stamps/scenes-$(NAME): stamps/genimages-$(NAME)
	ffprobe -hide_banner -show_frames -of compact=p=0 -f lavfi 'movie=$(DATA_LOC)/$(NAME)/%05d.png,select=gt(scene\,.2)' | tee $@

stamps/genimages-$(NAME): $(DATA_LOC)/$(NAME).mp4
	mkdir -p $(DATA_LOC)/$(NAME)
	ffmpeg -hide_banner -i $(DATA_LOC)/$(NAME).mp4 -vf fps=$(FPS) $(DATA_LOC)/$(NAME)/%05d.png
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
	rm -rf datasets/*-scenes

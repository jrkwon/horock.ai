.PHONY: all clean clean-all
.PHONY: git-clone setup-condaenv remove-condaenv setup
.PHONY: step1 step2

ANACONDAURL=https://repo.anaconda.com/archive/Anaconda3-2018.12-Linux-x86_64.sh
ANACONDAINST=download/anaconda-install.sh

FPS=30
VISDOM_PORT=8097
GPU_ID=1,2
TRAINING_SIZE=9000
TEST_SIZE=1000
DATA_LOC=datasets

-include local.mk

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
	make recycle-gan-data A=jaein B=muhyeon
	#make recycle-gan-data B=jaein A=muhyeon

## SINGLE faces
extract: stamps/arrange-$(NAME)

stamps/arrange-$(NAME): stamps/extract-$(NAME) ./scripts/arrange_training_set.sh
	bash ./scripts/arrange_training_set.sh $(NAME) $(DATA_LOC) $(TRAINING_SIZE) $(TEST_SIZE)
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

## RECYCLE datasets
recycle-gan-data: stamps/recycle-gan-data-$(A)-$(B)

stamps/recycle-gan-data-$(A)-$(B): stamps/arrange-$(A) stamps/arrange-$(B)
	bash ./scripts/recycle-gan-data.sh $(A) $(B) $(DATA_LOC)
	touch $@

## Training

.PHONY: train recycle-gan test

train:
	make recycle-gan A=muhyeon B=jaein
	#make recycle-gan B=muhyeon A=jaein

recycle-gan: stamps/recycle-gan-data-$(A)-$(B)
	bash scripts/train_recycle_gan.sh $(A) $(B) $(GPU_ID)

mon:
	@python -m visdom.server -port $(VISDOM_PORT) & echo $$! > .visdom.pid
	@echo "Visdom pid is `cat .visdom.pid`"
	@echo "You can kill it with 'make kmon'"

kmon:
	@kill `cat .visdom.pid` 2>/dev/null && echo "Killed visdom.." && rm -f .visdom.pid

## Cleaning

clean:
	rm -rf stamps/*

clean-all: clean
	find $(DATA_LOC) -maxdepth 1 -type -d -a -name '*-*' -a ! -name '*-known-*' -print0 | xargs -0 rm -rf

clean-data:
	rm -f stamps/splitscenes-* stamps/genimages-* stamps/scenes-* stamps/extract-* stamps/arrange-*

clean-train:
	rm -f stamps/recycle-gan-*

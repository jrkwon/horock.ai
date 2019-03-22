.PHONY: all clean clean-all
.PHONY: git-clone setup-condaenv remove-condaenv check-condaenv setup
.PHONY: step1 step2

## We set ".SECONDEXPANSION:" for expanding '$$*' in prerequsite condition to reuse '%' part string of target.
## Expanding '$*' in recipes is always working w/ w/o .SECONDEXPANSION
## Check with make -rnd <target> for clarity
.SECONDEXPANSION:

## Keep intermediate stamps for fast restarting
## We need .SECONDARY directive
.SECONDARY:

ANACONDAURL=https://repo.anaconda.com/archive/Anaconda3-2018.12-Linux-x86_64.sh
ANACONDAINST=download/anaconda-install.sh

A=muhyeon
B=jaein
FPS=30
VISDOM_PORT=8097
GPU_ID=0
TRAINING_SIZE=9000
TEST_SIZE=1000
DATA_LOC=datasets
EPOCH=latest

-include local.mk

all: help

help:
	@echo "Usage:"
	@echo "    make install    : install anaconda"
	@echo "    make data       : extract data for training"
	@echo "    make mon/kmon   : run visdom / kill visdom (you can change port with local.mk)"
	@echo "    make train      : train data"
	@echo "    make test       : evaluate model"
	@echo "    make video      : make videos from test"
	@echo ""
	@echo "    make -rnd <target> will show you the dependency chain"

## Install Toolchains

install: setup-condaenv

setup-condaenv: stamps/condainst stamps/condaenv

remove-condaenv:
	conda env remove -n horock
	rm -f stamps/conda-env

check-condaenv:
	@if test "$${CONDA_DEFAULT_ENV}" != "horock"; then echo "\nYou need '. conda-horock' to run\n"; exit 1; fi

stamps/condaenv: environment.yml
	conda env create -f environment.yml || conda env update -f environment.yml
	touch $@
	@echo ""
	@echo "Now you can use conda. with"
	@echo ""
	@echo " . conda-horock"
	@echo ""

stamps/condainst: $(ANACONDAINST)
	-@mkdir stamps
	@if which conda; then \
		echo 'Already conda installed'; \
	else \
		 bash +x $(ANACONDAINST) && exit 1; \
	fi
	touch $@

$(ANACONDAINST):
	-@mkdir stamps
	-@mkdir -p `dirname $(ANACONDAINST)`
	@if which curl 2>/dev/null >&2; then exit 0; else echo "You need curl, (e.g. ubuntu, sudo apt install curl)" && exit 1; fi
	curl $(ANACONDAURL) > $@

## Setup Data

.PHONY: data extract

data: recycle-gan-data 

## SINGLE faces

stamps/extract-%: stamps/facecrop-$$* ./scripts/arrange_training_set.sh
	bash ./scripts/arrange_training_set.sh $* $(DATA_LOC) $(TRAINING_SIZE) $(TEST_SIZE)
	touch $@

stamps/facecrop-%: stamps/facetag-$$*
	bash ./scripts/facecrop.sh $(DATA_LOC)/$*-scenes $(DATA_LOC)/$*-faces stamps/facetag-$*
	touch $@
	
stamps/facetag-%: stamps/splitscenes-$$* scripts/facetag.sh
	bash ./scripts/facetag.sh $* $(DATA_LOC) $@

stamps/splitscenes-%: stamps/scenes-$$*
	bash ./scripts/splitscenes.sh $(DATA_LOC)/$* $(DATA_LOC)/$*-scenes stamps/scenes-$*
	touch $@

stamps/scenes-%: stamps/genimages-$$*
	ffprobe -hide_banner -show_frames -of compact=p=0 -f lavfi 'movie=$(DATA_LOC)/$*/%06d.png,select=gt(scene\,.2)' | tee $@

stamps/genimages-%: $(DATA_LOC)/$$*.mp4
	mkdir -p $(DATA_LOC)/$*
	ffmpeg -hide_banner -i $(DATA_LOC)/$*.mp4 -vf fps=$(FPS) $(DATA_LOC)/$*/%06d.png
	touch $@

$(DATA_LOC)/%.mp4:
	@if test ! -f $(DATA_LOC)/$*.mp4; then echo "\nYou need $(DATA_LOC)/$*.mp4\n"; exit 1; fi

## RECYCLE datasets
recycle-gan-data: stamps/recycle-gan-data-$(A)-$(B)

stamps/recycle-gan-data-$(A)-$(B): stamps/extract-$(A) stamps/extract-$(B)
	bash ./scripts/recycle-gan-data.sh $(A) $(B) $(DATA_LOC)
	touch $@

## Training

.PHONY: train recycle-gan test

train: check-condaenv
	$(MAKE) recycle-gan 
	#make recycle-gan 

recycle-gan: stamps/recycle-gan-train-$(A)-$(B)
stamps/recycle-gan-train-$(A)-$(B): stamps/recycle-gan-data-$(A)-$(B)
	bash scripts/train_recycle_gan.sh $(A) $(B) $(GPU_ID) $(VISDOM_PORT)
	touch $@
mon: check-condaenv
	@echo "Use VISDOM_PORT $(VISDOM_PORT)"
	@python -m visdom.server -port $(VISDOM_PORT) & echo $$! > .visdom.pid
	@echo "Visdom pid is `cat .visdom.pid`"
	@echo "You can kill it with 'make kmon'"

kmon:
	@kill `cat .visdom.pid` 2>/dev/null && echo "Killed visdom.." && rm -f .visdom.pid

## Testing
.PHONY: test video recycle-gan-test make-video

test: check-condaenv
	$(MAKE) recycle-gan-test 
	#$(MAKE) recycle-gan

recycle-gan-test: stamps/recycle-gan-test-$(A)-$(B)
stamps/recycle-gan-test-$(A)-$(B): stamps/recycle-gan-train-$(A)-$(B)
	bash scripts/test_recycle_gan.sh $(A) $(B) $(TEST_SIZE) $(GPU_ID) $(EPOCH) 
	touch $@

video: check-condaenv
	$(MAKE) make-video

make-video: stamps/make-video-$(A)-$(B) stamps/recycle-gan-test-$(A)-$(B)
stamps/make-video-$(A)-$(B):
	bash scripts/make_video.sh AB $(A) $(B) $(EPOCH) 5
	bash scripts/make_video.sh BA $(A) $(B) $(EPOCH) 5
	touch $@

## Cleaning

clean:
	$(RM) -r stamps/*

clean-all: clean
	find $(DATA_LOC) -maxdepth 1 -type d -a -name '*-*' -a ! -name '*-known-*' -print0 | xargs -0 rm -rf

clean-data:
	$(RM) stamps/splitscenes-* stamps/genimages-* stamps/scenes-* stamps/extract-* stamps/facecrop-* stamps/facetag-*

clean-train:
	$(RM) stamps/recycle-gan-train-*

clean-test:
	$(RM) stamps/recycle-gan-test-*

clean-video:
	$(RM) stamps/make-video-*

gpuinfo:
	nvidia-smi -q -d MEMORY,TEMPERATURE
	nvidia-smi -L

gpuinfo2:
	nvidia-smi -q -d UTILIZATION,ECC,TEMPERATURE,POWER,CLOCK,COMPUTE,PIDS,PERFORMANCE,SUPPORTED_CLOCKS,PAGE_RETIREMENT,ACCOUNTING,ENCODER_STATS,FBC_STATS

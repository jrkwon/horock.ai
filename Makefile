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

stamps/facecrop-%: $(DATA_LOC)/$$*-faces-area.txt
	bash ./scripts/facecrop.sh $* $(DATA_LOC)
	touch $@
	
$(DATA_LOC)/%-faces-area.txt: stamps/splitscenes-$$* scripts/facetag.sh
	bash ./scripts/facetag.sh $* $(DATA_LOC) $@

stamps/splitscenes-%: stamps/scenes-$$*
	bash ./scripts/splitscenes.sh $* $(DATA_LOC) stamps/scenes-$*
	touch $@

stamps/scenes-%: stamps/genimages-$$*
	ffprobe -hide_banner -show_frames -of compact=p=0 -f lavfi 'movie=$(DATA_LOC)/$*/%06d.png,select=gt(scene\,.3)' | tee $@

stamps/genimages-%: $(DATA_LOC)/$$*.mp4
	-rm -rf $(DATA_LOC)/$*
	-mkdir -p $(DATA_LOC)/$*
	ffmpeg -hide_banner -t 00:30:00 -i $(DATA_LOC)/$*.mp4 -vf fps=$(FPS) $(DATA_LOC)/$*/%06d.png
	touch $@

$(DATA_LOC)/%.mp4:
	@if test ! -f $(DATA_LOC)/$*.mp4; then echo "\nYou need $(DATA_LOC)/$*.mp4\n"; exit 1; fi

## RECYCLE datasets
recycle-gan-data: stamps/recycle-gan-data-$(A)-$(B)

stamps/recycle-gan-data-$(A)-$(B): stamps/extract-$(A) stamps/extract-$(B) scripts/recycle-gan-data.sh
	bash ./scripts/recycle-gan-data.sh $(A) $(B) $(DATA_LOC)
	touch $@

## Training

.PHONY: train recycle-gan test

train: check-condaenv
	$(MAKE) recycle-gan 

recycle-gan: stamps/recycle-gan-train-$(A)-$(B)
stamps/recycle-gan-train-$(A)-$(B): stamps/recycle-gan-data-$(A)-$(B)
	-bash scripts/train_recycle_gan.sh $(A) $(B) $(GPU_ID) $(VISDOM_PORT)
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

clean-%:
	$(RM) -r stamps/*$** $(DATA_LOC)/$* $(DATA_LOC)/$*-faces* $(DATA_LOC)/$*-scenes*

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


## custom ffmpeg
LIBTENSORFLOW_VERSION=gpu-linux-x86_64-1.12.0
FFMPEG_COMMIT=b073fb9eeae8f021a4e18886ccf73cda9f67b00c

build-ffmpeg: /opt/bin/ffmpeg

/opt/bin/ffmpeg: FFmpeg/ffmpeg
	sudo cp FFmpeg/ffmpeg FFmpeg/ffprobe /opt/bin/

FFmpeg/ffmpeg: /opt/lib/libtensorflow.so FFmpeg/ffbuild/config.mak
	cd FFmpeg && make

FFmpeg/ffbuild/config.mak: FFmpeg/Makefile
	cd FFmpeg && CFLAGS=-I/opt/include LDFLAGS="-L/opt/lib -Wl,-rpath=/opt/lib" ./configure --prefix=/opt --enable-libtensorflow

FFmpeg/Makefile:
	if test ! -d FFmpeg; then git clone https://github.com/FFmpeg/FFmpeg; fi
	cd FFmpeg && git checkout $(FFMPEG_COMMIT)

/opt/lib/libtensorflow.so: download/libtensorflow-$(LIBTENSORFLOW_VERSION).tar.gz
	sudo mkdir -p /opt && sudo tar -C /opt -x -f $<

download/libtensorflow-$(LIBTENSORFLOW_VERSION).tar.gz:
	wget https://storage.googleapis.com/tensorflow/libtensorflow/libtensorflow-$(LIBTENSORFLOW_VERSION).tar.gz -O $@


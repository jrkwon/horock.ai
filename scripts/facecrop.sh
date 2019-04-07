#!/bin/bash

#Remove comment if you want to debug
#set -x

NAME="$1"
DATA_LOC="$2"
INDIR="$DATA_LOC/$NAME-scenes"
OUTDIR="$DATA_LOC/$NAME-faces"
PLAN="$DATA_LOC/$NAME-faces-area.txt"
KNOWN="$DATA_LOC/$NAME-known-faces"
FACEDETECT_MODEL=${FACEDETECT_MODEL:-cnn}

if test ! -d "$INDIR" -o ! -f "$PLAN"; then
	echo "Usage: $0 <name> <dataset-dir>"
	echo "Input needed: $INDIR"
	echo "Input needed: $PLAN"
	exit 1
fi

. $(dirname $0)/common.sh

echo "Face cropping begin ..."
echo "We use cropping plan file: $PLAN"
echo "Delete output dir: $OUTDIR"
#rm -rf "$OUTDIR" "$OUTDIR.1"
rm -rf "$OUTDIR"
mkdir -p "$OUTDIR" "$OUTDIR.1"

#CPUS=${CPUS:-4}
CPUS=$(cat /proc/cpuinfo 2>/dev/null | grep processor | wc -l)
SUBDIRSIZE=1000

echo "I found $CPUS cpus, mogrify will be launched at most $CPUS instances"

trap 'killall mogrify convert; exit' SIGINT
while read subdir begin_idx end_idx w h left top
do
	let 'w2 = w * 4'
	let 'h2 = h * 4'
	let 'top2 = top - ((h2 - h) / 2)'
	let 'left2 = left - ((w2 - w) / 2)'
	COUNT=0
	subdir2=$(zeropad 4 $((COUNT / SUBDIRSIZE)) )
	echo "Cropping dir $INDIR/$subdir $begin_idx -> $end_idx: origin ${w}x${h}+${left}+${top} -> extend ${w2}x${h2}+${left2}+${top2}: $OUTDIR.1/$subdir/$subdir2"
	mkdir -p "$OUTDIR.1/$subdir/$subdir2"
	find $INDIR/$subdir -name '*.png' -print | sort | while read file
	do
		filename=${file##*/}
		idx=${filename%.png}
		if (( 10#$idx < 10#$begin_idx || 10#$end_idx <= 10#$idx )); then
			continue
		fi

		let 'COUNT++'
		if test $((COUNT % SUBDIRSIZE)) -eq 0; then
			subdir2=$(zeropad 4 $((COUNT / SUBDIRSIZE)) )
			mkdir -p "$OUTDIR.1/$subdir/$subdir2"
		fi
		echo "$OUTDIR.1/$subdir/$subdir2" "$file"
	done | xargs -n 2 --max-procs=$CPUS mogrify -crop ${w2}x${h2}+${left2}+${top2} -path
done < "$PLAN"
	#done
#done < /dev/null

find $OUTDIR.1 -type d | sort | while read subdir
do
	ls $subdir 2>/dev/null | head -1
	if test ! -f $subdir/$(ls $subdir 2>/dev/null | head -1); then
		continue
	fi
	echo "Face detecting... $subdir (Log file) $subdir.detect.log"
	face_detection --model="${FACEDETECT_MODEL}" "$subdir" | sort > $subdir.detect.log || exit 1
done

find $OUTDIR.1 -name '*.detect.log' | sort | while read detectlog
do
	echo "Reading from $detectlog"
	read dummy dom_w <<<$(cat $detectlog | awk -F, '{ w = $3 - $5; h = $4 - $2; if( h > w ) { w = h; } if( w % 2 == 1 ) {w++;} print w}' | sort | uniq -c | sort -nr | head -1)
	let 'dom_w = dom_w * 2'
	while IFS=, read f top right bottom left
	do
		COUNT=1
		let "w = right - left"
		let "h = bottom - top"
		f2=${f/faces.1/faces}
		targetdir=${f2%/*png}
		if test ! -d "$targetdir"; then
			mkdir -p "$targetdir"
		fi
		let "top = top - (dom_w - h) / 2"
		let "top = ( top < 0 ) ? 0 : top"
		let "left = left - (dom_w - w) / 2"
		let "left = ( left < 0 ) ? 0 : left"
		echo convert "$f" +repage -crop ${dom_w}x${dom_w}+${left}+${top} +repage "$f2"
		convert "$f" +repage -crop ${dom_w}x${dom_w}+${left}+${top} +repage "$f2" &
		wait_max
	done < $detectlog

done

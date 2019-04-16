#!/bin/bash

#Remove comment if you want to debug
#set -x

NAME="$1"
DATA_LOC="$2"
INDIR="$DATA_LOC/$NAME-scenes"
OUTDIR="$DATA_LOC/$NAME-faces"
AREA="$DATA_LOC/$NAME-faces-area.txt"
KNOWN="$DATA_LOC/$NAME-known-faces"
FACEDETECT_MODEL=${FACEDETECT_MODEL:-cnn}
INTERSECT_RATIO=${INTERSECT_RATIO:-90}

if test ! -d "$INDIR" -o ! -f "$AREA"; then
	echo "Usage: $0 <name> <dataset-dir>"
	echo "Input needed: $INDIR"
	echo "Input needed: $AREA"
	exit 1
fi

. $(dirname $0)/common.sh

echo "Face cropping begin ..."
echo "We use cropping area file: $AREA"
echo "Delete output dir: $OUTDIR"
rm -rf "$OUTDIR" "$OUTDIR.1"
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
done < "$AREA"

find $OUTDIR.1 -type d | sort | while read subdir
do
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
	IFS=, read f otop oright obottom oleft <<<$(head -1 $detectlog)
	while IFS=, read f top right bottom left
	do
		COUNT=1
		f2=${f/faces.1/faces}
		targetdir=${f2%/*png}
		if test ! -d "$targetdir"; then
			mkdir -p "$targetdir"
		fi

		let "min_right = right < oright ? right : oright"
		let "min_bottom = bottom < obottom ? bottom : obottom"
		let "max_left = left > oleft ? left : oleft"
		let "max_top = top > otop ? top : otop"
		let "intersect = (min_right - max_left) * (min_bottom - max_top)"
		let "area = (right - left) * (bottom - top)"

		echo "($oleft,$otop)-($oright,$obottom) / ($left,$top)-($right,$bottom)" $(( intersect * 100 / area ))%
		if (( intersect * 100 / area < $INTERSECT_RATIO )); then
			otop=$top obottom=$bottom oleft=$left oright=$right
			#Drop this frame for discontinued sequence.
			continue
		fi

		let "w = oright - oleft"
		let "h = obottom - otop"

		let "top = otop - (dom_w - h) / 2"
		let "left = oleft - (dom_w - w) / 2"

		let "top = ( top < 0 ) ? 0 : top"
		let "left = ( left < 0 ) ? 0 : left"

		echo convert "$f" +repage -crop ${dom_w}x${dom_w}+${left}+${top} +repage -resize 256x256 "$f2"
		convert "$f" +repage -crop ${dom_w}x${dom_w}+${left}+${top} +repage -resize 256x256 "$f2" &
		wait_max
	done < $detectlog

done

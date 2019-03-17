#!/bin/bash

NAME="$1"
DATA_LOC="$2"
PLAN="$3"
FACEDETECT_MODEL=${FACEDETECT_MODEL:-cnn}
#SAMPLING_INTERVAL=200
SAMPLING_INTERVAL=10000

if test -z "$NAME" -o -z "$DATA_LOC"; then
	echo "Usage: $0 <name> <datasets-dir> <out-file>"
	exit 1
fi

INDIR="$DATA_LOC/$NAME-scenes"
KNOWN="$DATA_LOC/$NAME-known-faces"

if test ! -d "$INDIR"; then
	echo $INDIR needed for face tagging
	exit 1
fi

if test ! -d "$KNOWN"; then
	echo Reference directory $KNOWN needed for face tagging
	exit 1
fi

if which face_detection; then
	:
else
	echo "You need face_detection, install it via pip"
	echo "Or you haven't run '. conda-horock' maybe?"
	exit 1
fi

TMPDIR1="$INDIR-samples"
TMPDIR2="$INDIR-samplefaces"
rm -rf $TMPDIR1 $TMPDIR2
mkdir -p $TMPDIR1 $TMPDIR2
for d in $(cd $INDIR; ls -d [0-9]*)
do
	COUNT=0
	IMGCOUNT=0
	ls "$INDIR/$d/" | while read f
	do
		if test $((COUNT % SAMPLING_INTERVAL)) -eq 0; then
			let 'IMGCOUNT++'
			IMGFILE="$INDIR/$d/$f"
			echo Copying $IMGFILE to "$TMPDIR1/$d.$f"
			ln -Lf $IMGFILE "$TMPDIR1/$d.$f"
			#cp -f $IMGFILE "$TMPDIR1/$d.$f"
		fi
		let 'COUNT++'
	done
done

#Deprecated
rm -rf "$INDIR-firstimgs" "$INDIR-firstfaces"

echo "Face detecting... $TMPDIR2/detect.log (with model: $FACEDETECT_MODEL)"
face_detection --model="${FACEDETECT_MODEL}" "$TMPDIR1" | sort > $TMPDIR2/detect.log || exit 1

while IFS=, read f top right bottom left
do
	COUNT=1
	while true
	do
		OUT="$f.$COUNT.png"
		if test ! -f "$OUT"; then
			break
		fi
		let COUNT++
	done
	let "w = right - left"
	let "h = bottom - top"
	f2=$(basename $f)
	f2=${f2%.png}
	echo "$f": ${w}x${h}+${left}+${top} -> "$TMPDIR2/$f2.${w}.${h}.${left}.${top}.png"
	convert "$f" -crop ${w}x${h}+${left}+${top} "$TMPDIR2/$f2.${w}.${h}.${left}.${top}.png"
done < $TMPDIR2/detect.log

echo "Face recognizing... $TMPDIR2/recog.log"
face_recognition --tolerance=0.4 --show-distance=1 "$KNOWN" "$TMPDIR2" | sort > $TMPDIR2/recog.log || exit 1

while IFS=, read f tag distance
do
	if test "$tag" != "$NAME"; then
		continue
	fi
	IFS=. read subdir idx w h left top ext <<<"$f"
	subdir=${subdir##*/}
	echo $subdir $idx $w $h $left $top
	echo "SUBDIR $subdir $idx: $w*$h $left $top" >&2
done < $TMPDIR2/recog.log | sort > $INDIR/plan.log

cp $INDIR/plan.log "$PLAN"

echo "Cropping plan produced... $INDIR/plan.log"
echo "The plan copied to $PLAN"


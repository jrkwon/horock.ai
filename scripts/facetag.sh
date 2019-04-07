#!/bin/bash

NAME="$1"
DATA_LOC="$2"
AREA="$3"
FACEDETECT_MODEL=${FACEDETECT_MODEL:-cnn}
#SAMPLING_INTERVAL=200
SAMPLING_INTERVAL=100

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

echo "Input dir: $INDIR"
echo "Known faces: $KNOWN"

TMPDIR1="$INDIR-samples"
TMPDIR2="$INDIR-samplefaces"
rm -rf $TMPDIR1 $TMPDIR2
mkdir -p $TMPDIR1 $TMPDIR2

echo "Samples from scenes: $TMPDIR1"
echo "Extracted samples faces: $TMPDIR2"
echo ""
echo "(1) Copying sample scenes to $TMPDIR1"
for d in $(cd $INDIR; ls -d [0-9]*)
do
	COUNT=0
	IMGCOUNT=0
	ls "$INDIR/$d/" | while read f
	do
		if (( COUNT % SAMPLING_INTERVAL == 0 )); then
			let 'IMGCOUNT++'
			IMGFILE="$INDIR/$d/$f"
			echo "Copying(link) $IMGFILE to $TMPDIR1/$d.$f"
			ln -Lf $IMGFILE "$TMPDIR1/$d.$f" &
			#cp -f $IMGFILE "$TMPDIR1/$d.$f"
		fi
		let 'COUNT++'
	done
done

#Deprecated
rm -rf "$INDIR-firstimgs" "$INDIR-firstfaces"

echo "(2) Detecting faces from samples $TMPDIR1, log file: $TMPDIR2/detect.log (with model: $FACEDETECT_MODEL)"
face_detection --model="${FACEDETECT_MODEL}" "$TMPDIR1" | sort > $TMPDIR2/detect.log || exit 1
echo ""

echo "(3) Cropping the detected faces from $TMPDIR2/detect.log"
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
	convert "$f" -crop ${w}x${h}+${left}+${top} +repage "$TMPDIR2/$f2.${w}.${h}.${left}.${top}.png"
done < $TMPDIR2/detect.log

echo "(4) Recognizing faces from cropped images ... $TMPDIR2/recog.log"
face_recognition --tolerance=0.4 --show-distance=1 "$KNOWN" "$TMPDIR2" | sort > $TMPDIR2/recog.log || exit 1

prev_subdir=
bgin_idx=
end_idx=
prev_idx=0
grep ",$NAME," "$TMPDIR2/recog.log" | while IFS=, read f tag distance
do
	IFS=. read subdir idx w h left top ext <<<"$f"
	subdir=${subdir##*/}

	#Check beginning or end of subdir
	if test -z "$prev_subdir" -o "$prev_subdir" != "$subdir"; then
		if test -n "$prev_subdir"; then
			echo $prev_subdir $begin_idx $end_idx $w $h $left $top
			echo $prev_subdir $begin_idx $end_idx >&2
		fi
		prev_subdir=$subdir
		begin_idx=$idx
		end_idx=$idx
		prev_idx=$idx
		continue
	fi

	#Check end of streak in a subdir
	if (( 10#$prev_idx + $SAMPLING_INTERVAL != 10#$idx )); then
		echo $prev_subdir $begin_idx $end_idx $w $h $left $top
		echo $prev_subdir $begin_idx $end_idx >&2
		begin_idx=$idx
	fi
	prev_idx=$idx
	end_idx=$idx
	prev_subdir=$subdir
done | sort > "$AREA.tmp"

mv "$AREA.tmp" "$AREA"

echo "Cropping areas ($NAME only) produced... $AREA"


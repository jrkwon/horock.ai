#!/bin/bash

NAME="$1"
DATA_LOC="$2"
PROBES="$3"

INDIR="$DATA_LOC/$NAME"
OUTDIR="$INDIR-scenes"
MINFILESPERSCENE=300

. $(dirname $0)/common.sh

if test -z "$INDIR" -o -z "$PROBES"; then
	echo "Usage: $0 <png-extracted-dir> <ffprob result>"
	echo "Output: <png-extracted-dir>-scenes/<subdirs>/"
	exit 1
fi

echo "Probe file: $PROBES"
COUNT=1
while read LINE
do
	while read -d '|' KV
	do
		IFS="=" read KEY ENDFRAME <<< "$KV"
		if test "$KEY" != "pkt_pts"; then
			continue
		fi
		echo Scene change: $ENDFRAME >&2
		echo $ENDFRAME
		let 'COUNT++'
	done <<< "$LINE"
done < "$PROBES" > "$PROBES.1"
find "$INDIR" -name '[0-9]*.png' -print | awk -F/ '{print $NF}' | sort | tail -1 | tr -cd '0-9' >> "$PROBES.1"

echo "Total $COUNT scenes. Now splitting..."
SCENES=$(zeropad 4 $COUNT)

DIRCOUNT=1
COUNT=1
NAME=$(basename $INDIR)
rm -rf "$OUTDIR"
APPROVED=
while read ENDFRAME
do
	SUBDIR=$(zeropad 4 $DIRCOUNT)
	let 'FILES = ENDFRAME - COUNT + 1'
	echo "Scene #$SUBDIR/$SCENES ($FILES files)..."
	if test $FILES -lt $MINFILESPERSCENE; then
		echo "Skip this scene($SUBDIR) just has $FILES files. (min. $MINFILESPERSCENE files required)"
		let 'DIRCOUNT++'
		let 'COUNT = ENDFRAME'
		continue
	fi
	let 'COUNT += 30'
	mkdir -p "$OUTDIR/$SUBDIR/"
	IMGFILES=""
	APPROVED=Y
	while test $COUNT -le $ENDFRAME
	do
		IMGFILE="$(zeropad 6 $COUNT).png"
		IMGFILES="$IMGFILES ../../$NAME/$IMGFILE"
		let 'COUNT++'
		if test ${#IMGFILES} -gt 3000; then
			echo "Linking ${IMGFILES:0:25} --> ${IMGFILES:0-24} to $OUTDIR/$SUBDIR/"
			ln -sf $IMGFILES "$OUTDIR/$SUBDIR/" &
			IMGFILES=
		fi
	done
	if test -n "$IMGFILES"; then
		echo "Linking ${IMGFILES:0:25} --> ${IMGFILES:0-24} to $OUTDIR/$SUBDIR/"
		ln -sf $IMGFILES "$OUTDIR/$SUBDIR/" &
	fi
	let 'DIRCOUNT++'
done < "$PROBES.1"

if test -z "$APPROVED"; then
	exit 1
fi
exit 0

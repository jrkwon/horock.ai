#!/bin/bash

INDIR="$1"
OUTDIR="$2"
PROBES="$3"

if test -z "$INDIR" -o -z "$OUTDIR" -o -z "$PROBES"; then
	echo "Usage: $0 <png-extracted-dir> <output-dir> <ffprob result>"
	exit 1
fi

while read LINE
do
	while read -d '|' KV
	do
		IFS="=" read KEY ENDFRAME <<< "$KV"
		if test "$KEY" != "pkt_pts"; then
			continue
		fi
		echo $ENDFRAME
	done <<< "$LINE"
done < "$PROBES" > "$PROBES.1"
find "$INDIR" -name '[0-9]*.png' -print | awk -F/ '{print $NF}' | sort | tail -1 | tr -cd '0-9' >> "$PROBES.1"

DIRCOUNT=1
COUNT=1
NAME=$(basename $INDIR)
while read ENDFRAME
do
	SUBDIR=$(printf "%04d" $DIRCOUNT)
	echo "Scene #$SUBDIR ..."
	mkdir -p "$OUTDIR/$SUBDIR/"
	while test $COUNT -le $ENDFRAME
	do
		IMGFILE="$(printf "%06d" $COUNT).png"
		if test -f "$INDIR/$IMGFILE"; then
			ln -sf "../../$NAME/$IMGFILE" "$OUTDIR/$SUBDIR/"
		else
			echo "Oops $INDIR/IMGFILE missing."
		fi
		let COUNT++
	done
	let DIRCOUNT++
done < "$PROBES.1"


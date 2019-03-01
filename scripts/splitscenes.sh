#!/bin/bash

INDIR="$1"
PROBES="$2"

if test -z "$INDIR"; then
	echo "Usage: $0 <png-extracted-dir> <ffprob result>"
	exit 1
fi

DIRCOUNT=1
COUNT=1
while read LINE
do
	while read -d '|' KV
	do
		IFS="=" read KEY ENDFRAME <<< "$KV"
		if test "$KEY" = "pkt_pts"; then
			SUBDIR=$(printf "%04d" $DIRCOUNT)
			echo "Scene #$SUBDIR ..."
			mkdir -p "$INDIR/$SUBDIR/"
			while test "$COUNT" -le "$ENDFRAME"
			do
				IMGFILE="$(printf "%05d" $COUNT).png"
				mv "$INDIR/$IMGFILE" "$INDIR/$SUBDIR/$IMGFILE"
				let COUNT++
			done
			let DIRCOUNT++
		fi
	done <<< "$LINE"
done < "$PROBES"

# Move remains
SUBDIR=$(printf "%04d" $DIRCOUNT)
echo "Scene #$SUBDIR ..."
mkdir -p "$INDIR/$SUBDIR"
mv "$INDIR"/*.png "$INDIR/$SUBDIR/"

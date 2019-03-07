#!/bin/bash

INDIR="$1"
OUTDIR="$2"
PLAN="$3"

if test -z "$INDIR" -o -z "$PLAN"; then
	echo "Usage: $0 <png-extracted-dir> <output-dir> <cropping plan file>"
	exit 1
fi

rm -rf "$OUTDIR"
mkdir -p "$OUTDIR"

CPUS=$(cat /proc/cpuinfo 2>/dev/null | grep processor | wc -l)
CPUS=${CPUS:-4}

trap 'killall mogrify; exit' SIGINT
while read subdir w h left top
do
	let 'w2 = w * 2'
	let 'h2 = h * 2'
	let 'top2 = top - (h*2/5)'
	let 'left2 = left - (w/2)'
	echo "Cropping dir $INDIR/$subdir : origin ${w}x${h}+${left}+${top} -> extend ${w2}x${h2}+${left2}+${top2}"
	mogrify -path "$OUTDIR" -crop ${w2}x${h2}+${left2}+${top2} -resize 256x256^ $INDIR/$subdir/*.png &
	while RUNNING=$(jobs -r | wc -l) && test "$RUNNING" -ge $CPUS
	do
		sleep 2
	done
done < "$PLAN"

wait

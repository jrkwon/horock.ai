#!/bin/bash

INDIR="$1"
OUTDIR="$2"
PLAN="$3"

if test -z "$INDIR" -o -z "$PLAN"; then
	echo "Usage: $0 <png-extracted-dir> <output-dir> <cropping plan file>"
	exit 1
fi

echo "Face cropping begin ..."
echo "We use cropping plan file: $PLAN"

read INPUTFILESHASH DUMMY <<<$(while read subdir w h left top
do
	ls $INDIR/$subdir/* | xargs stat -L -c "%n %s"
done < "$PLAN" | md5sum -)
echo "Input files size hash: $INPUTFILESHASH"
PREEXISTEDHASH=$(cat $OUTDIR/hash.txt 2>/dev/null)

if test "$INPUTFILESHASH" = "$PREEXISTEDHASH"; then
	echo "Output has hash value: $PREEXISTEDHASH: $OUTDIR/hash.txt"
	echo "Skip face cropping."
	exit 0
else
	echo "Delete output dir: $OUTDIR"
	rm -rf "$OUTDIR"
	mkdir -p "$OUTDIR"
fi
exit

CPUS=$(cat /proc/cpuinfo 2>/dev/null | grep processor | wc -l)
CPUS=${CPUS:-4}

echo "I found $CPUS cpus, mogrify will be launched at most $CPUS instances"

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
echo $INPUTFILESHASH > $OUTDIR/hash.txt

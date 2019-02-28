#!/bin/bash

SRC_DATA_LOC=$1
DST_DATA_LOC=$2

FILE1=
FILE2=
FILE3=
COUNT=0
mkdir -p $DST_DATA_LOC 2>/dev/null
find $SRC_DATA_LOC/ -name '*.png' | sort | while read file
do
        FILE1=$FILE2
        FILE2=$FILE3
        FILE3=$file
        if test -n "$FILE1" -a -n "$FILE2" -a -n "$FILE3"; then
                outfile=$(printf %05d $COUNT)
                echo convert "$FILE1" "$FILE2" "$FILE3" +append $DST_DATA_LOC/$outfile.png
                convert "$FILE1" "$FILE2" "$FILE3" +append $DST_DATA_LOC/$outfile.png 

                ((COUNT++))
                if test $(( $COUNT % 200 )) -eq 0; then
                        echo "Waiting..."
                        wait
                fi
        fi
done


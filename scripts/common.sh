
_SKIP=0
wait_max() {
	# ORDER: Check and wait child process without executing any external command in bash
	if test $_SKIP -gt 0; then
		let '_SKIP--'
		return
	fi
	MAXCHILDREN="$1"
	if test $((MAXCHILDREN)) -eq 0; then
		echo "Usage: wait_max <max children>"
		echo "       wait if children count is over <max children>"
	fi

	while true
	do
		RUNNING=0
		while read line; do let 'RUNNING++'; done <<< "$(jobs -r)"
		if test "$RUNNING" -lt "$MAXCHILDREN"; then
			let '_SKIP = MAXCHILDREN - RUNNING'
			break
		fi
		read -t .2 x < /dev/zero
	done
}

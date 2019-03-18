
_SKIP=0
_CPUS=$(cat /proc/cpuinfo 2>/dev/null | grep processor | wc -l)
_CPUS=${_CPUS:-4}
wait_max() {
	# ORDER: Check and wait child process without executing any external command in bash
	if test $_SKIP -gt 0; then
		let '_SKIP--'
		return
	fi
	MAXCHILDREN="$1"
	if test $((MAXCHILDREN)) -eq 0; then
		MAXCHILDREN=$_CPUS
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

zeropad() {
	DIGITS="$1"
	NUMBER="$2"
	local out
	out="00000000000000000000000000$NUMBER"
	echo ${out:0-$DIGITS}
}

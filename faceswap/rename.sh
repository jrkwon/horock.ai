ls datasets/dataA/idol1/ | cat -n | while read n f; do mv datasets/dataA/idol1/"$f" `printf datasets/dataA/idol1/"%05d.png" $n`; done

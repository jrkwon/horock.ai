#!/usr/bin/env python
import os
import sys
from visdom import server

try:
    server.download_scripts_and_run()
except KeyboardInterrupt:
    print("Bye bye")

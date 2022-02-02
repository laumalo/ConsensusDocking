#!/bin/bash

# $1 original name (eg. ../piper/dockname)
# $2 new name (eg. ../piper/piper)

for i in $1_*; do mv ${i} ${i/#$1_/$2_}; done

#!/bin/bash

N=$1
ROBOCODE=$2

: "${N:=2}"
: "${ROBOCODE:="$HOME/robocode"}"

start_bg() {
  cd /home/patryk/robocode/ || return
  java --add-opens java.base/sun.net.www.protocol.jar=ALL-UNNAMED --add-exports java.desktop/sun.awt=ALL-UNNAMED -Xmx512M -Dsun.io.useCanonCaches=false -Ddebug=true -DNOSECURITY=true -DROBOTPATH=$ROBOCODE/robots -Dfile.encoding=UTF-8 -classpath "/home/patryk/studia/iwisum/Plato/plato-robot/bin/*:$ROBOCODE/libs/*:/home/patryk/studia/iwisum/libs/*" robocode.Robocode -battle /home/patryk/studia/iwisum/Plato/train.battle -tps 150 &
  sleep 1
  cd /home/patryk/studia/iwisum/Plato || return
}

rm -r /tmp/plato

javac -cp "plato-robot/libs/*:$ROBOCODE/libs/*:/home/patryk/studia/iwisum/libs/*" -Xlint:deprecation -d plato-robot/bin plato-robot/src/lk/*

tensorboard --logdir=/tmp/plato &
cd plato-server || return
python3 main.py &
cd ..

sleep 10

for i in $(seq 1 $N); do start_bg; done

trap "trap - SIGTERM && kill -- -$$" SIGINT SIGTERM EXIT

wait

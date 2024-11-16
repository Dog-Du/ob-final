./tools/deploy/obd.sh stop -n single
cp build_debug/src/observer/observer /tmp/obtest/bin/observer
./tools/deploy/obd.sh start -n single

bash ./build.sh debug --make -j40
./tools/deploy/obd.sh destroy -n single --rm
./tools/deploy/obd.sh prepare -p /tmp/obtest
./tools/deploy/obd.sh deploy -c ./tools/deploy/single.yaml

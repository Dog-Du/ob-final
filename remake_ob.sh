bash ./build.sh debug --make -j40
./tools/deploy/obd.sh destroy -n obcluster --rm
./tools/deploy/obd.sh prepare -p /tmp/obtest
./tools/deploy/obd.sh deploy -c obcluster.yaml

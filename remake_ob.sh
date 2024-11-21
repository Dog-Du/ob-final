bash ./build.sh release --make -j40
rm -rf ./tools/deploy/.obd/respository/oceanbase-ce/4.3.3.1
./tools/deploy/obd.sh destroy -n obcluster --rm
./tools/deploy/obd.sh prepare -p /tmp/obtest
./tools/deploy/obd.sh deploy -c obcluster.yaml

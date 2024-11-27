bash ./build.sh release --make -j32
rm -rf ./tools/deploy/.obd/repository/oceanbase-ce/4.3.3.1
echo "remove respository"
./tools/deploy/obd.sh destroy -n obcluster --rm
./tools/deploy/obd.sh prepare -p /tmp/obtest
./tools/deploy/obd.sh deploy -c obcluster.yaml
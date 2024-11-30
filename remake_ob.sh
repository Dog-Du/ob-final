rm -rf ./tools/deploy/.obd/repository/oceanbase-ce/4.3.3.1
./tools/deploy/obd.sh destroy -n obcluster --rm
echo "remove respository"
bash ./build.sh release --init --make -j24
./tools/deploy/obd.sh prepare -p /tmp/obtest
echo "4" | ./tools/deploy/obd.sh deploy -c obcluster.yaml
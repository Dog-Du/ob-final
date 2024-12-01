rm -rf ./tools/deploy/.obd/repository/oceanbase-ce/4.3.3.1
./tools/deploy/obd.sh destroy -n obcluster --rm < ./destroy.args
echo "remove respository"
bash ./build.sh release --init --make -j24
./tools/deploy/obd.sh prepare -p /tmp/obtest
./tools/deploy/obd.sh deploy -c obcluster.yaml < ./deploy.args
bash ./link_ob.sh < ./create_ten.args
bash ./load.sh
bash ./run_ann_benchmarks.sh

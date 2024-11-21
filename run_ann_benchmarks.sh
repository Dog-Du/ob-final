# 下载测试脚本
cd ann-benchmarks
# 运行基础 ann_benchmarks 测试
# time python run.py --algorithm oceanbase --local --force --dataset sift-128-euclidean
## 测试时导入数据并构建索引
time python run.py --algorithm oceanbase --local --force --dataset sift-128-euclidean --runs 3 --skip_fit
## 测试时跳过导入数据及构建索引
# time python run.py --algorithm oceanbase --local --force --dataset sift-128-euclidean --runs 1 --skip_fit
# 计算召回率及 QPS
time python plot.py --dataset sift-128-euclidean --recompute
# 重启 oceanbase 集群
bash ../tools/deploy/obd.sh start -n obcluster cluster restart obcluster

# 运行混合标量查询场景SQL，hybrid_ann.py 位于 ann_benchmarks/algorithms/oceanbase/
time python ./ann_benchmarks/algorithms/oceanbase/hybrid_ann.py
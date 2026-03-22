import sys, os

os.environ["PYTHONIOENCODING"] = "utf-8"

# 确保项目根目录在路径中
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from benchmark.performance_benchmark import BenchmarkRunner

if __name__ == "__main__":
    runner = BenchmarkRunner(device="cpu")  # 先用 CPU 测试
    runner.run_all_benchmarks()

import psutil
import time
import matplotlib.pyplot as plt
import signal
import sys

# 定义全局变量用于存储 I/O 数据
timestamps = []
read_io = []
write_io = []
running = True  # 用于控制脚本运行状态
last_read = 0
last_write = 0

def signal_handler(sig, frame):
    """
    捕获 Ctrl+C 信号，停止监控并生成图表
    """
    global running
    print("\nStopping monitoring and generating the chart...")
    running = False
    plot_io()

def monitor_io(pid, interval=1):
    """
    监控指定进程的 I/O 活动
    :param pid: 目标进程的 PID
    :param interval: 采样间隔（秒）
    """
    global timestamps, read_io, write_io, running, last_read, last_write

    # 尝试获取目标进程
    try:
        process = psutil.Process(pid)
        print(f"Monitoring I/O for process PID={pid}. Press Ctrl+C to stop.")
    except psutil.NoSuchProcess:
        print(f"Process with PID {pid} not found.")
        sys.exit(1)

    # 初始化读取和写入的初始值
    last_read, last_write = process.io_counters().read_bytes, process.io_counters().write_bytes

    # 监控循环
    while running:
        try:
            io_counters = process.io_counters()
            current_read, current_write = io_counters.read_bytes, io_counters.write_bytes

            # 计算间隔时间内的读写速率
            read_rate = (current_read - last_read) / (1024 * 1024) / interval  # 转换为 MB/s
            write_rate = (current_write - last_write) / (1024 * 1024) / interval  # 转换为 MB/s

            # 更新最后读取值
            last_read, last_write = current_read, current_write

            timestamps.append(time.time())
            read_io.append(read_rate)
            write_io.append(write_rate)

            time.sleep(interval)
        except psutil.NoSuchProcess:
            print(f"Process with PID {pid} terminated.")
            break
        except Exception as e:
            print(f"Error monitoring process: {e}")
            break

def plot_io():
    """
    绘制 I/O 图表
    """
    global timestamps, read_io, write_io

    if not timestamps:
        print("No data collected, skipping chart generation.")
        return

    # 将时间戳转换为秒（相对起始时间）
    start_time = timestamps[0]
    timestamps = [t - start_time for t in timestamps]

    # 绘制图表
    plt.figure(figsize=(10, 6))
    plt.plot(timestamps, read_io, label="Read I/O (MB/s)", color="blue")
    plt.plot(timestamps, write_io, label="Write I/O (MB/s)", color="orange")
    plt.xlabel("Time (s)")
    plt.ylabel("I/O Rate (MB/s)")
    plt.title("Process I/O Activity (MB/s)")
    plt.legend()
    plt.grid(True)
    plt.savefig("io_chart_mb.png")
    plt.show()
    print("Chart saved as 'io_chart_mb.png'.")

if __name__ == "__main__":
    # 捕获 Ctrl+C 信号
    signal.signal(signal.SIGINT, signal_handler)

    # 用户输入目标进程 PID
    try:
        target_pid = int(input("Enter the PID of the process to monitor: "))
        monitor_io(target_pid)
    except ValueError:
        print("Invalid PID.")
        sys.exit(1)

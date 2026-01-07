#!/usr/bin/env python3
"""
simple_rd_wr_analysis.py - 简化的读写分析脚本

功能：
- 生成 4DIE 的 latency vs 时间散点图
- 请求/返回计数随时间变化图
- 带宽随时间变化图

参数：
- --input: 输入目录
- --output: 输出目录
- --window: 时间窗口 (默认 1000 cycles)
- --freq: 频率 (默认 2.2 GHz)
"""

import re
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
import argparse
import glob
import bisect
from collections import defaultdict
from matplotlib import gridspec

def get_interface(file_path, core):
    """
    从文件名提取接口标识，包含 core，如 core1_ea0_tc0
    """
    stem = Path(file_path).stem
    match = re.search(r'(ea\d+_tc\d+|tc\d+_ea\d+)', stem)
    if match:
        parts = match.group().split('_')
        if parts[0].startswith('ea'):
            interface = f"{parts[0]}_{parts[1]}"
        else:
            interface = f"{parts[1]}_{parts[0]}"
        return f"{core}_{interface}"
    return f"{core}_unknown"

def parse_vec_file(file_path, max_cycle_cutoff=None):
    """
    解析 .vec 文件，返回字段名和数据 DataFrame
    如果提供了 max_cycle_cutoff，在列表阶段就进行过滤，避免创建包含所有数据的 DataFrame
    """
    fields = []
    data_lines = []
    in_data = False

    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('field['):
                # 提取字段名，如 field[12] = 16'h tag -> tag
                parts = line.split('=')
                if len(parts) >= 2:
                    field_def = parts[1].strip()
                    if 'h ' in field_def:
                        field_name = field_def.split('h ')[1].strip()
                        fields.append(field_name)
            elif line == 'data':
                in_data = True
            elif in_data and line and not line.startswith('#'):
                # 数据行：stream_name values @ timestamp
                if '@' in line:
                    parts = line.split('@')
                    values_part = parts[0].strip()
                    timestamp_hex = parts[1].strip()
                    # 解析时间戳（在列表阶段就应用 cycle cutoff 过滤）
                    timestamp = int(timestamp_hex, 10)  # 时间戳是十进制
                    if max_cycle_cutoff is not None and timestamp > max_cycle_cutoff:
                        continue  # 跳过超过 cutoff 的数据，不添加到列表
                    # 解析值
                    values = values_part.split()
                    if len(values) > 1:  # stream_name + values
                        data_values = values[1:]  # 跳过 stream_name
                        if len(data_values) == len(fields):
                            row = dict(zip(fields, data_values))
                            row['timestamp'] = timestamp  # 重用已计算的 timestamp
                            row['timestamp_hex'] = timestamp_hex
                            data_lines.append(row)

    df = pd.DataFrame(data_lines)
    # 转换数值列，排除 tag 和 timestamp
    for col in df.columns:
        if col not in ['timestamp', 'tag']:
            try:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            except:
                pass
    return df

def get_max_cycle_cutoff(input_dir):
    """
    从 *sq0_spi_msg_sim.vec 文件中解析 msg == '00' 的最大 cycle 值作为截至 cycle
    """
    input_path = Path(input_dir)
    spi_msg_files = list(input_path.glob('*sq0_spi_msg_sim.vec'))
    
    if not spi_msg_files:
        print("Warning: No *sq0_spi_msg_sim.vec files found in input directory. No cycle cutoff will be applied.")
        return None
    
    max_cycle = None
    max_file = None
    found_msg_00 = False
    
    for file_path in spi_msg_files:
        try:
            with open(file_path, 'r') as f:
                lines = f.readlines()
            
            in_data = False
            for line in lines:
                line = line.strip()
                if line.startswith('data'):
                    in_data = True
                    continue
                if not in_data or not line:
                    continue
                
                # 解析数据行
                if '@' in line:
                    parts = line.split('@')
                    if len(parts) == 2:
                        values_part = parts[0].strip()
                        timestamp_hex = parts[1].strip()
                        values = values_part.split()
                        if len(values) >= 3:  # stream_name, valid, msg, ...
                            msg = values[2]  # msg 是第三个字段
                            if msg == '00':
                                cycle = int(timestamp_hex)  # cycle 是十进制字符串
                                if max_cycle is None or cycle > max_cycle:
                                    max_cycle = cycle
                                    max_file = file_path.name
                                found_msg_00 = True
        except Exception as e:
            print(f"Error parsing {file_path}: {e}")
    
    if not found_msg_00:
        print("Warning: No records with msg == '00' found in *sq0_spi_msg_sim.vec files. No cycle cutoff will be applied.")
        return None
    
    print(f"Max cycle cutoff from msg == '00': {max_cycle} (from file: {max_file})")
    return max_cycle

def parse_rdreq_files(file_paths, max_cycle_cutoff=None):
    """
    解析 rdreq 文件
    """
    dfs = []
    for file_path in file_paths:
        # Check if the interface is valid (contains 'ea' and 'tc')
        stem = Path(file_path).stem
        if not (re.search(r'ea\d+_tc\d+', stem) or re.search(r'tc\d+_ea\d+', stem)):
            continue

        # 在解析时就应用 cycle cutoff 过滤（在列表阶段过滤，避免创建包含所有数据的 DataFrame）
        df = parse_vec_file(file_path, max_cycle_cutoff=max_cycle_cutoff)
        if not df.empty:
            # 重命名 timestamp 为 timestamp_req
            df = df.rename(columns={'timestamp': 'timestamp_req', 'timestamp_hex': 'timestamp_hex'})
            # 添加 file_name
            df['file_name'] = Path(file_path).name
            # 提取 core（与 pvm_latency.py 的逻辑保持一致）
            stem = Path(file_path).stem
            core_id = '0'  # 默认值
            core_match = re.match(r'^core(\d+)_+', stem)
            if core_match:
                core_id = core_match.group(1)
            core = f"core{core_id}"
            df['core'] = core
            # 提取 interface
            df['interface'] = get_interface(file_path, core)
            dfs.append(df)
    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()

def parse_rdret_files(file_paths, max_cycle_cutoff=None):
    """
    解析 rdret 文件
    """
    dfs = []
    for file_path in file_paths:
        # Check if the interface is valid (contains 'ea' and 'tc')
        stem = Path(file_path).stem
        if not (re.search(r'ea\d+_tc\d+', stem) or re.search(r'tc\d+_ea\d+', stem)):
            continue

        # 在解析时就应用 cycle cutoff 过滤（在列表阶段过滤，避免创建包含所有数据的 DataFrame）
        df = parse_vec_file(file_path, max_cycle_cutoff=max_cycle_cutoff)
        if not df.empty:
            # 重命名 timestamp 为 timestamp_ret
            df = df.rename(columns={'timestamp': 'timestamp_ret', 'timestamp_hex': 'timestamp_hex'})
            # 添加 file_name
            df['file_name'] = Path(file_path).name
            # 提取 core（与 pvm_latency.py 的逻辑保持一致）
            stem = Path(file_path).stem
            core_id = '0'  # 默认值
            core_match = re.match(r'^core(\d+)_+', stem)
            if core_match:
                core_id = core_match.group(1)
            core = f"core{core_id}"
            df['core'] = core
            # 提取 interface
            df['interface'] = get_interface(file_path, core)
            dfs.append(df)
    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()

def parse_wrreq_files(file_paths, max_cycle_cutoff=None):
    """
    解析 wrreq 文件
    """
    dfs = []
    for file_path in file_paths:
        # Check if the interface is valid (contains 'ea' and 'tc')
        stem = Path(file_path).stem
        if not (re.search(r'ea\d+_tc\d+', stem) or re.search(r'tc\d+_ea\d+', stem)):
            continue

        # 在解析时就应用 cycle cutoff 过滤（在列表阶段过滤，避免创建包含所有数据的 DataFrame）
        df = parse_vec_file(file_path, max_cycle_cutoff=max_cycle_cutoff)
        if not df.empty:
            # 重命名 timestamp 为 timestamp_req
            df = df.rename(columns={'timestamp': 'timestamp_req', 'timestamp_hex': 'timestamp_hex'})
            # 添加 file_name
            df['file_name'] = Path(file_path).name
            # 提取 core（与 pvm_latency.py 的逻辑保持一致）
            stem = Path(file_path).stem
            core_id = '0'  # 默认值
            core_match = re.match(r'^core(\d+)_+', stem)
            if core_match:
                core_id = core_match.group(1)
            core = f"core{core_id}"
            df['core'] = core
            # 提取 interface
            df['interface'] = get_interface(file_path, core)
            dfs.append(df)
    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()

def parse_wrret_files(file_paths, max_cycle_cutoff=None):
    """
    解析 wrret 文件
    """
    dfs = []
    for file_path in file_paths:
        # Check if the interface is valid (contains 'ea' and 'tc')
        stem = Path(file_path).stem
        if not (re.search(r'ea\d+_tc\d+', stem) or re.search(r'tc\d+_ea\d+', stem)):
            continue

        # 在解析时就应用 cycle cutoff 过滤（在列表阶段过滤，避免创建包含所有数据的 DataFrame）
        df = parse_vec_file(file_path, max_cycle_cutoff=max_cycle_cutoff)
        if not df.empty:
            # 重命名 timestamp 为 timestamp_ret
            df = df.rename(columns={'timestamp': 'timestamp_ret', 'timestamp_hex': 'timestamp_hex'})
            # 添加 file_name
            df['file_name'] = Path(file_path).name
            # 提取 core（与 pvm_latency.py 的逻辑保持一致）
            stem = Path(file_path).stem
            core_id = '0'  # 默认值
            core_match = re.match(r'^core(\d+)_+', stem)
            if core_match:
                core_id = core_match.group(1)
            core = f"core{core_id}"
            df['core'] = core
            # 提取 interface
            df['interface'] = get_interface(file_path, core)
            dfs.append(df)
    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()

def calculate_latency(req_df, ret_df, freq=2.2):
    """
    计算读 latency：按 interface 和 tag 匹配请求和返回，根据 size 确定返回数量
    使用列列表而不是字典列表，以节省内存
    """
    size_to_returns = {1: 1, 3: 2, 5: 3, 7: 4}

    # 预处理返回数据，按 (interface, tag) 分组
    # 使用 itertuples() 代替 iterrows() 以提高内存效率
    ret_dict = defaultdict(list)
    for ret in ret_df.itertuples(index=False):
        ret_dict[(ret.interface, ret.tag)].append(ret.timestamp_ret)
    # 排序每个列表
    for key in ret_dict:
        ret_dict[key].sort()

    # 预处理请求数据，按 interface 分组
    # 使用 itertuples() 代替 iterrows() 以提高内存效率
    req_dict = defaultdict(list)
    for req in req_df.itertuples(index=False):
        req_dict[req.interface].append(req)

    # 使用分批处理策略避免内存峰值
    MAX_BATCH_SIZE = 1_000_000  # 每批最多100万条记录
    
    # 收集所有批次的 DataFrame
    all_batch_dfs = []
    current_batch = {
        'tags': [],
        'interfaces': [],
        'timestamp_reqs': [],
        'timestamp_rets': [],
        'latencies_ns': [],
        'file_names': [],
        'timestamp_hexs': [],
        'cores': [],
        'sizes': []
    }
    batch_count = 0

    # 按 interface 分组处理
    for interface, req_list in req_dict.items():
        for req in req_list:
            # itertuples() 返回命名元组，使用属性访问
            tag = req.tag
            size = req.size
            timestamp_req = req.timestamp_req
            timestamp_req_hex = req.timestamp_hex

            # 确定需要的返回数量
            num_returns = size_to_returns.get(size, 1)

            # 获取对应 tag 的返回列表
            ret_list = ret_dict.get((interface, tag), [])

            # 找到该请求时间戳之后的所有返回（使用二分查找优化）
            # 使用 bisect 来找到第一个大于 timestamp_req 的位置
            start_idx = bisect.bisect_right(ret_list, timestamp_req)
            matched_rets = ret_list[start_idx:start_idx + num_returns]

            # 计算每个匹配返回的 latency
            for timestamp_ret in matched_rets:
                latency_cycles = timestamp_ret - timestamp_req
                latency_ns = latency_cycles / freq
                current_batch['tags'].append(tag)
                current_batch['interfaces'].append(interface)
                current_batch['timestamp_reqs'].append(timestamp_req)
                current_batch['timestamp_rets'].append(timestamp_ret)
                current_batch['latencies_ns'].append(latency_ns)
                current_batch['file_names'].append(req.file_name)
                current_batch['timestamp_hexs'].append(timestamp_req_hex)
                current_batch['cores'].append(req.core)
                current_batch['sizes'].append(size)
                
                batch_count += 1
                
                # 当批次达到一定大小时，转换为 DataFrame 并清空批次
                if batch_count >= MAX_BATCH_SIZE:
                    batch_df = _create_latency_dataframe_from_batch(current_batch)
                    all_batch_dfs.append(batch_df)
                    # 清空当前批次
                    for key in current_batch:
                        current_batch[key].clear()
                    batch_count = 0
                    import gc
                    gc.collect()  # 强制垃圾回收
    
    # 处理剩余的批次
    if batch_count > 0:
        batch_df = _create_latency_dataframe_from_batch(current_batch)
        all_batch_dfs.append(batch_df)
    
    # 合并所有批次的结果
    if not all_batch_dfs:
        return pd.DataFrame(columns=['tag', 'interface', 'timestamp_req', 'timestamp_ret', 'latency', 'file_name', 'timestamp_hex', 'core', 'size'])
    
    # 合并所有 DataFrame
    final_df = pd.concat(all_batch_dfs, ignore_index=True)
    return final_df

def _create_latency_dataframe_from_batch(batch):
    """从批次数据创建 DataFrame 的辅助函数"""
    if not batch['tags']:
        return pd.DataFrame(columns=['tag', 'interface', 'timestamp_req', 'timestamp_ret', 'latency', 'file_name', 'timestamp_hex', 'core', 'size'])
    
    # 将 tag 从字符串（可能是十六进制）转换为整数
    def parse_tag(tag_val):
        """将 tag 值转换为整数，支持十六进制和十进制字符串"""
        if isinstance(tag_val, (int, np.integer)):
            return int(tag_val)
        if isinstance(tag_val, str):
            try:
                return int(tag_val, 16)
            except ValueError:
                try:
                    return int(tag_val, 10)
                except ValueError:
                    return 0
        return int(tag_val)
    
    tags_int = [parse_tag(t) for t in batch['tags']]
    
    # 使用分步创建 DataFrame 来避免内存峰值
    data_dict = {
        'tag': np.array(tags_int, dtype=np.int32),
        'timestamp_req': np.array(batch['timestamp_reqs'], dtype=np.int32),
        'timestamp_ret': np.array(batch['timestamp_rets'], dtype=np.int32),
        'latency': np.array(batch['latencies_ns'], dtype=np.float32),
        'size': np.array(batch['sizes'], dtype=np.int8)
    }
    
    # 先创建只包含数值列的 DataFrame
    df = pd.DataFrame(data_dict)
    
    # 然后逐个添加字符串列
    df['interface'] = batch['interfaces']
    df['file_name'] = batch['file_names']
    df['timestamp_hex'] = batch['timestamp_hexs']
    df['core'] = batch['cores']
    
    return df

def calculate_write_latency(req_df, ret_df, freq=2.2):
    """
    计算写 latency：按 interface 和 tag 分组，根据 size 确定连续请求数量，匹配一个返回，计算每个请求的 latency
    使用列列表而不是字典列表，以节省内存
    """
    # 预处理返回数据，按 (interface, tag) 分组
    # 使用 itertuples() 代替 iterrows() 以提高内存效率
    ret_dict = defaultdict(list)
    for ret in ret_df.itertuples(index=False):
        ret_dict[(ret.interface, ret.tag)].append(ret.timestamp_ret)
    # 排序每个列表
    for key in ret_dict:
        ret_dict[key].sort()

    # 预处理请求数据，按 interface 分组
    # 使用 itertuples() 代替 iterrows() 以提高内存效率
    req_dict = defaultdict(list)
    for req in req_df.itertuples(index=False):
        req_dict[req.interface].append(req)

    # 使用列列表而不是字典列表，以节省内存
    tags = []
    interfaces = []
    timestamp_reqs = []
    timestamp_rets = []
    latencies_ns = []
    file_names = []
    sizes = []
    cores = []

    # 按 interface 分组处理
    for interface, req_list in req_dict.items():
        # 按 tag 和 timestamp 排序
        req_list.sort(key=lambda x: (x.tag, x.timestamp_req))

        i = 0
        while i < len(req_list):
            req = req_list[i]
            tag = req.tag
            size = req.size

            # 根据 size 确定分组大小
            group_size = {1: 1, 3: 2, 5: 3, 7: 4}.get(size, 1)
            if i + group_size > len(req_list):
                # 分组不完整，跳过
                i += 1
                continue

            # 检查分组内 tag 是否相同且连续
            group_reqs = req_list[i:i + group_size]
            if not all(r.tag == tag for r in group_reqs):
                i += 1
                continue

            # 取最早 timestamp
            earliest_ts = min(r.timestamp_req for r in group_reqs)

            # 匹配 wrret
            ret_list = ret_dict.get((interface, tag), [])
            # 使用二分查找找到第一个 timestamp > earliest_ts 的 wrret
            start_idx = bisect.bisect_right(ret_list, earliest_ts)
            if start_idx < len(ret_list):
                timestamp_ret = ret_list[start_idx]  # 取第一个
                # 计算每个请求的 latency
                for req in group_reqs:
                    latency_cycles = timestamp_ret - req.timestamp_req
                    latency_ns = latency_cycles / freq
                    tags.append(tag)
                    interfaces.append(interface)
                    timestamp_reqs.append(req.timestamp_req)
                    timestamp_rets.append(timestamp_ret)
                    latencies_ns.append(latency_ns)
                    file_names.append(req.file_name)
                    sizes.append(size)
                    cores.append(req.core)

            i += group_size

    # 如果列表为空，返回空 DataFrame
    if not tags:
        return pd.DataFrame(columns=['tag', 'interface', 'timestamp_req', 'timestamp_ret', 'latency', 'file_name', 'size', 'core'])
    
    # 将 tag 从字符串（可能是十六进制）转换为整数
    def parse_tag(tag_val):
        """将 tag 值转换为整数，支持十六进制和十进制字符串"""
        if isinstance(tag_val, (int, np.integer)):
            return int(tag_val)
        if isinstance(tag_val, str):
            # 尝试十六进制，如果失败则尝试十进制
            try:
                return int(tag_val, 16)
            except ValueError:
                try:
                    return int(tag_val, 10)
                except ValueError:
                    return 0
        return int(tag_val)
    
    tags_int = [parse_tag(t) for t in tags]
    
    # 使用分步创建 DataFrame 来避免内存峰值
    # 先创建数值列（使用 numpy 数组和指定数据类型）
    # 对于整数类型使用 int32 而不是 int64，可以节省一半内存
    data_dict = {
        'tag': np.array(tags_int, dtype=np.int32),
        'timestamp_req': np.array(timestamp_reqs, dtype=np.int32),
        'timestamp_ret': np.array(timestamp_rets, dtype=np.int32),
        'latency': np.array(latencies_ns, dtype=np.float32),
        'size': np.array(sizes, dtype=np.int8)  # size 值很小，用 int8 足够
    }
    
    # 先创建只包含数值列的 DataFrame
    df = pd.DataFrame(data_dict)
    
    # 然后逐个添加字符串列，避免一次性创建所有列时的内存峰值
    df['interface'] = interfaces
    df['file_name'] = file_names
    df['core'] = cores
    
    return df

def calculate_bandwidth(ret_df, window_cycles=1000):
    """
    计算读带宽：按时间窗口累加 data_size
    优化内存使用：只选择需要的列，避免不必要的复制
    """
    # 检查 DataFrame 是否为空或缺少必要的列
    if ret_df.empty or 'timestamp_ret' not in ret_df.columns:
        return pd.DataFrame(columns=['time_bin', 'bandwidth_bytes_per_cycle'])
    
    # 只选择需要的列，避免复制整个 DataFrame
    # 使用视图而不是复制，然后只复制 timestamp_ret 列
    timestamp_series = ret_df['timestamp_ret'].copy()
    
    # 排序（对 Series 排序比对整个 DataFrame 排序更节省内存）
    timestamp_series = timestamp_series.sort_values()
    
    # 创建时间 bins
    min_time = timestamp_series.min()
    max_time = timestamp_series.max()
    bins = np.arange(min_time, max_time + window_cycles, window_cycles)

    # 分配到 bins（使用 numpy 的 searchsorted 实现左闭右开区间，与 pd.cut(right=False) 一致）
    # searchsorted(side='right') 返回插入位置，减1得到 bin 索引
    bin_indices = np.searchsorted(bins, timestamp_series.values, side='right') - 1
    # 确保 bin_indices 在有效范围内（处理边界情况）
    bin_indices = np.clip(bin_indices, 0, len(bins) - 2)
    time_bins = bins[bin_indices]

    # 按 bin 累加（每个返回 64 字节）
    # 使用 numpy 的 bincount 更高效
    unique_bins, counts = np.unique(time_bins, return_counts=True)
    data_sizes = counts * 64  # 每个返回 64 字节
    
    # 创建结果 DataFrame
    bandwidth_df = pd.DataFrame({
        'time_bin': unique_bins,
        'data_size': data_sizes
    })
    bandwidth_df['bandwidth_bytes_per_cycle'] = bandwidth_df['data_size'] / window_cycles

    return bandwidth_df

def calculate_write_bandwidth(req_df, window_cycles=1000):
    """
    计算写带宽：按时间窗口累加 wrreq 数量 * 64 bytes
    优化内存使用：只选择需要的列，避免不必要的复制
    """
    # 检查 DataFrame 是否为空或缺少必要的列
    if req_df.empty or 'timestamp_req' not in req_df.columns:
        return pd.DataFrame(columns=['time_bin', 'bandwidth_bytes_per_cycle'])
    
    # 只选择需要的列，避免复制整个 DataFrame
    timestamp_series = req_df['timestamp_req'].copy()
    
    # 排序（对 Series 排序比对整个 DataFrame 排序更节省内存）
    timestamp_series = timestamp_series.sort_values()
    
    # 创建时间 bins
    min_time = timestamp_series.min()
    max_time = timestamp_series.max()
    bins = np.arange(min_time, max_time + window_cycles, window_cycles)

    # 分配到 bins（使用 numpy 的 searchsorted 实现左闭右开区间，与 pd.cut(right=False) 一致）
    # searchsorted(side='right') 返回插入位置，减1得到 bin 索引
    bin_indices = np.searchsorted(bins, timestamp_series.values, side='right') - 1
    # 确保 bin_indices 在有效范围内（处理边界情况）
    bin_indices = np.clip(bin_indices, 0, len(bins) - 2)
    time_bins = bins[bin_indices]

    # 按 bin 累加（每个请求 64 字节）
    # 使用 numpy 的 bincount 更高效
    unique_bins, counts = np.unique(time_bins, return_counts=True)
    data_sizes = counts * 64  # 每个请求 64 字节
    
    # 创建结果 DataFrame
    bandwidth_df = pd.DataFrame({
        'time_bin': unique_bins,
        'data_size': data_sizes
    })
    bandwidth_df['bandwidth_bytes_per_cycle'] = bandwidth_df['data_size'] / window_cycles

    return bandwidth_df

def plot_merge_latency_vs_time(read_latency_df, write_latency_df, output_dir, prefix="4DIE"):
    """
    合并读写延迟 vs 时间散点图
    """
    if read_latency_df.empty and write_latency_df.empty:
        return

    fig, ax = plt.subplots(figsize=(12, 6))

    # 读 latency
    if not read_latency_df.empty:
        ax.scatter(read_latency_df['timestamp_req'], read_latency_df['latency'], alpha=0.6, s=1, color='blue', label='Read')

    # 写 latency
    if not write_latency_df.empty:
        ax.scatter(write_latency_df['timestamp_req'], write_latency_df['latency'], alpha=0.6, s=1, color='red', label='Write')

    ax.set_xlabel('Request Time (Cycles)')
    ax.set_ylabel('Latency (ns)')
    ax.set_title(f'{prefix} Read/Write Latency vs Request Time Scatter Plot')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / f'{prefix}_6_latency_vs_time.png', dpi=150)
    plt.close()

def plot_merge_bandwidth_over_time(read_bw_df, write_bw_df, output_dir, freq=2.2, prefix="4DIE"):
    """
    合并读写带宽随时间变化图
    """
    if read_bw_df.empty and write_bw_df.empty:
        return

    # 合并数据
    merged_df = pd.DataFrame()
    if not read_bw_df.empty:
        read_bw_df = read_bw_df.copy()
        read_bw_df['read_bandwidth'] = read_bw_df['bandwidth_bytes_per_cycle'] * freq
        merged_df = pd.merge(merged_df, read_bw_df[['time_bin', 'read_bandwidth']], on='time_bin', how='outer') if not merged_df.empty else read_bw_df[['time_bin', 'read_bandwidth']].copy()
    if not write_bw_df.empty:
        write_bw_df = write_bw_df.copy()
        write_bw_df['write_bandwidth'] = write_bw_df['bandwidth_bytes_per_cycle'] * freq
        merged_df = pd.merge(merged_df, write_bw_df[['time_bin', 'write_bandwidth']], on='time_bin', how='outer') if not merged_df.empty else write_bw_df[['time_bin', 'write_bandwidth']].copy()

    # 填充缺失值为 0
    merged_df = merged_df.fillna(0)

    # 按 time_bin 排序
    merged_df = merged_df.sort_values('time_bin')

    # 计算总带宽（允许只有读或只有写的情况）
    read_bw = merged_df['read_bandwidth'] if 'read_bandwidth' in merged_df.columns else 0
    write_bw = merged_df['write_bandwidth'] if 'write_bandwidth' in merged_df.columns else 0
    merged_df['total_bandwidth'] = read_bw + write_bw

    fig, ax = plt.subplots(figsize=(12, 6))

    # 读带宽
    if 'read_bandwidth' in merged_df.columns:
        ax.plot(merged_df['time_bin'], merged_df['read_bandwidth'], label='Read Bandwidth', linewidth=1, color='blue')

    # 写带宽
    if 'write_bandwidth' in merged_df.columns:
        ax.plot(merged_df['time_bin'], merged_df['write_bandwidth'], label='Write Bandwidth', linewidth=1, color='red')

    # 总带宽
    ax.plot(merged_df['time_bin'], merged_df['total_bandwidth'], label='Total Bandwidth', linewidth=1, color='orange')

    ax.set_xlabel('Time (Cycles)')
    ax.set_ylabel('Bandwidth (GB/s)')
    ax.set_title(f'{prefix} Read/Write/Bandwidth Over Time')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 添加 max 标注（放在图右上角上侧）
    stats_text = ''
    if 'read_bandwidth' in merged_df.columns:
        read_max = merged_df['read_bandwidth'].max()
        stats_text += f'Read Max: {read_max:.2f} GB/s\n'
    if 'write_bandwidth' in merged_df.columns:
        write_max = merged_df['write_bandwidth'].max()
        stats_text += f'Write Max: {write_max:.2f} GB/s\n'
    total_max = merged_df['total_bandwidth'].max()
    stats_text += f'Max: {total_max:.2f} GB/s'
    if stats_text:
        # 添加 max 标注
        ax.text(0.98, 0.98, stats_text.strip(), transform=ax.transAxes, fontsize=10,
                verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.savefig(output_dir / f'{prefix}_3_bandwidth_over_time.png', dpi=150, bbox_inches='tight')
    plt.close()

def plot_total_bandwidth_over_time(total_bw_df, output_dir, freq=2.2, prefix="4DIE"):
    """
    总带宽随时间变化图
    """
    if total_bw_df.empty:
        return

    fig, ax = plt.subplots(figsize=(12, 6))

    bandwidth_gb_per_sec = total_bw_df['bandwidth_bytes_per_cycle'] * freq
    ax.plot(total_bw_df['time_bin'], bandwidth_gb_per_sec, label='Total Bandwidth', linewidth=1, color='black')

    ax.set_xlabel('Time (Cycles)')
    ax.set_ylabel('Bandwidth (GB/s)')
    ax.set_title(f'{prefix} Bandwidth Over Time')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 添加 max 标注（放在图右上角上侧）
    max_bw = bandwidth_gb_per_sec.max()
    stats_text = f'Max: {max_bw:.2f} GB/s'
    # 统计信息放在图右上角、图例下方（仍在图内）
    ax.text(0.98, 0.98, stats_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.savefig(output_dir / f'{prefix}_10_total_bandwidth_over_time.png', dpi=150, bbox_inches='tight')
    plt.close()

def plot_combined_bandwidth_over_time(read_bw_df, write_bw_df, total_bw_df, output_dir, freq=2.2, prefix="4DIE"):
    """
    合并读写带宽和总带宽随时间变化图，使用子图布局
    """
    if read_bw_df.empty and write_bw_df.empty and total_bw_df.empty:
        return

    # 合并读写数据
    merged_df = pd.DataFrame()
    if not read_bw_df.empty:
        read_bw_df = read_bw_df.copy()
        read_bw_df['read_bandwidth'] = read_bw_df['bandwidth_bytes_per_cycle'] * freq
        merged_df = pd.merge(merged_df, read_bw_df[['time_bin', 'read_bandwidth']], on='time_bin', how='outer') if not merged_df.empty else read_bw_df[['time_bin', 'read_bandwidth']].copy()
    if not write_bw_df.empty:
        write_bw_df = write_bw_df.copy()
        write_bw_df['write_bandwidth'] = write_bw_df['bandwidth_bytes_per_cycle'] * freq
        merged_df = pd.merge(merged_df, write_bw_df[['time_bin', 'write_bandwidth']], on='time_bin', how='outer') if not merged_df.empty else write_bw_df[['time_bin', 'write_bandwidth']].copy()

    # 填充缺失值为 0
    merged_df = merged_df.fillna(0)

    # 按 time_bin 排序
    merged_df = merged_df.sort_values('time_bin')

    # 创建子图
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    # 上方子图：读写带宽
    if 'read_bandwidth' in merged_df.columns:
        ax1.plot(merged_df['time_bin'], merged_df['read_bandwidth'], label='Read Bandwidth', linewidth=1)
    if 'write_bandwidth' in merged_df.columns:
        ax1.plot(merged_df['time_bin'], merged_df['write_bandwidth'], label='Write Bandwidth', linewidth=1)
    ax1.set_ylabel('Bandwidth (GB/s)')
    ax1.set_title(f'{prefix} Read/Write Bandwidth Over Time')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 添加 max 标注到上方子图（放在图右上角上侧）
    stats_text = ''
    if 'read_bandwidth' in merged_df.columns:
        read_max = merged_df['read_bandwidth'].max()
        stats_text += f'Read Max: {read_max:.2f} GB/s\n'
    if 'write_bandwidth' in merged_df.columns:
        write_max = merged_df['write_bandwidth'].max()
        stats_text += f'Write Max: {write_max:.2f} GB/s'
    if stats_text:
        # 统计信息放在图右上角、图例下方（仍在图内）
        ax1.text(0.98, 0.98, stats_text.strip(), transform=ax1.transAxes, fontsize=10,
                verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # 下方子图：总带宽
    if not total_bw_df.empty:
        bandwidth_gb_per_sec = total_bw_df['bandwidth_bytes_per_cycle'] * freq
        ax2.plot(total_bw_df['time_bin'], bandwidth_gb_per_sec, label='Total Bandwidth', linewidth=1, color='black')
    ax2.set_xlabel('Time (Cycles)')
    ax2.set_ylabel('Bandwidth (GB/s)')
    ax2.set_title(f'{prefix} Bandwidth Over Time')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # 添加 max 标注到下方子图（放在图右上角上侧）
    if not total_bw_df.empty:
        max_bw = bandwidth_gb_per_sec.max()
        stats_text = f'Max: {max_bw:.2f} GB/s'
        # 添加 max 标注到下方子图
        ax2.text(0.98, 0.98, stats_text, transform=ax2.transAxes, fontsize=10,
                verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.savefig(output_dir / f'{prefix}_11_combined_bandwidth_over_time.png', dpi=150, bbox_inches='tight')
    plt.close()
def calculate_channel_avg_latency(read_latency_df, write_latency_df):
    """
    计算每个 channel 的平均延迟，按 die 分组
    返回: (die_read_avg_latencies, die_write_avg_latencies)
    die_read_avg_latencies: {die_id: [avg_lat_0, avg_lat_1, ..., avg_lat_15]}
    """
    def extract_die_and_channel(interface_str):
        """从 interface 字符串中提取 die_id 和 channel_id (tc_id)"""
        # interface 格式: "core{core_id}_ea{ea_id}_tc{tc_id}" 或 "core{core_id}_tc{tc_id}_ea{ea_id}"
        match = re.search(r'core(\d+)', interface_str)
        if not match:
            return None, None
        die_id = int(match.group(1))
        
        # 提取 tc_id
        tc_match = re.search(r'tc(\d+)', interface_str)
        if not tc_match:
            return None, None
        channel_id = int(tc_match.group(1))
        
        return die_id, channel_id
    
    # 处理读延迟：{die_id: {channel_id: [latencies]}}
    # 使用 itertuples() 代替 iterrows() 以提高内存效率
    die_read_latencies = defaultdict(lambda: defaultdict(list))
    if not read_latency_df.empty:
        for row in read_latency_df.itertuples(index=False):
            die_id, channel_id = extract_die_and_channel(row.interface)
            if die_id is not None and channel_id is not None and 0 <= channel_id < 16:
                die_read_latencies[die_id][channel_id].append(row.latency)
    
    # 计算每个 channel 的平均延迟
    # 确保包含所有4个die（0-3），即使某些die没有数据
    die_read_avg_latencies = {}
    all_die_ids = set(range(4))  # 确保包含die 0, 1, 2, 3
    all_die_ids.update(die_read_latencies.keys())
    
    for die_id in sorted(all_die_ids):
        avg_lats = []
        for ch_id in range(16):
            if die_id in die_read_latencies and ch_id in die_read_latencies[die_id] and len(die_read_latencies[die_id][ch_id]) > 0:
                avg_lats.append(np.mean(die_read_latencies[die_id][ch_id]))
            else:
                avg_lats.append(0.0)
        die_read_avg_latencies[die_id] = avg_lats
    
    # 处理写延迟
    # 使用 itertuples() 代替 iterrows() 以提高内存效率
    die_write_latencies = defaultdict(lambda: defaultdict(list))
    if not write_latency_df.empty:
        for row in write_latency_df.itertuples(index=False):
            die_id, channel_id = extract_die_and_channel(row.interface)
            if die_id is not None and channel_id is not None and 0 <= channel_id < 16:
                die_write_latencies[die_id][channel_id].append(row.latency)
    
    # 计算每个 channel 的平均延迟
    # 确保包含所有4个die（0-3），即使某些die没有数据
    die_write_avg_latencies = {}
    all_die_ids = set(range(4))  # 确保包含die 0, 1, 2, 3
    all_die_ids.update(die_write_latencies.keys())
    
    for die_id in sorted(all_die_ids):
        avg_lats = []
        for ch_id in range(16):
            if die_id in die_write_latencies and ch_id in die_write_latencies[die_id] and len(die_write_latencies[die_id][ch_id]) > 0:
                avg_lats.append(np.mean(die_write_latencies[die_id][ch_id]))
            else:
                avg_lats.append(0.0)
        die_write_avg_latencies[die_id] = avg_lats
    
    return die_read_avg_latencies, die_write_avg_latencies

def plot_combined_overview(read_latency_df, write_latency_df, total_latency_df, read_bw_df, write_bw_df, total_bw_df, read_req_df, write_req_df, output_dir, window_cycles, freq=2.2, y_axis_tick_threshold=5000, y_tick_step=100, prefix="4DIE"):
    """
    合并图9、图11、图5到一张图：3x2布局，左下和右下添加 channel average latency
    """
    # 计算全局最大时间，用于统一横轴范围
    max_time = 0
    if not read_latency_df.empty:
        max_time = max(max_time, read_latency_df['timestamp_req'].max())
    if not write_latency_df.empty:
        max_time = max(max_time, write_latency_df['timestamp_req'].max())

    # 计算 channel average latency
    die_read_avg_latencies, die_write_avg_latencies = calculate_channel_avg_latency(read_latency_df, write_latency_df)

    fig = plt.figure(figsize=(20, 20))
    gs = gridspec.GridSpec(4, 2, height_ratios=[1, 0.5, 0.5, 0.8], hspace=0.3, wspace=0.3)

    ax9_left = fig.add_subplot(gs[0, 0])  # latency vs time
    ax9_right = fig.add_subplot(gs[0, 1])  # latency percentage
    ax_req = fig.add_subplot(gs[1:3, 0])  # request count, span two rows
    ax_rw_bw = fig.add_subplot(gs[1, 1])  # read write bandwidth
    ax_total_bw = fig.add_subplot(gs[2, 1])  # total bandwidth
    ax_ch_lat_read = fig.add_subplot(gs[3, 0])  # channel average latency (read)
    ax_ch_lat_write = fig.add_subplot(gs[3, 1])  # channel average latency (write)

    # 图9：上方 - 延迟分析

    # 左图：latency vs 时间 scatter
    if not read_latency_df.empty:
        ax9_left.scatter(read_latency_df['timestamp_req'], read_latency_df['latency'], alpha=0.6, s=0.1, color='blue', label='Read')
    if not write_latency_df.empty:
        ax9_left.scatter(write_latency_df['timestamp_req'], write_latency_df['latency'], alpha=0.6, s=0.1, color='red', label='Write')
    ax9_left.set_xlabel('Time (Cycles)')
    ax9_left.set_ylabel('Latency (ns)')
    ax9_left.set_title('Latency vs Request Time')
    ax9_left.tick_params(axis='y', labelsize=10)  # 调整Y轴刻度数字字体大小
    # 图例始终保留 Read / Write，两条线可能有一条没有数据
    from matplotlib.lines import Line2D
    latency_legend_handles = [
        Line2D([], [], color='blue', label='Read'),
        Line2D([], [], color='red', label='Write'),
    ]
    ax9_left.legend(handles=latency_legend_handles, loc='upper left', bbox_to_anchor=(1.02, 1.0), borderaxespad=0.)
    ax9_left.grid(True, alpha=0.3)
    ax9_left.set_xlim(0, max_time)
    # 计算统一的Y轴最大值，确保左右两个图保持一致
    max_lat = max(read_latency_df['latency'].max() if not read_latency_df.empty else 0, write_latency_df['latency'].max() if not write_latency_df.empty else 0)
    if max_lat > 0:
        ax9_left.set_ylim(bottom=0, top=max_lat * 1.05)
        # 当max_lat < y_axis_tick_threshold时，每100画一个格子；否则自适应
        if max_lat < y_axis_tick_threshold:
            ax9_left.set_yticks(np.arange(0, max_lat * 1.05 + y_tick_step, y_tick_step))

    # 添加 max/min latency 标注（放在图右上角上侧）
    stats_text = ''
    if not read_latency_df.empty:
        read_max = read_latency_df['latency'].max()
        read_min = read_latency_df['latency'].min()
        stats_text += f'Read Max: {read_max:.2f} ns\nRead Min: {read_min:.2f} ns\n'
    if not write_latency_df.empty:
        write_max = write_latency_df['latency'].max()
        write_min = write_latency_df['latency'].min()
        stats_text += f'Write Max: {write_max:.2f} ns\nWrite Min: {write_min:.2f} ns'
    if stats_text:
        # 统计信息放在图右上角、图例下方（仍在图内）
        ax9_left.text(0.98, 0.98, stats_text.strip(), transform=ax9_left.transAxes, fontsize=8,
                verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # 右图：latency 比例分布
    # 使用与左图相同的max_lat，确保Y轴范围完全一致
    if max_lat > 0:
        bins = np.arange(0, max_lat + 50, 50)
        bin_centers = bins[:-1] + 25

        # 读 latency
        if not read_latency_df.empty:
            hist, _ = np.histogram(read_latency_df['latency'], bins=bins)
            percentages = (hist / hist.sum()) * 100
            ax9_right.plot(percentages, bin_centers, label='Read Latency', linewidth=1, color='blue')

        # 写 latency
        if not write_latency_df.empty:
            hist, _ = np.histogram(write_latency_df['latency'], bins=bins)
            percentages = (hist / hist.sum()) * 100
            ax9_right.plot(percentages, bin_centers, label='Write Latency', linewidth=1, color='red')

    ax9_right.set_xlabel('Percentage of Total Latency (%)')
    ax9_right.set_xscale('symlog')
    ax9_right.set_xticks([0, 1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
    ax9_right.set_ylabel('Latency (ns)')
    ax9_right.set_title('Latency Percentage Distribution (50ns bins)')
    ax9_right.tick_params(axis='y', labelsize=10)  # 调整Y轴刻度数字字体大小
    # 使用与左图相同的Y轴范围和刻度设置，确保完全一致
    if max_lat > 0:
        ax9_right.set_ylim(bottom=0, top=max_lat * 1.05)
        # 当max_lat < y_axis_tick_threshold时，每100画一个格子；否则自适应
        if max_lat < y_axis_tick_threshold:
            ax9_right.set_yticks(np.arange(0, max_lat * 1.05 + y_tick_step, y_tick_step))
    # 图例放在图外侧
    latency_pct_legend_handles = [
        Line2D([], [], color='blue', label='Read Latency'),
        Line2D([], [], color='red', label='Write Latency'),
    ]
    ax9_right.legend(handles=latency_pct_legend_handles, loc='upper left', bbox_to_anchor=(1.02, 1.0), borderaxespad=0.)
    ax9_right.grid(True, alpha=0.3)

    # 图11：左下方 - 带宽（拆分为读写和总）
    # 合并读写数据
    merged_df = pd.DataFrame()
    if not read_bw_df.empty:
        read_bw_df_copy = read_bw_df.copy()
        read_bw_df_copy['read_bandwidth'] = read_bw_df_copy['bandwidth_bytes_per_cycle'] * freq
        merged_df = pd.merge(merged_df, read_bw_df_copy[['time_bin', 'read_bandwidth']], on='time_bin', how='outer') if not merged_df.empty else read_bw_df_copy[['time_bin', 'read_bandwidth']].copy()
    if not write_bw_df.empty:
        write_bw_df_copy = write_bw_df.copy()
        write_bw_df_copy['write_bandwidth'] = write_bw_df_copy['bandwidth_bytes_per_cycle'] * freq
        merged_df = pd.merge(merged_df, write_bw_df_copy[['time_bin', 'write_bandwidth']], on='time_bin', how='outer') if not merged_df.empty else write_bw_df_copy[['time_bin', 'write_bandwidth']].copy()

    # 填充缺失值为 0
    merged_df = merged_df.fillna(0)

    # 按 time_bin 排序
    merged_df = merged_df.sort_values('time_bin')

    # 计算总带宽（允许只有读或只有写的情况）
    read_bw = merged_df['read_bandwidth'] if 'read_bandwidth' in merged_df.columns else 0
    write_bw = merged_df['write_bandwidth'] if 'write_bandwidth' in merged_df.columns else 0
    merged_df['total_bandwidth'] = read_bw + write_bw

    # 绘制读写带宽
    if 'read_bandwidth' in merged_df.columns:
        ax_rw_bw.plot(merged_df['time_bin'], merged_df['read_bandwidth'], linewidth=1, color='blue')
    if 'write_bandwidth' in merged_df.columns:
        ax_rw_bw.plot(merged_df['time_bin'], merged_df['write_bandwidth'], linewidth=1, color='red')

    ax_rw_bw.set_xlabel('Time (Cycles)')
    ax_rw_bw.set_ylabel('Bandwidth (GB/s)')
    ax_rw_bw.set_title('Read/Write Bandwidth Over Time')
    # 图例始终显示 Read / Write，放在图外
    bw_legend_handles = [
        Line2D([], [], color='blue', label='Read'),
        Line2D([], [], color='red', label='Write'),
    ]
    ax_rw_bw.legend(handles=bw_legend_handles, loc='upper left', bbox_to_anchor=(1.02, 1.0), borderaxespad=0.)
    ax_rw_bw.grid(True, alpha=0.3)
    ax_rw_bw.set_xlim(0, max_time)

    # 添加 max 标注（放在图右上角上侧）
    stats_text = ''
    if 'read_bandwidth' in merged_df.columns:
        read_max = merged_df['read_bandwidth'].max()
        stats_text += f'Read Max: {read_max:.2f} GB/s\n'
    if 'write_bandwidth' in merged_df.columns:
        write_max = merged_df['write_bandwidth'].max()
        stats_text += f'Write Max: {write_max:.2f} GB/s'
    if stats_text:
        # 统计信息放在图右上角、图例下方（仍在图内）
        ax_rw_bw.text(0.98, 0.98, stats_text.strip(), transform=ax_rw_bw.transAxes, fontsize=8,
                verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # 绘制总带宽
    ax_total_bw.plot(merged_df['time_bin'], merged_df['total_bandwidth'], label='Bandwidth', linewidth=1, color='orange')

    ax_total_bw.set_xlabel('Time (Cycles)')
    ax_total_bw.set_ylabel('Bandwidth (GB/s)')
    ax_total_bw.set_title('Bandwidth Over Time')
    ax_total_bw.legend(loc='upper left', bbox_to_anchor=(1.02, 1.0))
    ax_total_bw.grid(True, alpha=0.3)
    ax_total_bw.set_xlim(0, max_time)

    # 添加 max 标注（放在图右上角上侧）
    total_max = merged_df['total_bandwidth'].max()
    stats_text = f'Max: {total_max:.2f} GB/s'
    # 统计信息放在图右上角、图例下方（仍在图内）
    ax_total_bw.text(0.98, 0.98, stats_text, transform=ax_total_bw.transAxes, fontsize=8,
            verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # 图5：右下方 - 请求计数
    # 合并所有时间戳

    # 合并所有时间戳
    all_times = []
    if not read_req_df.empty:
        all_times.extend(read_req_df['timestamp_req'].tolist())
    if not write_req_df.empty:
        all_times.extend(write_req_df['timestamp_req'].tolist())

    if all_times:
        min_time = min(all_times)
        max_time = max(all_times)
        bins = np.arange(min_time, max_time + window_cycles, window_cycles)

        # 计算读请求加权计数
        size_to_count = {1: 1, 3: 2, 5: 3, 7: 4}
        if not read_req_df.empty:
            read_req_df_copy = read_req_df.copy()
            read_req_df_copy['time_bin'] = pd.cut(read_req_df_copy['timestamp_req'], bins, labels=bins[:-1], right=False)
            read_req_df_copy['time_bin'] = read_req_df_copy['time_bin'].astype(float)
            read_req_df_copy['weighted_count'] = read_req_df_copy['size'].map(size_to_count).fillna(1)
            read_req_counts = read_req_df_copy.groupby('time_bin')['weighted_count'].sum()
        else:
            read_req_counts = pd.Series(dtype=float)

        # 计算写请求计数
        if not write_req_df.empty:
            write_req_counts = pd.cut(write_req_df['timestamp_req'], bins, labels=bins[:-1], right=False).value_counts().sort_index()
        else:
            write_req_counts = pd.Series(dtype=int)

        # 合并数据
        time_bins = pd.DataFrame({'time_bin': bins[:-1]})
        time_bins = time_bins.merge(read_req_counts.rename('read_req'), left_on='time_bin', right_index=True, how='left').fillna(0)
        time_bins = time_bins.merge(write_req_counts.rename('write_req'), left_on='time_bin', right_index=True, how='left').fillna(0)

        ax_req.plot(time_bins['time_bin'], time_bins['read_req'], linewidth=1, color='blue')
        ax_req.plot(time_bins['time_bin'], time_bins['write_req'], linewidth=1, color='red')

        ax_req.set_xlabel('Time (Cycles)')
        ax_req.set_ylabel('Request Count')
        ax_req.set_title(f'Read/Write Request Count Over Time (Window: {window_cycles} cycles)')
        # 图例始终显示 Read / Write，放在图外
        req_legend_handles = [
            Line2D([], [], color='blue', label='Read Requests'),
            Line2D([], [], color='red', label='Write Requests'),
        ]
        ax_req.legend(handles=req_legend_handles, loc='upper left', bbox_to_anchor=(1.02, 1.0), borderaxespad=0.)
        ax_req.grid(True, alpha=0.3)
        ax_req.set_xlim(0, max_time)

        # 添加比例标注
        total_read = time_bins['read_req'].sum()
        total_write = time_bins['write_req'].sum()
        total = total_read + total_write
        if total > 0:
            read_pct = total_read / total * 100
            write_pct = total_write / total * 100
            pct_text = f'Read: {read_pct:.3f}%\nWrite: {write_pct:.3f}%'
            # 统计信息放在图右上角、图例下方（仍在图内）
            ax_req.text(0.98, 0.98, pct_text, transform=ax_req.transAxes, fontsize=8,
                    verticalalignment='top', horizontalalignment='right',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # ========= 左下：Channel Average Latency (Read) =========
    # 计算 y 轴范围的辅助函数
    def calculate_ylim(avg_latencies_dict):
        all_latencies = []
        die_avg_values = []
        for die_id in sorted(avg_latencies_dict.keys()):
            avg_lats = avg_latencies_dict[die_id]
            if len(avg_lats) < 16:
                avg_lats = avg_lats + [0.0] * (16 - len(avg_lats))
            elif len(avg_lats) > 16:
                avg_lats = avg_lats[:16]
            # 只收集非零值用于计算范围（避免全0的die影响范围计算）
            non_zero_lats = [lat for lat in avg_lats if lat > 0]
            if non_zero_lats:
                all_latencies.extend(non_zero_lats)
            die_avg = np.mean([lat for lat in avg_lats if lat > 0]) if any(lat > 0 for lat in avg_lats) else 0.0
            # if die_avg > 0:  # 只添加非零的平均值
            die_avg_values.append(die_avg)
        
        if not all_latencies or not die_avg_values:
            return 0, 100
        
        max_latency = max(all_latencies)
        min_latency = min(all_latencies)
        max_avg = max(die_avg_values)
        min_avg = min(die_avg_values)
        avg_diff = max_avg - min_avg if max_avg > min_avg else max_latency * 0.1  # 如果所有die平均值相同，使用10%的范围
        
        y_max = max_latency# * 1.4
        y_min = min_latency# * 0.6
        return y_min, y_max
    
    # 计算 read 和 write 的 y 轴范围，然后取最大值和最小值
    y_min_read, y_max_read = 0, 100  # 默认值
    y_min_write, y_max_write = 0, 100  # 默认值
    
    if die_read_avg_latencies:
        y_min_read, y_max_read = calculate_ylim(die_read_avg_latencies)
    
    if die_write_avg_latencies:
        y_min_write, y_max_write = calculate_ylim(die_write_avg_latencies)
    
    # 使用 read 和 write 中最大的上限和最小的下限
    print("test: ", min(y_min_read, y_min_write))
    y_min_combined = 0#min(y_min_read, y_min_write) 
    y_max_combined = max(y_max_read, y_max_write) * 1.4
    
    # 如果只有一个有数据，使用那个的范围
    if not die_read_avg_latencies and die_write_avg_latencies:
        y_min_combined, y_max_combined = y_min_write, y_max_write
    elif die_read_avg_latencies and not die_write_avg_latencies:
        y_min_combined, y_max_combined = y_min_read, y_max_read
    
    if die_read_avg_latencies:
        x_pos = np.arange(16)
        colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red']
        
        # 确保绘制所有4个die（0-3），即使某些die没有数据
        all_die_ids_to_plot = sorted(set(range(4)) | set(die_read_avg_latencies.keys()))
        
        for idx, die_id in enumerate(all_die_ids_to_plot):
            if die_id not in die_read_avg_latencies:
                # 如果die没有数据，创建一个全0的列表
                avg_lats = [0.0] * 16
            else:
                avg_lats = die_read_avg_latencies[die_id]
            
            if len(avg_lats) < 16:
                avg_lats = avg_lats + [0.0] * (16 - len(avg_lats))
            elif len(avg_lats) > 16:
                avg_lats = avg_lats[:16]
            
            color = colors[die_id % len(colors)]  # 使用die_id而不是idx来选择颜色，确保颜色一致
            ax_ch_lat_read.plot(x_pos, avg_lats, linestyle='-', linewidth=2,
                              color=color, alpha=0.8, label=f'Die{die_id}')
        
        ax_ch_lat_read.set_xlabel('Channel', fontsize=11)
        ax_ch_lat_read.set_ylabel('Average Latency(ns)', fontsize=11)
        ax_ch_lat_read.set_title('Read Average Latency Per Channel', fontsize=12)
        ax_ch_lat_read.set_xticks(x_pos)
        ax_ch_lat_read.set_xticklabels([str(i) for i in range(16)])
        ax_ch_lat_read.set_ylim(bottom=y_min_combined, top=y_max_combined)
        ax_ch_lat_read.grid(True, linestyle='--', alpha=0.4)
        # 将图例移到图外右侧，避免遮挡曲线
        ax_ch_lat_read.legend(loc='upper left', bbox_to_anchor=(1.02, 1.0), fontsize=9, borderaxespad=0.)
    
    # ========= 右下：Channel Average Latency (Write) =========
    if die_write_avg_latencies:
        x_pos = np.arange(16)
        colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red']
        
        # 确保绘制所有4个die（0-3），即使某些die没有数据
        all_die_ids_to_plot = sorted(set(range(4)) | set(die_write_avg_latencies.keys()))
        
        for idx, die_id in enumerate(all_die_ids_to_plot):
            if die_id not in die_write_avg_latencies:
                # 如果die没有数据，创建一个全0的列表
                avg_lats = [0.0] * 16
            else:
                avg_lats = die_write_avg_latencies[die_id]
            
            if len(avg_lats) < 16:
                avg_lats = avg_lats + [0.0] * (16 - len(avg_lats))
            elif len(avg_lats) > 16:
                avg_lats = avg_lats[:16]
            
            color = colors[die_id % len(colors)]  # 使用die_id而不是idx来选择颜色，确保颜色一致
            ax_ch_lat_write.plot(x_pos, avg_lats, linestyle='-', linewidth=2,
                               color=color, alpha=0.8, label=f'Die{die_id}')
        
        ax_ch_lat_write.set_xlabel('Channel', fontsize=11)
        ax_ch_lat_write.set_ylabel('Average Latency(ns)', fontsize=11)
        ax_ch_lat_write.set_title('Write Average Latency Per Channel', fontsize=12)
        ax_ch_lat_write.set_xticks(x_pos)
        ax_ch_lat_write.set_xticklabels([str(i) for i in range(16)])
        # 使用 read 和 write 中最大的上限和最小的下限
        ax_ch_lat_write.set_ylim(bottom=y_min_combined, top=y_max_combined)
        ax_ch_lat_write.grid(True, linestyle='--', alpha=0.4)
        # 将图例移到图外右侧，避免遮挡曲线
        ax_ch_lat_write.legend(loc='upper left', bbox_to_anchor=(1.02, 1.0), fontsize=9, borderaxespad=0.)

    # 为右侧图例预留空间（底部两个子图的图例都在右侧）
    plt.tight_layout(rect=[0.0, 0.0, 0.78, 1.0])
    plt.savefig(output_dir / f'{prefix}_12_combined_overview.png', dpi=150, bbox_inches='tight')
    plt.close()
def plot_merge_request_return_count_over_time(read_req_df, write_req_df, output_dir, window_cycles, prefix="4DIE"):
    """
    绘制读写请求计数随时间变化图（根据 size 加权读请求）
    """
    if read_req_df.empty and write_req_df.empty:
        return

    # 合并所有时间戳
    all_times = []
    if not read_req_df.empty:
        all_times.extend(read_req_df['timestamp_req'].tolist())
    if not write_req_df.empty:
        all_times.extend(write_req_df['timestamp_req'].tolist())

    if not all_times:
        return

    min_time = min(all_times)
    max_time = max(all_times)
    bins = np.arange(min_time, max_time + window_cycles, window_cycles)

    # 计算读请求加权计数
    size_to_count = {1: 1, 3: 2, 5: 3, 7: 4}
    if not read_req_df.empty:
        read_req_df_copy = read_req_df.copy()
        read_req_df_copy['time_bin'] = pd.cut(read_req_df_copy['timestamp_req'], bins, labels=bins[:-1], right=False)
        read_req_df_copy['time_bin'] = read_req_df_copy['time_bin'].astype(float)
        read_req_df_copy['weighted_count'] = read_req_df_copy['size'].map(size_to_count).fillna(1)
        read_req_counts = read_req_df_copy.groupby('time_bin')['weighted_count'].sum()
    else:
        read_req_counts = pd.Series(dtype=float)

    # 计算写请求计数
    if not write_req_df.empty:
        write_req_counts = pd.cut(write_req_df['timestamp_req'], bins, labels=bins[:-1], right=False).value_counts().sort_index()
    else:
        write_req_counts = pd.Series(dtype=int)

    # 合并数据
    time_bins = pd.DataFrame({'time_bin': bins[:-1]})
    time_bins = time_bins.merge(read_req_counts.rename('read_req'), left_on='time_bin', right_index=True, how='left').fillna(0)
    time_bins = time_bins.merge(write_req_counts.rename('write_req'), left_on='time_bin', right_index=True, how='left').fillna(0)

    fig, ax = plt.subplots(figsize=(12, 6))

    ax.plot(time_bins['time_bin'], time_bins['read_req'], label='Read Requests', linewidth=1, color='blue')
    ax.plot(time_bins['time_bin'], time_bins['write_req'], label='Write Requests', linewidth=1, color='red')

    ax.set_xlabel('Time (Cycles)')
    ax.set_ylabel('Request Count')
    ax.set_title(f'{prefix} Read/Write Request Count Over Time (Window: {window_cycles} cycles)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 添加比例标注
    total_read = time_bins['read_req'].sum()
    total_write = time_bins['write_req'].sum()
    total = total_read + total_write
    if total > 0:
        read_pct = total_read / total * 100
        write_pct = total_write / total * 100
        pct_text = f'Read: {read_pct:.3f}%\nWrite: {write_pct:.3f}%'
        # 统计信息放在图右上角、图例下方（仍在图内）
        ax.text(0.98, 0.98, pct_text, transform=ax.transAxes, fontsize=10,
                verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.savefig(output_dir / f'{prefix}_5_request_return_count_over_time.png', dpi=150, bbox_inches='tight')
    plt.close()

def plot_read_write_latency_percentage(read_latency_df, write_latency_df, output_dir, prefix="4DIE"):
    """
    绘制读写 latency 比例曲线图（50ns 跨度）
    """
    fig, ax = plt.subplots(figsize=(12, 6))

    # 读 latency
    if not read_latency_df.empty:
        max_lat = read_latency_df['latency'].max()
        bins = np.arange(0, max_lat + 50, 50)
        hist, bin_edges = np.histogram(read_latency_df['latency'], bins=bins)
        percentages = (hist / hist.sum()) * 100
        x = bin_edges[:-1] + 25  # bin centers
        ax.plot(x, percentages, label='Read Latency', linewidth=1)

    # 写 latency
    if not write_latency_df.empty:
        max_lat = write_latency_df['latency'].max()
        bins = np.arange(0, max_lat + 50, 50)
        hist, bin_edges = np.histogram(write_latency_df['latency'], bins=bins)
        percentages = (hist / hist.sum()) * 100
        x = bin_edges[:-1] + 25  # bin centers
        ax.plot(x, percentages, label='Write Latency', linewidth=1)

    ax.set_xlabel('Latency (ns)')
    ax.set_ylabel('Percentage of Total Latency (%)')
    ax.set_yscale('symlog')
    ax.set_yticks([0.01, 1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
    ax.set_title(f'{prefix} Read/Write Latency Percentage Distribution (50ns bins)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xticks(np.arange(0, max(max_lat for df in [read_latency_df, write_latency_df] if not df.empty) + 100, 100))
    plt.tight_layout()
    plt.savefig(output_dir / f'{prefix}_7_read_write_latency_percentage.png', dpi=150)
    plt.close()

def plot_total_latency_percentage(total_latency_df, output_dir, prefix="4DIE"):
    """
    绘制整个 latency 比例曲线图（50ns 跨度）
    """
    if total_latency_df.empty:
        return

    fig, ax = plt.subplots(figsize=(12, 6))

    max_lat = total_latency_df['latency'].max()
    bins = np.arange(0, max_lat + 50, 50)
    hist, bin_edges = np.histogram(total_latency_df['latency'], bins=bins)
    percentages = (hist / hist.sum()) * 100
    x = bin_edges[:-1] + 25  # bin centers
    ax.plot(x, percentages, label='Total Latency', linewidth=1, color='orange')

    ax.set_xlabel('Latency (ns)')
    ax.set_ylabel('Percentage of Total Latency (%)')
    ax.set_yscale('symlog')
    ax.set_yticks([0.01, 1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
    ax.set_title(f'{prefix} Total Latency Percentage Distribution (50ns bins)')
    ax.set_xticks(np.arange(0, max_lat + 100, 100))
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / f'{prefix}_8_total_latency_percentage.png', dpi=150)
    plt.close()

def plot_combined_latency_analysis(read_latency_df, write_latency_df, total_latency_df, output_dir, prefix="4DIE"):
    """
    合并 latency 分析图：左边 latency vs 时间，右边 latency 比例分布
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))

    # 左图：latency vs 时间 scatter
    if not read_latency_df.empty:
        ax1.scatter(read_latency_df['timestamp_req'], read_latency_df['latency'], alpha=0.6, s=1, color='blue', label='Read')
    if not write_latency_df.empty:
        ax1.scatter(write_latency_df['timestamp_req'], write_latency_df['latency'], alpha=0.6, s=1, color='red', label='Write')
    ax1.set_xlabel('Request Time (Cycles)')
    ax1.set_ylabel('Latency (ns)')
    ax1.set_title('Latency vs Request Time')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 添加 max/min latency 标注（放在图右上角上侧）
    stats_text = ''
    if not read_latency_df.empty:
        read_max = read_latency_df['latency'].max()
        read_min = read_latency_df['latency'].min()
        stats_text += f'Read Max: {read_max:.2f} ns\nRead Min: {read_min:.2f} ns\n'
    if not write_latency_df.empty:
        write_max = write_latency_df['latency'].max()
        write_min = write_latency_df['latency'].min()
        stats_text += f'Write Max: {write_max:.2f} ns\nWrite Min: {write_min:.2f} ns'
    if stats_text:
        # 统计信息放在图右上角、图例下方（仍在图内）
        ax1.text(0.98, 0.98, stats_text.strip(), transform=ax1.transAxes, fontsize=10,
                verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # 右图：latency 比例分布 (纵坐标 latency, 横坐标百分比)
    if not total_latency_df.empty:
        max_lat = total_latency_df['latency'].max()
        bins = np.arange(0, max_lat + 50, 50)
        bin_centers = bins[:-1] + 25

        # 读 latency
        if not read_latency_df.empty:
            hist, _ = np.histogram(read_latency_df['latency'], bins=bins)
            percentages = (hist / hist.sum()) * 100
            ax2.plot(percentages, bin_centers, label='Read Latency', linewidth=1, color='blue')

        # 写 latency
        if not write_latency_df.empty:
            hist, _ = np.histogram(write_latency_df['latency'], bins=bins)
            percentages = (hist / hist.sum()) * 100
            ax2.plot(percentages, bin_centers, label='Write Latency', linewidth=1, color='red')

    ax2.set_xlabel('Percentage of Total Latency (%)')
    ax2.set_xscale('symlog')
    ax2.set_xticks([0, 1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
    ax2.set_ylabel('Latency (ns)')
    ax2.set_title('Latency Percentage Distribution (50ns bins)')
    ax2.set_yticks(np.arange(0, max_lat + 100, 100))
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / f'{prefix}_9_combined_latency_analysis.png', dpi=150, bbox_inches='tight')
    plt.close()

def main(input_dir, output_dir, window_cycles=1000, freq=2.2, disable_cycle_cutoff=False, y_axis_tick_threshold=5000, y_tick_step=100):
    """
    主函数
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    print("=== Simple Read/Write Analysis ===")
    print(f"Input directory: {input_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Window: {window_cycles} cycles")
    print(f"Frequency: {freq} GHz")

    # 查找文件
    rdreq_files = list(input_path.glob('*rdreq*.vec'))
    rdret_files = list(input_path.glob('*rdret*.vec'))
    wrreq_files = list(input_path.glob('*wrreq*.vec'))
    wrret_files = list(input_path.glob('*wrret*.vec'))

    print(f"Found {len(rdreq_files)} rdreq, {len(rdret_files)} rdret, {len(wrreq_files)} wrreq, {len(wrret_files)} wrret files")

    # 允许只有读或只有写的场景：
    # - 只有 rdreq/rdret：只分析读
    # - 只有 wrreq/wrret：只分析写
    # - 两者都没有：直接返回
    has_read = bool(rdreq_files) and bool(rdret_files)
    has_write = bool(wrreq_files) and bool(wrret_files)

    if not has_read and not has_write:
        print("Missing files! Need at least read (rdreq+rdret) or write (wrreq+wrret) files.")
        return
    if not has_read:
        print("Warning: No read files found. Will only process write operations.")
    if not has_write:
        print("Warning: No write files found. Will only process read operations.")

    # 获取截至 cycle
    if disable_cycle_cutoff:
        print("Cycle cutoff disabled by user.")
        max_cycle_cutoff = None
    else:
        max_cycle_cutoff = get_max_cycle_cutoff(input_dir)

    # 解析文件（在解析时就应用 cycle cutoff 过滤，避免创建大型 DataFrame）
    print("\nParsing files...")
    if max_cycle_cutoff is not None:
        print(f"Applying cycle cutoff during parsing: {max_cycle_cutoff}")
    
    if has_read:
        read_req_df = parse_rdreq_files(rdreq_files, max_cycle_cutoff)
        read_ret_df = parse_rdret_files(rdret_files, max_cycle_cutoff)
    else:
        read_req_df = pd.DataFrame()
        read_ret_df = pd.DataFrame()

    if has_write:
        write_req_df = parse_wrreq_files(wrreq_files, max_cycle_cutoff)
        write_ret_df = parse_wrret_files(wrret_files, max_cycle_cutoff)
    else:
        write_req_df = pd.DataFrame()
        write_ret_df = pd.DataFrame()

    print(f"Parsed {len(read_req_df)} read requests, {len(read_ret_df)} read returns")
    print(f"Parsed {len(write_req_df)} write requests, {len(write_ret_df)} write returns")

    # 基于解析后的数据重新判断是否有有效数据
    has_read_data = not read_req_df.empty and not read_ret_df.empty
    has_write_data = not write_req_df.empty and not write_ret_df.empty

    # 计算
    print("\nCalculating...")
    read_latency_df = calculate_latency(read_req_df, read_ret_df, freq) if has_read_data else pd.DataFrame()
    write_latency_df = calculate_write_latency(write_req_df, write_ret_df, freq) if has_write_data else pd.DataFrame()
    
    if args.split_by_core:
        # 获取所有存在的 core
        cores = set()
        if not read_req_df.empty and 'core' in read_req_df.columns:
            cores.update(read_req_df['core'].unique())
        if not write_req_df.empty and 'core' in write_req_df.columns:
            cores.update(write_req_df['core'].unique())
        
        sorted_cores = sorted(list(cores))
        print(f"Splitting analysis by core: {sorted_cores}")
        
        for core in sorted_cores:
            print(f"\nProcessing {core}...")
            core_output_dir = output_path / core
            core_output_dir.mkdir(exist_ok=True)
            
            # 筛选当前 core 的数据
            core_read_req = read_req_df[read_req_df['core'] == core] if not read_req_df.empty and 'core' in read_req_df.columns else pd.DataFrame()
            core_read_ret = read_ret_df[read_ret_df['core'] == core] if not read_ret_df.empty and 'core' in read_ret_df.columns else pd.DataFrame()
            core_write_req = write_req_df[write_req_df['core'] == core] if not write_req_df.empty and 'core' in write_req_df.columns else pd.DataFrame()
            core_write_ret = write_ret_df[write_ret_df['core'] == core] if not write_ret_df.empty and 'core' in write_ret_df.columns else pd.DataFrame()
            
            core_read_latency = read_latency_df[read_latency_df['core'] == core] if not read_latency_df.empty and 'core' in read_latency_df.columns else pd.DataFrame()
            core_write_latency = write_latency_df[write_latency_df['core'] == core] if not write_latency_df.empty and 'core' in write_latency_df.columns else pd.DataFrame()
            
            # 计算当前 core 的带宽
            core_read_bw = calculate_bandwidth(core_read_ret, window_cycles) if not core_read_ret.empty else pd.DataFrame(columns=['time_bin', 'bandwidth_bytes_per_cycle'])
            core_write_bw = calculate_write_bandwidth(core_write_req, window_cycles) if not core_write_req.empty else pd.DataFrame(columns=['time_bin', 'bandwidth_bytes_per_cycle'])
            
            # 计算当前 core 的总带宽
            core_total_bw = pd.DataFrame()
            if not core_read_bw.empty:
                core_total_bw = core_read_bw[['time_bin', 'bandwidth_bytes_per_cycle']].copy()
            if not core_write_bw.empty:
                if core_total_bw.empty:
                    core_total_bw = core_write_bw[['time_bin', 'bandwidth_bytes_per_cycle']].copy()
                else:
                    core_total_bw = pd.merge(core_total_bw, core_write_bw[['time_bin', 'bandwidth_bytes_per_cycle']], on='time_bin', how='outer', suffixes=('', '_write')).fillna(0)
                    core_total_bw['bandwidth_bytes_per_cycle'] += core_total_bw.get('bandwidth_bytes_per_cycle_write', 0)
                    core_total_bw = core_total_bw[['time_bin', 'bandwidth_bytes_per_cycle']]
            
            if not core_total_bw.empty:
                core_total_bw = core_total_bw.sort_values('time_bin')
            
            # 生成当前 core 的合并图
            core_total_latency = pd.concat([core_read_latency, core_write_latency], ignore_index=True)
            
            plot_combined_overview(core_read_latency, core_write_latency, core_total_latency, 
                                 core_read_bw, core_write_bw, core_total_bw, 
                                 core_read_req, core_write_req, 
                                 core_output_dir, window_cycles, freq, y_axis_tick_threshold, y_tick_step, prefix=core)
            print(f"[OK] {core} combined overview plot saved")

    else:
        # 原有逻辑：合并所有 core
        read_bw_df = calculate_bandwidth(read_ret_df, window_cycles) if has_read_data else pd.DataFrame(columns=['time_bin', 'bandwidth_bytes_per_cycle'])
        write_bw_df = calculate_write_bandwidth(write_req_df, window_cycles) if has_write_data else pd.DataFrame(columns=['time_bin', 'bandwidth_bytes_per_cycle'])

        print(f"Calculated read latency for {len(read_latency_df)} transactions")
        print(f"Calculated write latency for {len(write_latency_df)} transactions")

        # 生成图表
        print("\nGenerating plots...")
        die_output_dir = output_path / "4DIE"
        die_output_dir.mkdir(exist_ok=True)

        # 计算总带宽
        total_bw_df = pd.DataFrame()
        if not read_bw_df.empty:
            total_bw_df = read_bw_df[['time_bin', 'bandwidth_bytes_per_cycle']].copy()
        if not write_bw_df.empty:
            if total_bw_df.empty:
                total_bw_df = write_bw_df[['time_bin', 'bandwidth_bytes_per_cycle']].copy()
            else:
                total_bw_df = pd.merge(total_bw_df, write_bw_df[['time_bin', 'bandwidth_bytes_per_cycle']], on='time_bin', how='outer', suffixes=('', '_write')).fillna(0)
                total_bw_df['bandwidth_bytes_per_cycle'] += total_bw_df.get('bandwidth_bytes_per_cycle_write', 0)
                total_bw_df = total_bw_df[['time_bin', 'bandwidth_bytes_per_cycle']]

        # 按 time_bin 排序
        if not total_bw_df.empty:
            total_bw_df = total_bw_df.sort_values('time_bin')

        # 新增 latency 比例图
        total_latency_df = pd.concat([read_latency_df, write_latency_df], ignore_index=True)

        plot_combined_overview(read_latency_df, write_latency_df, total_latency_df, read_bw_df, write_bw_df, total_bw_df, read_req_df, write_req_df, die_output_dir, window_cycles, freq, y_axis_tick_threshold, y_tick_step)
        print("[OK] 4DIE combined overview plot saved")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Simple Read/Write Analysis')
    parser.add_argument('--input', required=True, help='Input directory containing .vec files')
    parser.add_argument('--output', required=True, help='Output directory for plots')
    parser.add_argument('--window', type=int, default=1000, help='Time window in cycles (default: 1000)')
    parser.add_argument('--freq', type=float, default=2.2, help='Frequency in GHz (default: 2.2)')
    parser.add_argument('--y-tick-threshold', type=float, default=5000, help='Y-axis tick threshold for latency plots (default: 5000)')
    parser.add_argument('--disable-cycle-cutoff', action='store_true', help='Disable cycle cutoff filtering (default: enabled)')
    parser.add_argument('--y-tick-step', type=float, default=100, help='Y-axis tick step (default: 100)')
    parser.add_argument('--split-by-core', action='store_true', help='Generate separate plots for each core instead of a combined 4DIE plot')

    args = parser.parse_args()
    main(args.input, args.output, args.window, args.freq, args.disable_cycle_cutoff, args.y_tick_threshold, args.y_tick_step)
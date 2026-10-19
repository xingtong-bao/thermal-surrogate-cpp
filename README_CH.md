# thermal-surrogate-cpp

[![ci](https://github.com/xingtong-bao/thermal-surrogate-cpp/actions/workflows/ci.yml/badge.svg)](https://github.com/xingtong-bao/thermal-surrogate-cpp/actions/workflows/ci.yml)
![C++17](https://img.shields.io/badge/C%2B%2B-17-blue)
![ONNX Runtime 1.23.2](https://img.shields.io/badge/ONNX%20Runtime-1.23.2-lightgrey)
![License MIT](https://img.shields.io/badge/license-MIT-green)
[English](README.md)

面向芯片热仿真神经网络代理模型的 C++17 推理库，以公开的 IC-ThermBench level4 U-Net（7 张输入图 → 64×64 稳态温度场）为对象。库按书面合同校验输入、检测分布外的设计、给每次预测附带标定过的误差界、在模型不可信时回退到物理求解器，并提供更新、把关和发布新模型版本的工具链。本文件里的每个数字都在本仓库实测，并链接到保存它的文件。

![架构](docs/figures/architecture.png)

## 功能

- **合同驱动的校验** — 形状、数据类型、逐通道训练范围、配置新颖度检查（规则 I1–I5），状态码在 C++、JSON Schema 和 C 接口中一致。[contract/](contract/)
- **回退调度器** — 代理 → 外部求解器（HotSpot 子进程，带超时）→ 进程内有限差分求解器；每个回答都注明由哪个后端产生。[docs/integration_design.md](docs/integration_design.md)
- **逐预测误差界** — 共形预测给出场 MAE 和热点误差的 90% 界，按预测温升分档。
- **物理精确的 what-if** — 全芯片功率缩放由热传导的线性直接给出，不经过模型。
- **模型生命周期** — 用求解器答案或增强数据微调、导出 ONNX、对拍、重标定误差界、在全量测试集上跑发布门槛的脚本。
- **与模型无关** — U-Net、U-FNO（谱层改写为矩阵乘后导出）和集成模型通过同一 API 加载。
- **绑定** — 不透明句柄的 C 接口（`ts_c_api.h`）、pybind11 模块。
- **工程** — 零拷贝输入、调用方拥有输出、线程安全预测（ThreadSanitizer 验证）、哈希钉死的依赖、38 个测试、Linux CI。

## 结果

测量环境：Apple M4（4 性能核 + 6 能效核，16 GB），ONNX Runtime 1.23.2 CPU。"已知芯片" = level4 测试集（训练过的 10 个芯片家族）；"未见芯片" = level5（从未训练的 5 个家族）。

**代理模型本身**

| 指标 | 数值 | 文件 |
|---|---|---|
| 已知芯片 MAE，50 样本提交子集 / 全部 3000 | 1.05 K / 1.83 K（作者 1.83 K） | [baseline_notes.md](docs/results/baseline_notes.md) |
| 未见芯片 MAE（50） | 14.8 K | [backend_comparison.csv](docs/results/backend_comparison.csv) |
| 合同门槛（基线 × 1.1）：MAE / 最差 5% / 热点中位 | 2.012 / 3.792 / 0.952 K | [contract.md §4](contract/contract.md) |
| 延迟 p50，1 / 4 / 8 线程 | 79.9 / 30.3 / 27.2 ms | [latency_single.json](docs/results/latency_single.json) |
| 吞吐，batch 1 / 8 / 32 / 64 | 48.8 / 33.8 / 34.3 / 34.6 样本/秒 | [throughput.json](docs/results/throughput.json) |
| 更大网格 128² / 256² / 512² | 117 ms / 468 ms / 1.87 s（每千格 7.14 ms） | [dynamic_size.csv](docs/results/dynamic_size.csv) |
| PyTorch ↔ ONNX Runtime（100 样本）/ ONNX Runtime Python ↔ C++（10 对） | 最大 8.2e-4 K / 0 K | [parity_notes.md](docs/results/parity_notes.md) |

**守卫与回退**

| 指标 | 数值 | 文件 |
|---|---|---|
| 新颖度检查（I5）：未见芯片拒绝率 / 已知芯片误拒率 | 100% / 1% | [layout_novelty.json](docs/results/layout_novelty.json) |
| 仅逐通道范围检查（I4）：未见芯片拒绝率 | 20% | [extrapolation_notes.md](docs/results/extrapolation_notes.md) |
| 共形 90% MAE 界（按预测温升）/ 留出集覆盖率 | 1.6–3.7 K / 1.00 | [conformal_notes.md](docs/results/conformal_notes.md) |
| 调度器 + HotSpot，未见芯片 MAE | 3.4 K（仅代理 14.8 K） | [backend_comparison_notes.md](docs/results/backend_comparison_notes.md) |
| 50/50 已知/未见混合流的混合 MAE | 2.2 K（仅代理 7.9 K） | [fallback_curve.json](docs/results/fallback_curve.json) |
| 参考求解器：差分（进程内）/ HotSpot（子进程），已知芯片 MAE，延迟 | 12.8 K，34 ms / 13.4 K，2.7 s | [fd_solver_validation.json](docs/results/fd_solver_validation.json)，[hotspot_validation.json](docs/results/hotspot_validation.json) |
| 单会话 8 线程 × 100 次预测 | 逐位相同，TSan 无告警 | [concurrency.json](docs/results/concurrency.json) |

**同一库下的不同模型**

| 模型 | 已知芯片 MAE / 热点中位 | 未见芯片 MAE | 延迟 p50（4 线程） | 文件 |
|---|---|---|---|---|
| U-Net（发布版） | 1.05 / 0.60 K | 14.8 K | 30 ms | [models/metadata.json](models/metadata.json) |
| U-FNO | 0.60 / 0.27 K | 24.6 K | 760 ms | [model_swap_notes.md](docs/results/model_swap_notes.md) |
| U-Net + U-FNO 集成 | 0.69 / 0.27 K | 18.4 K | 745 ms | [ensemble_spread.json](docs/results/ensemble_spread.json) |

**模型更新与发布门槛**（全部 3000 条测试集；"通过" = 三条合同门槛全过）

| 候选 | 目的 | 未见芯片 MAE | 已知芯片 MAE | 热点中位 | 门槛 | 文件 |
|---|---|---|---|---|---|---|
| v1.0（发布版） | — | 15.8 K | 1.83 K | 0.87 K | 通过 | [baseline_unet_level4.json](docs/results/baseline_unet_level4.json) |
| v1.1，每块未见芯片 50 个求解器答案 | 适应新芯片 | 2.6 K | 5.8 K | — | 不过（遗忘） | [lifecycle_notes.md](docs/results/lifecycle_notes.md) |
| v1.1 + 回放 250 个训练样本 | 同上，不遗忘 | 2.8 K | 1.45 K | 1.19 K | 不过（热点） | 同上 |
| v1.4，功率增强 ±15% | 功率灵敏度 0.1 → 0.85 | 13.4 K | 2.07 K | 1.09 K | 不过（MAE、热点） | [power_aug_notes.md](docs/results/power_aug_notes.md) |

**发布版模型的特性刻画** — 功率灵敏度为求解器响应的 0.1–0.3（HotSpot 验证），散热灵敏度斜率为求解器的 2 倍，热点位置精确命中 3.7%，误差随功率密度和温升增长：[docs/failure_analysis.md](docs/failure_analysis.md)、[sensitivity_notes.md](docs/results/sensitivity_notes.md)、[error_buckets_notes.md](docs/results/error_buckets_notes.md)、[exploration_notes.md](docs/results/exploration_notes.md)。

![结果](docs/figures/results_summary.png)

## 快速开始

```bash
# 依赖：cmake >= 3.24、ninja、C++17 编译器、python3（fixture）、网络（ONNX Runtime 与模型）
git clone https://github.com/xingtong-bao/thermal-surrogate-cpp && cd thermal-surrogate-cpp
gh release download models-v1 --dir models          # model.onnx + model_dynamic.onnx（各 20 MB）
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build
ctest --test-dir build --output-on-failure        # 38 个测试，约 20 秒
./build/ts_demo --info                            # 运行时版本 + 模型 IO
./build/c_caller models contract tests/fixtures/sample_input.npy   # C 接口示例
```

没有 `gh` 时：从 [models-v1 release](https://github.com/xingtong-bao/thermal-surrogate-cpp/releases/tag/models-v1) 下载两个文件并按 [models/metadata.json](models/metadata.json) 校验，或按 [models/README.md](models/README.md) 重新生成。

## 用法

```cpp
#include "thermal_surrogate/dispatcher.h"
#include "thermal_surrogate/fd_solver.h"
using namespace thermal_surrogate;

Validator validator("contract/stats.json", "models/config_bank.npy");
Predictor predictor("models/model.onnx", "models/metadata.json");
std::vector<std::unique_ptr<ISolverBackend>> chain;
chain.push_back(std::make_unique<SurrogateBackend>(predictor));
chain.push_back(std::make_unique<FdSolverBackend>(load_fd_params("models/fd_solver_params.json")));
Dispatcher dispatcher(validator, std::move(chain));
dispatcher.set_conformal(ConformalTable("contract/conformal.json"));   // 可选：误差界

Output out;                                   // 构造之后永不抛异常
dispatcher.solve(Input{data /* 64*64*1*7 个 float */, kInputShape}, out);
if (out.status.ok()) use(out.field, out.tmax, out.mae_bound_90_K, out.source, out.oob_flag);
dispatcher.solve_power_scaled(Input{data, kInputShape}, 1.1, out);   // 所有芯粒功率 +10%
```

输入是调用方缓冲区的非拥有视图；`predict_into` 把温度场写入调用方自己的缓冲区；`predict_batch` 用动态批量图跑多个样本。C 接口：[examples/c_caller.c](examples/c_caller.c)。Python：[python/test_binding.py](python/test_binding.py)。所有权、生命周期、线程、错误与版本规则：[docs/integration_design.md](docs/integration_design.md)。

## 合同

[contract/contract.md](contract/contract.md) 定义接口；[contract/schema.json](contract/schema.json) 是其 JSON Schema 形式；[contract/stats.json](contract/stats.json) 和 [contract/conformal.json](contract/conformal.json) 存库读取的实测常量。

| 规则 | 条件 | 违反时 |
|---|---|---|
| I1–I3 | float32、有限值、形状 (64,64,1,7)、7 通道 | `InvalidInput` / `ShapeMismatch` / `ChannelMismatch`（硬拒绝） |
| I4 | 每个通道在训练范围内 | `OutOfDistribution` → 回退 |
| I5 | 配置向量到最近训练芯片的距离 ≤ 0.091 个标准差单位 | `OutOfDistribution` → 回退 |
| F1–F4 | 后端按顺序尝试；失败跳过；报告最后一个失败 | `source` 注明回答的后端 |
| §4 | 精度门槛以全量测试集为准；提交的 50 样本子集为冒烟测试 | 阻止发布 |
| §9 | 全芯片功率缩放由基准预测直接推出 | — |

## 架构

```
输入 (64,64,1,7)
   │
   ▼
Validator ── I1–I3 拒绝 ──► Status
   │ I4 / I5
   ▼
Dispatcher ──► SurrogateBackend（ONNX Runtime）        约 30 ms
           ──► ExternalReferenceBackend（HotSpot）     约 2 s，分布外或失败时
           ──► FdSolverBackend（二维 SOR）             约 35 ms，最后兜底
   │
   ▼
输出：field、tmax、tmax_index、误差界、source、oob_flag、latency、status
```

公开头文件不含任何 ONNX Runtime 类型；运行时藏在 pimpl 后面。依赖（ONNX Runtime、GoogleTest、nlohmann/json、pybind11）由 CMake 拉取并钉死版本，运行时包另钉 SHA-256。

## 目录

```
contract/        contract.md, schema.json, stats.json, conformal.json
include/, src/   Predictor, Validator, Dispatcher, ConformalTable, 差分求解器, 外部后端, C 接口
apps/ts_demo     --info --parity --fd-calibrate --fd-validate --compare
bench/           延迟、吞吐、动态网格尺寸
tests/           GoogleTest 套件 + fixture（对拍样本对、50 样本子集、校验器期望值）
tools/           Python：数据拉取、基线、统计、分析、ONNX 导出、对拍、HotSpot 包装、生命周期
python/          pybind11 模块 + 检查脚本
models/          metadata.json, config_bank.npy, fd_solver_params.json（ONNX 文件走 GitHub release）
models_*/        候选版本（微调、增强、U-FNO、集成）各自的元数据、配置库、误差界
docs/results/    全部实测数字及笔记；docs/failure_analysis.md；docs/integration_design.md
```

## 复现数字

| 步骤 | 命令 | 产物 |
|---|---|---|
| 数据与权重（从公开压缩包做 HTTP Range 抽取） | `python tools/fetch_assets.py [--checkpoint-members level4_UNet/,level4_UFNO/]` | `docs/results/data_manifest.json` |
| 基线 | `tools/run_baseline.sh --device cpu`；`tools/baseline_per_sample.py` | `baseline_*.json` |
| 统计量、fixture | `python tools/export_stats.py --bench ../external/ThermalBench` | `contract/stats.json`、`tests/fixtures/` |
| 模型特性刻画 | `tools/sensitivity.py`、`error_buckets.py`、`extrapolation.py`、`layout_novelty.py`、`fallback_curve.py`、`power_gain_verification.py`、`exploration_demo.py` | `docs/results/*` |
| ONNX 导出、对拍 | `tools/export_onnx.py`、`parity_check.py`、`export_ufno_onnx.py`、`export_ensemble_onnx.py` | `models*/`、`parity_python.json`、`tests/fixtures/parity/` |
| C++ 对拍与基准 | `ts_demo --parity`、`bench_latency`、`bench_throughput`、`bench_dynamic_size` | `parity_cpp.json`、`latency_*.json`、`throughput.json`、`dynamic_size.csv` |
| 参考求解器、后端对比 | `ts_demo --fd-calibrate` / `--fd-validate`；`ts_demo --compare ... python tools/reference_cli.py` | `fd_solver_*.json`、`backend_comparison*.csv` |
| 误差界 | `python tools/conformal.py` | `contract/conformal.json` |
| 模型更新 | `finetune_fewshot.py` + `lifecycle_v11.py`；`finetune_power_aug.py` + `lifecycle_poweraug.py` | `lifecycle_*.json`、`power_aug_*.json`、`models_v1.*/` |
| TSan 并发 | `cmake -B build-tsan -DTS_SANITIZE=thread && ./build-tsan/ts_tests --gtest_filter='Concurrency.*'` | `concurrency.json` |

Python 工具需要 IC-ThermBench 检出在 `../external/ThermalBench` 及其环境（Python 3.10、torch 2.0.1、onnxruntime 1.23.2）。C++ 构建只需要 CMake、编译器和下载钉死版本 ONNX Runtime 包的网络。

## 测试

`ctest` 运行 38 个测试：`Skeleton.*`（运行时版本、模型 IO）、`Predictor.*`（构造、拒绝路径、对拍、零拷贝、批量）、`Validator.*`（每个状态码、两个 fixture、特征提取）、`Fixture.*`（调度器规则 F1–F4）、`FdSolver.*`、`ExternalBackend.*`（协议、退出码、超时）、`Concurrency.*`、`AccuracyGate.*`（合同门槛、schema 拼写）、`Conformal.*`、`PowerWhatIf.*`。

## 已知局限

- 单芯粒功率变化：发布版模型的灵敏度为物理响应的 0.1；增强候选达到 0.85 但被发布门槛拦下。全芯片缩放精确（§9）。
- 规则 I5 在本基准上实际是十种训练芯片配置的白名单；集成分歧是一个独立信号。
- 两个参考求解器都不能复现数据集的生成器（HotSpot 已知芯片差 13 K、未见芯片 3.4 K；差分 11–18 K）。
- 热点位置不可复现（精确命中 3.7%）；`tmax_index` 仅供参考。
- 仅 CPU 执行提供器（macOS arm64、Linux x64）。

## 许可与致谢

代码：MIT（[LICENSE](LICENSE)）。模型权重与数据：IC-ThermBench（CC BY 4.0），未作修改；导出的 ONNX 文件衍生自其发布的 level4 U-Net 与 U-FNO 权重。HotSpot 作为外部工具按其自身许可使用。ONNX Runtime 为微软的预编译包。

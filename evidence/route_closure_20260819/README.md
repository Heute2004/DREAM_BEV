# 路由证据闭环实验报告

## 协议

- 数据：IF-only 87 Train / 34 Development / 8 Locked Test 合同。
- 本轮全部统计、训练和阈值无关评估只使用 Train/Development；Locked Test 未读取。
- 训练随机种子：3407。按用户最终要求，不汇总其他 seed。
- 冻结组件：DREAM-BEV 基座、五个既有专家、策略 trace、融合/guard 合同和 evaluator。
- 独立统计单位：完整 sequence；bootstrap 在五条 route 内按完整 sequence 分层重采样。
- 论文工程未修改；本报告仅供决定哪些实验进入论文。

## 实验一：路由分配是否优于非条件控制

所有非 identity 臂均从同一组五路 forced-route 输出组合。Learned route 按冻结的因果策略 trace 选择；策略尚不可用的 warmup 帧严格使用 DREAM-BEV identity。

| 方法 | AP@0.25 | AP@0.30 | AP@0.50 | Route-macro AP@0.30 | Mean (m) | P90 (m) |
|---|---:|---:|---:|---:|---:|---:|
| DREAM-BEV identity | 0.5725 | 0.7271 | 0.8989 | 0.7521 | 0.3839 | 0.4395 |
| 最佳固定专家（weak-return） | 0.6188 | 0.7546 | 0.8991 | 0.7780 | 0.3692 | 0.4384 |
| 静态均匀五专家混合 | 0.6593 | 0.7700 | 0.9047 | 0.7944 | 0.3298 | 0.4132 |
| Learned causal route | 0.6605 | 0.7742 | 0.9055 | 0.7963 | 0.3632 | 0.4273 |
| Ledger-matched route | 0.6604 | 0.7742 | 0.9055 | 0.7963 | 0.3632 | 0.4273 |

补充控制：100 组 held random route 的 pooled AP@0.30 为 `0.7494 +/- 0.0053`，范围 `[0.7355, 0.7612]`。

完整 sequence 分层 bootstrap：

- Learned 相对最佳固定专家：route-macro `+1.83 pp`，95% CI `[+0.58, +2.92]`。
- Learned 相对静态均匀混合：route-macro `+0.19 pp`，95% CI `[-1.23, +1.36]`。
- Learned 与 ledger-matched：`-0.001 pp`，可视为相同；非 warmup 帧路线一致率为 `99.66%`。

结论：学习路由显著优于任一固定专家和随机路由；它以单个活动专家的计算量达到并略高于同时运行五个专家的静态混合，但对静态混合的精度差异不显著。因此最稳的主张是条件计算效率与固定路由优势，不是对全池并行混合的显著精度碾压。

## 实验二：异构专家是否优于参数匹配的通用专家

| 方法 | Trainable params | AP@0.25 | AP@0.30 | AP@0.50 | Route-macro AP@0.30 | Mean (m) | P90 (m) |
|---|---:|---:|---:|---:|---:|---:|---:|
| DREAM-BEV identity | 0 | 0.5725 | 0.7271 | 0.8989 | 0.7521 | 0.3839 | 0.4395 |
| 通用专家，活动容量匹配 | 72,652 | 0.6088 | 0.7467 | 0.8970 | 0.7741 | 0.3744 | **0.4247** |
| 通用专家，专家池总容量匹配 | 363,253 | 0.6216 | 0.7535 | 0.8973 | 0.7775 | 0.3726 | 0.4291 |
| 路由异构专家池 | 72,650 active / 363,248 bank | **0.6264** | **0.7617** | **0.9041** | **0.7863** | **0.3710** | 0.4350 |

异构池相对活动容量通用专家：AP@0.30 `+1.50 pp`，route-macro `+1.22 pp`。相对总池容量通用专家：AP@0.30 `+0.82 pp`，route-macro `+0.88 pp`。

结论：异构设计在严格 AP、宽松 AP 和平均误差上优于两个通用控制，即使后者拥有完整专家池参数量；但异构池的 P90 略差。因此该实验支持“异构专门化提高命中与平均精度”，不支持“所有尾部指标全面占优”。

## 实验三：路线标签是否对应可观测雷达签名

- 34 条完整 Development sequence、9,969 个评分帧。
- 11 维冻结 RA/RA-MTI 签名在 leave-one-sequence-out 最近质心分类中达到 `70.59%` accuracy、`67.56%` macro-F1。
- 机制严重度分数与专家收益的总体 Spearman 相关仅为 `0.1013`。

| Route | Sequence 数 | 严重度与专家收益 Spearman rho |
|---|---:|---:|
| open indoor LOS | 7 | -0.1429 |
| indoor clutter | 3 | 1.0000 |
| open outdoor weak return | 8 | 0.4762 |
| outdoor dry complex | 3 | 0.5000 |
| outdoor wet return | 13 | 0.0659 |

结论：路线在原始雷达统计中具有中等可分性，足以反驳“纯人工任意分组”；但统一严重度分数不能稳定预测专家收益，且两个 route 仅有 3 条 sequence。该项应定位为描述性/补充材料证据，不能宣称发现了完备物理故障机制或单调严重度规律。

## 实验四：无状态重置的因果路线切换

构造全部 20 个有序 route pair，每个 episode 300 帧，边界为第 150 帧；16 帧雷达窗口跨越边界，专家因果状态不重置。策略结果固定延迟 32 帧，并按 30 帧 segment hold。该实验是拼接式压力诊断，不等同于自然连续天气变化。

| 路由臂 | Overall AP@0.30 | 稳定后 AP@0.30 | 稳定后 P90 (m) | 切换成功 | 平均切换延迟 |
|---|---:|---:|---:|---:|---:|
| Learned delayed | 0.7626 | **0.7951** | 0.3513 | 20/20 | 33.5 帧 |
| Ledger delayed | 0.7625 | 0.7950 | 0.3513 | 20/20 | 32.0 帧 |
| Ledger instant upper bound | **0.7864** | 0.7950 | 0.3513 | 20/20 | 0 帧 |
| Best fixed expert | 0.7676 | 0.7588 | **0.3512** | 4/20 | 不适用 |

Learned policy 在 19/20 个 transition 上于首个合法可用时刻切换；`open_indoor_los -> open_outdoor_weak_return` 晚一个 30 帧策略段（62 帧）。

结论：冻结策略能在无状态重置条件下完成全部路线切换，稳定后明显优于固定专家；32 帧因果等待会降低全 episode 指标，说明延迟代价真实存在。该结果支持“低频因果切换能力”，但只应称为拼接诊断，不应扩展为已验证自然动态环境。

## 证据等级

| 实验 | 证据等级 | 建议用途 |
|---|---|---|
| Learned / fixed / random / static route | 强，但有静态混合边界 | 主文路由表；强调单专家活动成本与固定路由优势 |
| 参数匹配通用专家 | 中强，有 P90 取舍 | 主文或补充材料；支持异构专门化，不写全面支配 |
| 雷达机制签名 | 探索性 | 补充材料；只证明 route 可观测，不证明完整物理机制 |
| 无重置路线切换 | 中强诊断 | 主文小表或补充；明确为拼接压力测试 |

## 工件

- `canonical/route_allocation_final/route_allocation_report.json`
- `canonical/universal_summary_seed3407/universal_expert_report.json`
- `canonical/mechanism_signatures/mechanism_signature_report.json`
- `canonical/route_transitions/evaluation/route_transition_report.json`
- `route_closure_evidence_seed3407.tar.gz`

归档 SHA256：`6e4e7697dac86c4a42e3ff1cdc8112d8272509d7dcbba7357ed89282cb03811f`。

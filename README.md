# 四视角初态 → Genesis 穿衣仿真复建

这是根据四张初始图和任务描述独立建立的**近似模型**，不是原仿真工程或物理参数的恢复。没有读取原仿真的资产、动作、参数、数据，也没有复用工作区里已有的其他复建工程。

## 四张原始输入图片

以下四张图是用户提供的同一初始状态的不同视角。文件保存在 [references/](references/README.md)，与原始 PNG 逐字节一致。

| 图 1：斜前方 | 图 2：俯视 |
|---|---|
| [![图 1：斜前方](references/view_1.png)](references/view_1.png) | [![图 2：俯视](references/view_2.png)](references/view_2.png) |
| 图 3：侧面 | 图 4：背面 |
| [![图 3：侧面](references/view_3.png)](references/view_3.png) | [![图 4：背面](references/view_4.png)](references/view_4.png) |

## 本次交付结果

默认配置对应已实际完成的 `outputs/wrap_v13`。Genesis CPU 计算了 17 秒动作：取布、抬起、从手掌端穿袖、绕到背部闭合魔术贴、释放夹爪并撤离。最终两个袖孔均可靠地穿过对应手臂，56 对黏合连接保持；14.6–17.0 秒的所有保存样本连续满足松手保持条件，共 2.4 秒。

- 左袖首次通过为 9.6 秒、右袖为 9.9 秒；交点参数从手掌端外侧进入手臂线段，区别于旧的不连通肩部模型可能出现的错误近端进入。
- 12.4–14.5 秒拉紧阶段，保守的肩—掌线段判据存在失败样本；释放后衣片下落，双袖从 14.6 秒起重新持续通过。报告保留了这些中间失败，不表示全程每个样本都通过。
- 使用实际手臂椭球轴做独立复核时，双袖在 9.9–17.0 秒的所有保存样本都通过；这说明中间结果对肩根端点定义敏感。正式标准和阈值保持原样，复核没有替代它们。
- 全程最大的边长比第 99 百分位为 1.348，粒子对躯干椭球的最大径向侵入估计约 1.90 mm。这是本近似离散模型的实际误差和变形，不是恢复出的材料实验结果。
- 在独立开启机器人自身碰撞的场景中，检查了 851 个规划姿态，以及 171 个实际保存姿态；均未检出机器人与熊、桌面、另一台机器人或自身非相邻连杆的接触。检查是离散采样，不是连续时间证明。
- 魔术贴全长 16 cm 的每个 2 cm 分段都有有效黏合点，底缘与顶缘的最近接缝间距分别约 7.95 mm、9.00 mm。它是全高多点黏合，不代表每平方毫米贴合；约 9 mm 的粒子接触尺度限制了细节。
- 29 项几何、判据、黏合与摩擦单元检查通过。关节路径规划器已重新运行，生成命令与交付路径逐位相同。

观看 `outputs/wrap_v13/simulation.mp4` 的整体动作，或 `outputs/wrap_v13/replay/overview.mp4` 的正面/背面近景。定量结果见该目录中的 `assessment.json`、`validation_timeline.png`、`actual_joint_collision_report.json`。黏合覆盖详见 `seam_coverage.json`。粒子位置、机器人关节和时间保存在 `trajectory.npz`。

## 运行

本机直接运行：

```bash
bash run.sh --video
```

只生成初态五视角图片：

```bash
bash run.sh --initial-only --output initial
```

无渲染运行物理计算：

```bash
bash run.sh --no-render --output physics_only
```

运行结果保存在 `outputs/<名称>/`。默认使用本机 `/home/chloe/Desktop/genesis-local/.venv/bin/python` 中的 Genesis 1.4.3。换环境时可设置 `GENESIS_PYTHON`。CPU 与 EGL 软件离屏渲染已经实际运行；本次环境不能使用 NVIDIA 驱动。`--backend gpu` 留作有可用驱动的环境使用，未在本次验证。

```bash
GENESIS_PYTHON=/path/to/python bash run.sh --video
```

重算交付的关节路径（默认另存，不覆盖已验证路径）：

```bash
/home/chloe/Desktop/genesis-local/.venv/bin/python planning/plan_outward_backup.py
```

该规划采用腕部朝外的姿态，避免绕背时两侧腕部碰撞。输入与初始解均保存在 `planning/`，不依赖任何旧实验输出。抬升阶段允许约 16.1 mm 的规划夹点误差；穿袖、闭合和保持阶段规划误差小于 0.5 mm。实际 PD 跟踪误差另见运行指标，不能用规划误差代替。

## 已实现的模型

- 两台 Genesis 自带 Franka Panda，使用逆运动学规划关节目标、关节 PD 控制执行，并限制关节目标的变化速度。保存的关节路径是控制命令，机器人在运行时仍由动力学求解器推进。
- 桌面、桌腿、独立程序生成的棕色玩具熊；熊的碰撞部分固定在场景中。
- 单一连通布片，两个真实袖孔，左右短边的蓝黑色和棕色魔术贴纹理。Genesis 重网格化后约 1,523 个粒子。
- Genesis 原生 `PBD.Cloth` 求解重力、拉伸、弯曲、9 mm 粒子级自碰撞以及与桌子/熊的接触；这不是连续三角面无穿透保证。
- `grasp_particle_count=1` 时，每侧夹爪通过单个粒子的理想点抓取约束固定位置，允许衣料绕夹点转动。未设置这个选项的早期实验使用小片刚性抓取区域。抓取检查实际夹点距离，松手释放约束。
- 魔术贴是新增的宏观 XPBD 黏合约束：在两条 28 mm 宽的接触区域中寻找相距不超过 20 mm、相对速度较低的点，建立受材料粒子数量限制的连接（本次可用 72 个连接槽）；间距超过断裂阈值时分离。已黏合点保持配对，不施加预定衣物形状。它不是微观钩环模型。早期实验使用按材料坐标预先配对的 `fixed` 模式，当前配置使用 `nearest` 接触模式。

物理运行没有用逐帧目标曲面替换整片布料。`replay.py` 则专门用于回放已经保存的 Genesis 状态；它不重新计算动力学。

## 从图片确定与自行假设的部分

图片支持：左右机械臂、桌面中央的熊、前方平铺的矩形布、两个开孔、两端魔术贴。图中没有可确认的管状袖子，因此本模型采用开孔衣片。

绝对尺度、相机参数、衣料物性、熊的柔顺性、魔术贴强度及动作轨迹不能从四张静态图唯一确定。本模型假设桌高 0.72 m、桌面 1.35 × 1.00 m、衣片 0.70 × 0.16 m、袖孔约 60 × 64 mm。熊由多个椭球近似。所有这些都可以在 `config.json` 和 `assets/manifest.json` 中检查。

`assets/manifest_connected.json` 使用腹部半轴 (0.120, 0.103, 0.145) m、双臂各外移 15 mm 的较饱满候选，并补齐了实际连接躯干与手臂的肩部。较饱满的腹部依据图中腹部/头部比例、衣片与孔距一起推定；最大周长约 0.702 m，不代表袖口高度处也具有相同周长。肩部与手臂、躯干均有实际网格重叠，手臂上端埋入肩部，详见 `assets/arm_connectivity_diagnostic.json`。

早期原尺寸及丰满版本分别存在约 0.673 mm、2.428 mm 的臂—躯干间隙，其暴露的肩端会允许不正确的近端穿入或滑脱。它们仅保留用于实验记录。`manifest_original_connected.json` 是修补肩部后的原尺寸候选；旧的联合碰撞网格仅适用于未修补的原尺寸模型。每轮实际使用的模型以该轮 `config.json` 为准。

本机版本的传统 PBD—刚体耦合没有实现摩擦项，因此增加了可关闭的固定熊表面库仑摩擦近似，假设静/动摩擦系数为 0.3/0.2。它只在单个明确接触上施加受法向修正量限制的切向修正；重叠部件的多重接触会保守跳过，没有额外黏附或固定布料。零摩擦一致性和倾斜平面静/动摩擦检查已通过。这仍不是精确的布—毛绒摩擦实验。熊没有软体变形或手臂关节；穿袖通过衣片移动经过固定手臂完成。夹爪抓取也没有模拟纱线、夹紧压力与真实摩擦滑移。

机械臂与布料之间只保留配置指定的抓取约束，其他机械臂表面没有参与布料碰撞。机械臂仍参与刚体间接触，但动力学运行没有启用单台机器人自身各连杆的自碰撞。这限制了控制轨迹的现实可迁移性。规划阶段单独开启自身碰撞的检查，不能替代实际动态轨迹的检查。

原生“粒子固定到连杆”抓取是单向约束：布料跟随夹爪局部位置，但布料载荷不会反馈到机械臂。机器人自身的 PD 动力学和刚体碰撞仍在计算。这不是完整的双向布料—机器人力学耦合。

配置中的 `joint_trajectory` 若指向冻结的 `.npy` 文件，实际控制会使用该文件；修改 `trajectory` 中的目标点不会自动重算关节路径。必须同时保留引用的路径文件，重新规划后再运行。多视角回放也会读取该文件以构建相同初态。

魔术贴的配对、接近阈值、速度阈值、断裂阈值均为建模假设；当前不模拟面朝向筛选、压紧过程、剥离方向或钩环纤维。

## 验证与结果解释

```bash
/home/chloe/Desktop/genesis-local/.venv/bin/python assess_run.py outputs/运行名称
/home/chloe/Desktop/genesis-local/.venv/bin/python plot_metrics.py outputs/运行名称
/home/chloe/Desktop/genesis-local/.venv/bin/python replay.py --run 运行名称
/home/chloe/Desktop/genesis-local/.venv/bin/python replay.py --run 运行名称 --close-up
/home/chloe/Desktop/genesis-local/.venv/bin/python -m unittest test_validation test_assess_run test_velcro test_contact_friction
```

`assessment.json` 记录：

- 网格是否保持单一连通面和三个边界环。
- 每只手臂的肩—掌线段是否穿过对应孔内部；孔过度非平面、自交或边界不明确时不判成功。
- 粒子对躯干椭球的径向侵入估计，以及布料边长变化。
- 黏合点数量、是否观察到松手，以及松手后保存样本是否至少持续 1 秒满足条件。
- 早期是否出现数值爆炸；出现时整轮不会被判定为成功。

这些检查是保存样本上的几何与状态检查，不是连续时间无穿透证明。侵入估计只覆盖躯干椭球的粒子，不覆盖所有三角形内部与所有身体部分。成功与否以实际输出为准，不能由视频看起来包住熊来替代判断。

时间标签表示每一步开始时的控制时间，保存的是该步执行后的状态（相差一个 0.01 s 步长）。17 秒运行保存 171 个每隔 0.1 秒的样本，10 fps 视频因此包含 17.1 秒画面。初态图片在第一次物理步之前生成。

## 文件

| 文件 | 用途 |
|---|---|
| `simulate.py`、`run.sh` | 实际 Genesis 物理运行 |
| `config.json` | 尺寸、材料、求解器和动作参数 |
| `assets/build_assets.py` | 独立生成熊和双孔布片 |
| `velcro.py` | 稀疏、可断开的黏合约束 |
| `contact_friction.py` | 可关闭的固定表面接触摩擦近似 |
| `validation.py`、`assess_run.py` | 几何与整段动作检查 |
| `plot_metrics.py` | 从实际评估结果绘制验证时间线 |
| `replay.py` | 保存状态的多视角渲染回放 |
| `requirements.txt` | 本机实际运行的依赖版本 |
| `references/` | 用户提供的四张输入图及文件哈希 |
| `outputs/` | 实际渲染、视频、粒子/关节轨迹、指标 |

`velcro.py` 使用 Genesis 1.4.3 的私有子步挂接点，但不修改引擎安装文件。升级 Genesis 时需要复核这个接口。

技术依据：[Genesis 官方 PBD 文档](https://genesis-world.readthedocs.io/en/latest/api_reference/engine/solvers/pbd_solver.html)，以及本机 Genesis 1.4.3 源码。具体调用以本机已测试版本为准。

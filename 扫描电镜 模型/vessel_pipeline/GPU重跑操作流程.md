# 扫描电镜导管腔识别 — GPU 重跑操作流程

> 2026-08-29 在本机(RTX 4070 SUPER)完整重跑验证通过:1-40 与 41-63 两批共 526 张,Excel 已导出。

---

## 一、环境(已配好,勿动)

| 项 | 值 |
|---|---|
| Python | **3.13**(`py -3.13`;默认 3.14 是 CPU 版 torch,不能用) |
| 依赖目录 | `vessel_pipeline\.pylibs`(torch 2.9.1+cu128、ultralytics 8.3.0、opencv、numpy 2.5.1 等) |
| 显卡 | NVIDIA GeForce RTX 4070 SUPER,12GB,CUDA 12.8 |
| 模型权重 | `output/models/lumen_yolov8seg/weights/best.pt` |
| 样品原图 | `H:\尉明杰\扫描电镜 模型\63个样品 总图（共计526张）`(526 张 tif) |

**统一运行前缀**(每次都要带):

```bash
cd "H:\尉明杰\扫描电镜 模型\vessel_pipeline"
export PYTHONPATH="H:/尉明杰/扫描电镜 模型/vessel_pipeline/.pylibs"
# 下面所有 python 都用 py -3.13
```

验证环境(可选):

```bash
py -3.13 -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
# 期望: 2.9.1+cu128 True NVIDIA GeForce RTX 4070 SUPER
```

---

## 二、日常闭环(照抄即可)

### 1) 自动预标(混合模式 = YOLO-seg 找腔 + 规则配对/测量)

```bash
py -3.13 run_review_batch.py 1 40 --out-dir 复核_1-40 --wipe --hybrid
py -3.13 run_review_batch.py 41 63 --out-dir 复核_41-63 --wipe --hybrid
```

- 每张约 1 秒,342 + 184 张共约 8 分钟
- 结果:`output/复核_1-40/`、`output/复核_41-63/`(左原图、右自动标注)
- 绿轮廓=导管腔;橙框=双导管区;紫十字=Ai;蓝线=TVW

### 2) 人工复核(手动步骤)

打开对照图,检查漏检/误检/双导管/TVW。修订后的图放入:

```
output/人工标定/单导管/              ← 红线圈单导管
output/人工标定/双导管+壁厚+框/      ← 蓝框圈双导管区 + 红线圈腔 + 短蓝线标壁厚
```

> ⚠️ 注意:说明文档里的 `_patch_singles_from_human.py` / `_patch_dual_from_human.py` 在当前目录中**不存在**。
> 现行等效命令是通用入库 + 重跑:

```bash
py -3.13 ingest_review_marks.py --from-review
# 或: py -3.13 continue_annotate_loop.py ingest --from-review 复核_1-40
# 入库后重跑对应批次即可让右侧跟随人工标定
```

### 3) 从复核图导出 Excel

```bash
py -3.13 run_excel_from_review.py --review 复核_1-40 --lo 1 --hi 40
py -3.13 run_excel_from_review.py --review 复核_41-63 --lo 41 --hi 63
```

- 产物:`output/excel_samples/样品1-40_扫描电镜指标_自动分析结果.xlsx` 等
- 本次结果:vessels=1781 / pairs=172(1-40);vessels=860 / pairs=112(41-63)

### 4) (可选)用复核图增量训练,下一批更准

```bash
py -3.13 export_from_review_pngs.py --review 复核_1-40 --wipe
py -3.13 train_lumen_seg.py --epochs 20 --batch 4 --model output/models/lumen_yolov8seg/weights/best.pt --device 0
```

---

## 三、本次为跑通所做的代码修改(备忘)

| 文件 | 改动 | 原因 |
|---|---|---|
| `run_review_batch.py` 等 6 个脚本 | `DATA` 从 `D:\BaiduNetdiskDownload\资料\资料` 改为 `H:\尉明杰\扫描电镜 模型` | 数据已迁移到 H 盘,D 盘原路径已不存在 |
| `run_batch_excel63.py` | Wand 模板查找改为可空;找不到模板时跳过两张说明表 | 原模板 xlsx 不在本机,原代码会在导入时报 StopIteration |
| `exclusion_rules.py` | `convexityDefects` 结果加 `reshape(-1, 4)` | 新 opencv 返回形状与旧版不同,原代码在首张图即崩溃 |

---

## 四、常见问题

| 现象 | 处理 |
|---|---|
| `No module named 'torch'` / 跑到 CPU | 检查是否用了 `py -3.13` 且 PYTHONPATH 指向 `.pylibs`;默认 `python` 是 3.14 CPU 版 |
| 下载 pytorch 官网 SSL 失败 | 网络问题;已改用 `download-r2.pytorch.org` 断点续传 |
| Excel 少两张说明表 | Wand 模板不在本机,属正常降级;把模板 xlsx 放到含"例子"的文件夹里即恢复 |
| 控制台中文乱码 | 仅显示问题,不影响文件输出 |

---

## 五、输出物一览

```
output/复核_1-40/          342 张左右对照图
output/复核_41-63/         184 张左右对照图
output/excel_samples/      样品1-40 / 样品41-63 的 xlsx + 汇总 csv
output/learned_rules.json  通用配对/排除规则(跨图有效)
```

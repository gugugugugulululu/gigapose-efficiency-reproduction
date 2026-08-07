# 中文说明

这个仓库用于干净地复现论文中的两个 LM-O 配置：

- Combined Top-3：物理裁剪到每个 target 三个 pose hypotheses，再运行 MegaPose 五次 refinement；
- Combined Object-adaptive：按固定 object policy 分配 `K=3–5 / I=3–5`，分组运行、合并结果，并重建 measured compute time。

推荐从冻结的 accelerated coarse main CSV 和 `MultiHypothesis` CSV 开始。这样可以把 refinement policy 与 coarse generation 分开检查。原始完整 Colab shell cell 保存在 `legacy_reference/`，用于审计和对照。

运行前先看英文 `README.md`，其中列出了固定 commit、checkpoint SHA256、目录结构、命令、输出文件和限制。

## References

- [GigaPose](https://github.com/nv-nguyen/gigapose)
- [GigaPose paper](https://openaccess.thecvf.com/content/CVPR2024/html/Nguyen_GigaPose_Fast_and_Robust_Novel_Object_Pose_Estimation_via_One_CVPR_2024_paper.html)
- [MegaPose](https://github.com/megapose6d/megapose6d)
- [MegaPose paper](https://arxiv.org/abs/2212.06870)
- [BOP Benchmark](https://bop.felk.cvut.cz/)
- [BOP Toolkit](https://github.com/thodan/bop_toolkit)


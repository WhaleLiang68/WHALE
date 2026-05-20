# ua-flp-LSA2

## 软著命名定稿（硕士毕业申报）

- 主申报名：大数据驱动不等面积设施布局智能优化系统 V1.0
- 副标题：基于势能曲面变平法（ELP）的优化求解方法
- 技术摘要首句：本软件面向不等面积设施布局问题，融合大数据特征建模与势能曲面变平法实现高效优化求解。
- 关键词：大数据、不等面积设施布局、智能优化、势能曲面变平法、ELP

## 软著材料文件

- `docs/softcopyright/00_命名定稿.md`
- `docs/softcopyright/01_申请表-命名信息.md`
- `docs/softcopyright/02_软件说明书-模板.md`
- `docs/softcopyright/03_源代码文档封面-模板.md`
- `docs/softcopyright/naming_manifest.json`

## 一致性检查

```bash
python tools/check_softcopyright_naming.py
```

检查项覆盖：

- 主名称在申报材料中一致
- 副标题与摘要首句已纳入模板
- 关键词已覆盖
- 版本号统一为 `V1.0`

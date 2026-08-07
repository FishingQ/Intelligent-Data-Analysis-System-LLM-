"""
训练数据集导入脚本
将结构化 Excel 和非结构化 Markdown 数据注册到系统中

用法:
    python scripts/import_datasets.py           # 导入全部
    python scripts/import_datasets.py --summary # 仅查看摘要
    python scripts/import_datasets.py --structured 金融  # 仅导入指定数据集
"""
import sys
import os
import argparse
import shutil
from pathlib import Path

# 确保项目根目录在 sys.path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from backend.training.dataset_loader import DatasetLoader


def copy_structured_excels(loader: DatasetLoader, target_dir: Path):
    """将结构化 Excel 文件复制到 data/uploads/"""
    target_dir.mkdir(parents=True, exist_ok=True)
    copied = []

    for name, cfg in loader.STRUCTURED_DATASETS.items():
        src = loader.base_dir / cfg["excel"]
        if not src.exists():
            print(f"  ⚠️  {name}: 源文件不存在 {src}")
            continue

        dst = target_dir / src.name
        if dst.exists():
            src_size = src.stat().st_size
            dst_size = dst.stat().st_size
            if src_size == dst_size:
                print(f"  ✅ {name}: {src.name} (已存在，跳过)")
                copied.append(str(dst))
                continue

        shutil.copy2(src, dst)
        size_mb = dst.stat().st_size / 1024 / 1024
        print(f"  📋 {name}: {src.name} → data/uploads/ ({size_mb:.1f}MB)")
        copied.append(str(dst))

    return copied


def copy_unstructured_data(loader: DatasetLoader, target_dir: Path):
    """将非结构化 Markdown 数据复制到 data/rag_docs/"""
    target_dir.mkdir(parents=True, exist_ok=True)
    copied = []

    for name, cfg in loader.UNSTRUCTURED_DATASETS.items():
        src = loader.base_dir / cfg["data"]
        if not src.exists():
            print(f"  ⚠️  {name}: 源文件不存在 {src}")
            continue

        dst = target_dir / src.name
        if dst.exists():
            src_size = src.stat().st_size
            dst_size = dst.stat().st_size
            if src_size == dst_size:
                print(f"  ✅ {name}: {src.name} (已存在，跳过)")
                copied.append(str(dst))
                continue

        shutil.copy2(src, dst)
        size_mb = dst.stat().st_size / 1024 / 1024
        print(f"  📝 {name}: {src.name} → data/rag_docs/ ({size_mb:.1f}MB)")
        copied.append(str(dst))

    return copied


def main():
    parser = argparse.ArgumentParser(description="导入训练数据集")
    parser.add_argument("--summary", "-s", action="store_true", help="仅显示数据集摘要")
    parser.add_argument("--structured", nargs="*", help="指定导入的结构化数据集名称")
    parser.add_argument("--unstructured", nargs="*", help="指定导入的非结构化数据集名称")
    parser.add_argument("--skip-copy", action="store_true", help="跳过文件复制")
    args = parser.parse_args()

    loader = DatasetLoader(base_dir=str(PROJECT_ROOT))

    # 显示摘要
    print()
    print(loader.get_summary())
    print()

    if args.summary:
        return

    # 确定要导入的数据集
    struct_names = args.structured if args.structured else list(loader.STRUCTURED_DATASETS.keys())
    unstruct_names = args.unstructured if args.unstructured else list(loader.UNSTRUCTURED_DATASETS.keys())

    # 复制文件
    if not args.skip_copy:
        print("=" * 60)
        print("  导入结构化数据 (Excel → data/uploads/)")
        print("=" * 60)

        # 临时过滤 loader 仅处理选定的数据集
        uploads_dir = PROJECT_ROOT / "data" / "uploads"
        copy_structured_excels(loader, uploads_dir)

        print()
        print("=" * 60)
        print("  导入非结构化数据 (Markdown → data/rag_docs/)")
        print("=" * 60)

        rag_dir = PROJECT_ROOT / "data" / "rag_docs"
        copy_unstructured_data(loader, rag_dir)

    # 验证加载
    print()
    print("=" * 60)
    print("  验证数据集加载")
    print("=" * 60)

    for name in struct_names:
        ds = loader.load_structured(name)
        if ds:
            print(f"  ✅ structured/{name}: {ds.task_count} 条任务加载成功")
        else:
            print(f"  ❌ structured/{name}: 加载失败")

    for name in unstruct_names:
        ds = loader.load_unstructured(name)
        if ds:
            print(f"  ✅ unstructured/{name}: {ds.task_count} 条任务加载成功")
        else:
            print(f"  ❌ unstructured/{name}: 加载失败")

    print()
    print("🎉 训练数据集导入完成！")
    print()
    print("下一步:")
    print("  python scripts/run_eval.py --dataset 金融     # 评测单个数据集")
    print("  python scripts/run_eval.py --all               # 评测全部数据集")


if __name__ == "__main__":
    main()

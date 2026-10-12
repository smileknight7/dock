#!/usr/bin/env python
"""Meeko 速查工具 —— 直接问你环境里真实装着的 Meeko,不依赖会过时的文档。

用法:
    python meeko_help.py                      总览:版本/路径 + CLI 工具 + 全部公开 API
    python meeko_help.py Polymer              看某个类/模块有哪些方法
    python meeko_help.py Polymer.from_pdb_string   看某方法的签名 + 文档
    python meeko_help.py -v Polymer.export_static_atom_params
                                              加 -v:附带源码文件:行号 + 完整文档
    python meeko_help.py -g pdbqt             按关键词模糊搜名字(类名/方法名都搜)

速查(Meeko 0.7.1,可能随版本变,以脚本实时输出为准):

  命令行工具(在 PATH 上,终端直接调):
    mk_prepare_ligand.py    配体准备
    mk_prepare_receptor.py  受体准备(prepare_receptor_pdbqt 调的就是它)
    mk_export.py            导出

  常用 Python API:
    MoleculePreparation                        分子准备器(mk_prep)
    Polymer / Monomer                          蛋白/残基对象
      .from_pdb_string / .export_static_atom_params / .get_valid_monomers
    ResidueChemTemplates / ResidueTemplate / ResiduePadder   残基模板
    PDBQTMolecule / PDBQTReceptor / PDBQTWriterLegacy         读写 PDBQT
    RDKitMoleculeSetup / AtomTyper / gridbox / Hydrate / Restraint
"""
import argparse
import inspect
import os
import sys


def public_names(obj):
    """列对象的公开名字。优先用 __all__,没有就退回 dir() 过滤下划线。"""
    names = getattr(obj, "__all__", None)
    if names:
        return list(names)
    return [n for n in dir(obj) if not n.startswith("_")]


def safe_getattr(obj, name):
    """__all__ 里可能声明了实际取不到的名字(懒加载/声明失效),别让它崩掉。"""
    try:
        return getattr(obj, name), True
    except AttributeError:
        return None, False


def resolve(dotted):
    """把 'Polymer.from_pdb_string' 这种点路径解析成真实对象。"""
    import meeko

    obj = meeko
    for part in dotted.split("."):
        obj = getattr(obj, part)
    return obj


def show_signature(obj):
    """打印签名;C 扩展函数拿不到签名时给出说明而不是崩溃。"""
    try:
        print("  签名:", inspect.signature(obj))
    except (ValueError, TypeError):
        print("  签名: (拿不到 —— 可能是 C 扩展函数,看 docstring 首行)")


def show_location(obj):
    """打印源码文件:行号,方便你去 grep 实现。编译代码会失败。"""
    try:
        src_file = inspect.getsourcefile(obj) or inspect.getfile(obj)
        _, line = inspect.getsourcelines(obj)
        print(f"  源码: {src_file}:{line}")
    except (TypeError, OSError):
        print("  源码: (无 Python 源码 —— C/编译扩展)")


def overview():
    import meeko

    print(f"Meeko 版本: {getattr(meeko, '__version__', '?')}")
    print(f"安装路径:   {os.path.dirname(meeko.__file__)}")

    bindir = os.path.dirname(sys.executable)
    clis = sorted(
        f for f in os.listdir(bindir)
        if f.startswith("mk_") or "meeko" in f.lower()
    )
    print("\n命令行工具(直接终端调,加 --help 看用法):")
    for c in clis:
        print(f"  {c}")

    print("\n公开 Python API(meeko.__all__):")
    for n in public_names(meeko):
        obj, ok = safe_getattr(meeko, n)
        kind = type(obj).__name__ if ok else "__all__ 里声明了但当前取不到"
        print(f"  {n:<32} [{kind}]")

    print("\n看细节:  python meeko_help.py <名字>       例: python meeko_help.py Polymer")
    print("搜名字:  python meeko_help.py -g <关键词>   例: python meeko_help.py -g pdbqt")


def describe(dotted, verbose=False):
    try:
        obj = resolve(dotted)
    except AttributeError:
        print(f"找不到 '{dotted}'。用 python meeko_help.py -g {dotted.split('.')[-1]} 搜搜看。")
        return

    print(f"{dotted}  ->  {type(obj).__name__}")

    if inspect.isclass(obj):
        methods = public_names(obj)
        print(f"\n公开方法/属性({len(methods)} 个):")
        for m in methods:
            member, ok = safe_getattr(obj, m)
            tag = "()" if ok and callable(member) else ""
            print(f"  {m}{tag}")
        print(f"\n看某方法细节: python meeko_help.py {dotted}.<方法名>")
        if verbose:
            show_location(obj)
    elif inspect.ismodule(obj):
        print("\n模块成员:")
        for n in public_names(obj):
            print(f"  {n}")
    elif callable(obj):
        show_signature(obj)
        if verbose:
            show_location(obj)
        doc = inspect.getdoc(obj) or "(无 docstring)"
        print("\n文档:")
        print("  " + (doc.replace("\n", "\n  ") if verbose else doc.split("\n")[0]))
    else:
        print(f"  值: {obj!r}")
        doc = inspect.getdoc(obj)
        if doc:
            print("  " + doc.split("\n")[0])


def search(keyword):
    import meeko

    kw = keyword.lower()
    hits = []
    for n in public_names(meeko):
        if kw in n.lower():
            hits.append(n)
        obj, ok = safe_getattr(meeko, n)
        if ok and inspect.isclass(obj):
            for m in public_names(obj):
                if kw in m.lower():
                    hits.append(f"{n}.{m}")
    if not hits:
        print(f"没找到含 '{keyword}' 的名字。")
        return
    print(f"含 '{keyword}' 的名字:")
    for h in hits:
        print(f"  {h}")


def main():
    p = argparse.ArgumentParser(
        description="Meeko 速查工具 —— 详见文件顶部 docstring",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("name", nargs="?", help="类/模块/方法名,支持点路径,如 Polymer.from_pdb_string")
    p.add_argument("-v", "--verbose", action="store_true", help="附带源码位置和完整文档")
    p.add_argument("-g", "--grep", metavar="KW", help="按关键词模糊搜名字")
    args = p.parse_args()

    try:
        import meeko  # noqa: F401
    except ImportError:
        print("当前环境没装 meeko。先 conda activate 到装了 meeko 的环境(比如 mopac)。")
        sys.exit(1)

    if args.grep:
        search(args.grep)
    elif args.name:
        describe(args.name, verbose=args.verbose)
    else:
        overview()


if __name__ == "__main__":
    main()

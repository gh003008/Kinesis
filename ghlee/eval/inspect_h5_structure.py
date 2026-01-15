"""inspect_h5_structure.py

2025-11-28 ghlee: helper script to inspect HDF5 motion files under /home/gunhee/LD
and print their internal structure (groups, datasets, shapes, and some attrs).

This is ONLY for inspection; it does not modify any data.
Use this before tuning h5_to_kit_convert.py so we know exactly where joint
angles, root translation, and metadata live.

Usage (from project root):
  python ghlee/inspect_h5_structure.py --base /home/gunhee/LD --max_files 3

"""

import argparse
import os
import h5py


def dump_file(path: str, max_datasets: int = 200) -> None:
    print("=" * 80)
    print("FILE:", path)
    try:
        with h5py.File(path, "r") as f:
            count = 0

            def walk(g, prefix=""):
                nonlocal count
                for k in g:
                    obj = g[k]
                    p = f"{prefix}/{k}" if prefix else k
                    if isinstance(obj, h5py.Dataset):
                        print("  DATASET", p, "shape=", obj.shape, "dtype=", obj.dtype)
                        count += 1
                        if count >= max_datasets:
                            return
                    elif isinstance(obj, h5py.Group):
                        print("  GROUP  ", p)
                        walk(obj, p)
                        if count >= max_datasets:
                            return

            walk(f)

            # Print some common metadata if present
            print("  --- META ---")
            for key in [
                "joint_names",
                "joints",
                "labels",
                "joint_label",
                "marker_names",
                "mocap_framerate",
                "framerate",
                "fps",
            ]:
                if key in f:
                    obj = f[key]
                    try:
                        val = obj[...]
                        print(f"  DATASET {key}: shape={getattr(obj,'shape',None)} sample={val[:5]}")
                    except Exception:
                        print(f"  DATASET {key}: shape={getattr(obj,'shape',None)} (could not preview)")
                if key in f.attrs:
                    print(f"  ATTR {key} = {f.attrs[key]}")

    except Exception as e:
        print("  ERROR opening/reading:", e)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=str, default="/home/gunhee/LD")
    parser.add_argument("--max_files", type=int, default=3,
                        help="max number of .h5 files to inspect")
    args = parser.parse_args()

    if not os.path.isdir(args.base):
        print("Base dir does not exist:", args.base)
        return

    h5_files = [
        os.path.join(args.base, f)
        for f in sorted(os.listdir(args.base))
        if f.endswith(".h5")
    ]
    if not h5_files:
        print("No .h5 files directly under", args.base)
        return

    for i, path in enumerate(h5_files):
        if i >= args.max_files:
            break
        dump_file(path)


if __name__ == "__main__":
    main()

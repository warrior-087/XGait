import re
import pathlib

USER_ID = 19
ROOT = pathlib.Path(f"output_converted/user19")

PAT = re.compile(r"^MVI_(\d+)$", re.IGNORECASE)

def key_func(p: pathlib.Path) -> int:
    m = PAT.match(p.stem)
    return int(m.group(1)) if m else 10**18

def rename_one_folder(folder: pathlib.Path, path_id: int) -> None:
    if not folder.exists():
        print(f"[SKIP] missing: {folder}")
        return

    files = [
        f for f in folder.iterdir()
        if f.is_file()
        and f.suffix.lower() == ".npy"
        and PAT.match(f.stem)
    ]

    if not files:
        print(f"[SKIP] no matching npy in: {folder}")
        return

    files.sort(key=key_func)

    targets = [folder / f"user{USER_ID}-{path_id}-{idx}.npy" for idx in range(1, len(files) + 1)]

    # 1) Target conflict check (avoid overwriting existing files)
    conflicts = [t for t in targets if t.exists()]
    if conflicts:
        raise FileExistsError(
            f"[PATH {path_id}] Target exists, will NOT overwrite: {[c.name for c in conflicts]}"
        )

    # 2) Rename to temp name first to avoid intermediate collision overwriting
    tmp_files = []
    for i, old in enumerate(files, start=1):
        tmp = folder / f".__tmp__user{USER_ID}-{path_id}-{i}__{old.name}"
        if tmp.exists():
            raise FileExistsError(f"[PATH {path_id}] Temp file exists: {tmp.name}")
        old.rename(tmp)
        tmp_files.append(tmp)

    # 3) Rename from temp name to final name
    for tmp, final in zip(tmp_files, targets):
        print(f"[PATH {path_id}] {tmp.name} -> {final.name}")
        tmp.rename(final)

    print(f"[DONE] PATH {path_id}: renamed {len(files)} files in {folder}")

def main():
    for path_id in range(9,10):
        folder = ROOT / str(path_id)
        rename_one_folder(folder, path_id)

    print("All done!")

if __name__ == "__main__":
    main()

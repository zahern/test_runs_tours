"""Choose the code-only file set for the publish branch.

Input  : _pubtmp/head_z.bin   (git ls-tree -r -z HEAD)
         _pubtmp/cp_files.txt (files already on origin/clean-publish-371)
Output : _pubtmp/wanted_z.bin (index-info -z records to ADD/REPLACE)
         _pubtmp/drop_z.bin   (index-info -z records to REMOVE)

Scope
-----
Publish the project's own source, at HEAD:
  * .gitignore / .gitattributes and root-level source files
  * source/config/doc files at the top of code/ and demand_/
  * source anywhere under the project's own source dirs
    (code/_patches, code/old_code_and_runs, code/docs, code/simulation,
     code/matsim, code/compat, abm_safety, demand_, prepare_network,
     pyram, estimation_sf, review)
Keep, unconditionally, every path already on the publish branch: it is
already public, and a refresh must not silently delete it.

Drop: generated data and build artefacts, backups, wheels, pickles,
jars/classes, gz, figures, saved_data/, results/, and vendored copies of
third-party projects.
"""
import collections
import os
import re
import sys

# Scratch files live in a repo-local dir: git-bash /tmp is not visible to the
# native Windows python that runs this script.
TMP = os.environ.get("PUB_TMP", "_pubtmp")
HEAD_Z = os.path.join(TMP, "head_z.bin")
CP_LIST = os.path.join(TMP, "cp_files.txt")
OUT_WANT = os.path.join(TMP, "wanted_z.bin")
OUT_DROP = os.path.join(TMP, "drop_z.bin")

EXT_OK = (".py", ".sh", ".pbs", ".md", ".txt", ".yml", ".yaml",
          ".cfg", ".toml", ".java", ".tex", ".sql", ".bib", ".sty",
          ".bst", ".cls")

# The project's own source trees (path prefixes, matched anywhere below).
SOURCE_DIRS = ("code/_patches/", "code/old_code_and_runs/", "code/docs/",
               "code/simulation/", "code/matsim/", "code/compat/",
               "abm_safety/", "demand_/", "prepare_network/", "pyram/",
               "estimation_sf/", "review/")

# Third-party clones and generated output: never published.
VENDORED = ("matsim-example-project-master/", "HypothesisX-fernando-gui/")
GENERATED_DIRS = ("code/saved_data/", "code/results/", "code/runs/",
                  "code/_backup_pre_jaxfix_20260822/", "demand_/results/",
                  "demand_/suncorp_telematics/results/")

# Never published: caches, binaries, editor/backup cruft.
BAD_SUBSTR = ("/.bak", ".bak_", ".bak.", ".orig", ".rej", "~",
              "/.ipynb_checkpoints/", "/__pycache__/", "/.pytest_cache/",
              "/node_modules/", "/.git/")
BAD_EXT = (".pkl", ".pickle", ".whl", ".jar", ".class", ".gz", ".zip",
           ".png", ".jpg", ".jpeg", ".pdf", ".parquet", ".feather",
           ".h5", ".hdf5", ".npz", ".npy", ".pt", ".ckpt", ".omx",
           ".dat", ".xlsx", ".xls", ".html", ".csv", ".geojson",
           ".gpkg", ".dll", ".lib", ".log", ".aux", ".blg", ".itf",
           ".svg", ".jsx", ".done_files")
# Scratch copies that are not part of the codebase.
BAD_NAMES = ("code/_run_sa_bandit - test.py",)


def wanted(path):
    if path in BAD_NAMES:
        return False, "scratch-copy"
    if path.startswith(VENDORED):
        return False, "vendored-third-party"
    for d in GENERATED_DIRS:
        if path.startswith(d):
            return False, "generated-dir"
    for s in BAD_SUBSTR:
        if s in path:
            return False, "backup-or-cache"
    low = path.lower()
    for e in BAD_EXT:
        if low.endswith(e):
            return False, "non-source-ext"
    if path in (".gitignore", ".gitattributes"):
        return True, "root-config"
    if not low.endswith(EXT_OK):
        return False, "ext-not-included"
    if "/" not in path:                       # root-level source (README.md, ...)
        return True, "root-source"
    if re.match(r"^(code|demand_)/[^/]+$", path):
        return True, "top-level-source"
    for d in SOURCE_DIRS:
        if path.startswith(d):
            return True, "nested-source"
    return False, "outside-scope"


def parse_z(blob):
    recs = []
    for raw in blob.split(b"\0"):
        if not raw:
            continue
        meta, path = raw.split(b"\t", 1)
        mode, typ, sha = meta.split(b" ")
        recs.append((mode.decode(), typ.decode(), sha.decode(),
                     path.decode("utf-8", "surrogateescape")))
    return recs


def main():
    try:
        head = parse_z(open(HEAD_Z, "rb").read())
    except FileNotFoundError:
        sys.exit(f"missing {HEAD_Z}: run git ls-tree -r -z HEAD > {HEAD_Z}")
    existing = set()
    if os.path.exists(CP_LIST):
        existing = {ln.strip() for ln in open(CP_LIST, encoding="utf-8")
                    if ln.strip()}

    head_by_path = {p: (m, t, s) for m, t, s, p in head}

    want, reasons = {}, collections.Counter()
    for mode, typ, sha, path in head:
        keep, why = wanted(path)
        if path in existing:            # already public -> always keep
            keep, why = True, "already-published"
        if keep:
            want[path] = (mode, typ, sha)
        else:
            reasons[why] += 1

    missing = sorted(p for p in existing if p not in head_by_path)
    reasons["dropped-from-HEAD"] += len(missing)

    with open(OUT_WANT, "wb") as fh:
        for path in sorted(want):
            mode, typ, sha = want[path]
            fh.write(f"{mode} {typ} {sha}\t{path}".encode(
                "utf-8", "surrogateescape") + b"\0")

    zero = "0" * 40
    with open(OUT_DROP, "wb") as fh:
        for path in missing:
            fh.write(f"0 {zero}\t{path}".encode(
                "utf-8", "surrogateescape") + b"\0")

    print(f"HEAD tracked          : {len(head)}")
    print(f"already on branch     : {len(existing)}")
    print(f"-> wanted             : {len(want)}")
    print(f"   refreshed          : {sum(1 for p in want if p in existing)}")
    print(f"   new to branch      : {sum(1 for p in want if p not in existing)}")
    print(f"-> dropped (gone@HEAD): {len(missing)}")
    for p in missing:
        print(f"     - {p}")
    print("excluded, by reason   :")
    for why, n in reasons.most_common():
        print(f"     {n:6d}  {why}")
    print("\nkept by top-level area:")
    for area, n in collections.Counter(
            p.split("/")[0] if "/" in p else "<root>" for p in want
    ).most_common():
        print(f"     {n:6d}  {area}")


if __name__ == "__main__":
    main()

"""Generate the code reference pages and navigation."""

from pathlib import Path
import logging

import mkdocs_gen_files

nav = mkdocs_gen_files.Nav()

root = Path(__file__).parent.parent
src_suffix = "src/arv"
src = root / src_suffix
generated_directory_name = "User_C_API"

files_found = sorted(src.rglob("*.h"))
print(f"{files_found = }")
for path in files_found:
    ignore_final_dirs = ["tests", "demos"]
    if (
            any([path.parent.as_posix().endswith(ignore_dir) for ignore_dir in ignore_final_dirs])
            or any([part.startswith("_") or part.endswith("_") for part in path.relative_to(src).parent.parts])
    ):
        continue

    module_path = path.relative_to(src).with_suffix("")
    doc_path = path.relative_to(src)
    full_doc_path = Path(generated_directory_name, doc_path)
    parts = tuple(module_path.parts)

    part_names = tuple([part.replace("_", " ").strip() for part in parts])
    nav[part_names] = doc_path.as_posix()
    
    print(f"{path = } {full_doc_path = } {doc_path = } {module_path = } {parts = } {part_names = }")

    with mkdocs_gen_files.open(full_doc_path, "w") as fd:
        fd.write(f"::: {Path(src_suffix, doc_path)}")

    mkdocs_gen_files.set_edit_path(full_doc_path, path.relative_to(root))

with mkdocs_gen_files.open(f"{generated_directory_name}/SUMMARY.md", "w") as nav_file:
    nav_file.writelines(nav.build_literate_nav())

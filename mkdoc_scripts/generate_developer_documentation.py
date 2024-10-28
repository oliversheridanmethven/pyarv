"""Generate the code reference pages and navigation."""

from pathlib import Path, PurePath
import logging

import mkdocs_gen_files

nav = mkdocs_gen_files.Nav()

root = Path(__file__).parent.parent
src = root / "src/pyarv"
generated_directory_name = "developer_reference"

for path in sorted(src.rglob("*.py")):
    ignore_final_dirs = ["tests", "demos"]
    if (
            any([path.parent.as_posix().endswith(ignore_dir) for ignore_dir in ignore_final_dirs])
            # or path.name.startswith("_")

    ):
        logging.info(f"Ignoring {path = }")
        continue

    module_path = path.relative_to(src).with_suffix("")
    doc_path = path.relative_to(src).with_suffix(".md")
    full_doc_path = Path(generated_directory_name, doc_path)

    parts = tuple(module_path.parts)

    if parts[-1] == "__init__":
        parts = parts[:-1]
        # continue
        # Do nothing with the __init__ stuff.
        rename_init_as = "index"
        doc_path = doc_path.with_name(f"{rename_init_as}.md")
        full_doc_path = full_doc_path.with_name(f"{rename_init_as}.md")
    elif parts[-1] == "__main__":
        continue

    if not parts:
        # Skipping the __init__ that might be in the root module.
        continue

    part_names = tuple([part.replace("_", " ").strip() for part in parts])
    nav[part_names] = doc_path.as_posix()

    with mkdocs_gen_files.open(full_doc_path, "w") as fd:
        ident = ".".join(parts)
        fd.write(f"::: {ident}")

    mkdocs_gen_files.set_edit_path(full_doc_path, path.relative_to(root))

with mkdocs_gen_files.open(f"{generated_directory_name}/SUMMARY.md", "w") as nav_file:
    nav_file.writelines(nav.build_literate_nav())

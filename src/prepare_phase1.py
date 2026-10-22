
from pathlib import Path
import copy
import shutil
import subprocess
import sys
import yaml
from config import load_config
from datetime import datetime

ROOT = Path(__file__).resolve().parents[1]
PACK_DIR = ROOT / "course_pack_v1"
SCORE_PATH = ROOT / "score.py"

CONFIG_FILES = [
    "configs/closed_book.yaml",
    "configs/oracle.yaml",
    "configs/dense.yaml",
]

RUN_FOLDERS = {
    "closed_book": "closed_book",
    "oracle": "oracle",
    "dense_rag": "dense",
}


def prepare_prediction(source, destination):
    if not source.is_file():
        raise FileNotFoundError(source)

    destination.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    if source.resolve() == destination.resolve():
        return

    if destination.exists():
        if source.read_bytes() != destination.read_bytes():
            raise RuntimeError(
                f"Different prediction file already exists: "
                f"{destination}"
            )
        return

    shutil.copy2(source, destination)


def create_run_config(
    config,
    split,
    config_filename,
    predictions_path,
):
    command = config.get("command") or (
        "jupyter nbconvert --to notebook --execute "
        "advanced-nlp.ipynb "
        "--output advanced-nlp-executed.ipynb"
    )

    run_date = config.get("date")
    if not run_date:
        run_date = datetime.fromtimestamp(
            predictions_path.stat().st_mtime
        ).date().isoformat()

    settings = copy.deepcopy(config["settings"])

    if not config["models"].get("embedder"):
        settings.pop("embedding", None)

    return {
        "system_name": config["system_name"],
        "phase": config["phase"],
        "split": split,
        "description": config.get("description", ""),
        "config_file": config_filename,
        "models": copy.deepcopy(config["models"]),
        "settings": settings,
        "command": command,
        "date": run_date,
    }

def save_run_config(data, destination):
    content = yaml.safe_dump(
        data,
        allow_unicode=True,
        sort_keys=False,
    )

    if destination.exists():
        old_content = destination.read_text(
            encoding="utf-8"
        )

        if old_content != content:
            backup = destination.with_suffix(".yaml.bak")
            if backup.exists():
                raise RuntimeError(
                    f"Backup already exists: {backup}"
                )

            shutil.copy2(destination, backup)

    destination.write_text(
        content,
        encoding="utf-8"
    )


def validate_run(predictions_path, split):
    command = [
        sys.executable,
        str(SCORE_PATH),
        str(predictions_path),
        "--pack",
        str(PACK_DIR),
        "--split",
        split,
        "--validate-only",
    ]

    subprocess.run(
        command,
        check=True,
    )


def main():
    total = 0

    for config_filename in CONFIG_FILES:
        config = load_config(config_filename)

        name = config["system_name"]
        folder_name = RUN_FOLDERS[name]

        for split, relative_path in config[
            "source_predictions"
        ].items():

            source = ROOT / relative_path

            output_dir = (
                ROOT / "runs" / split / folder_name
            )

            predictions_path = (
                output_dir / "predictions.jsonl"
            )

            run_config_path = (
                output_dir / "run_config.yaml"
            )

            prepare_prediction(
                source,
                predictions_path
            )

            validate_run(
                predictions_path,
                split
            )

            run_config = create_run_config(
                config,
                split,
                config_filename,
                predictions_path,
            )

            save_run_config(
                run_config,
                run_config_path
            )

            total += 1

            print(f"\n[OK] {name} / {split}")
            print(f"  Predictions: {predictions_path}")
            print(f"  Config: {run_config_path}")

    if total != 5:
        raise RuntimeError(
            f"Expected 5 runs, found {total}"
        )

    print("\nAll five Phase-1 runs are prepared.")


if __name__ == "__main__":
    main()

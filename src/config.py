
from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parents[1]


def load_config(config_path):
    path = Path(config_path)

    if not path.is_absolute():
        path = ROOT / path

    if not path.is_file():
        raise FileNotFoundError(path)

    with path.open(encoding="utf-8") as f:
        config = yaml.safe_load(f)

    required = [
        "system_name",
        "phase",
        "models",
        "settings",
    ]

    for key in required:
        if key not in config:
            raise ValueError(
                f"Missing configuration key: {key}"
            )

    return config

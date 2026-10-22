
from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parents[1]

LLM_DEFAULTS = {
    "quantization": "nf4",
    "double_quantization": True,
    "compute_dtype": "float16",
    "load_in_4bit": True,
    "model_dtype": "float16",
    "attention_implementation": "sdpa",
    "device_map": "auto",
    "low_cpu_mem_usage": True,
}

EMBEDDING_DEFAULTS = {
    "query_instruction": (
        "Given a question, retrieve passages that contain "
        "the information needed to answer the question"
    ),
    "document_batch_size": 4,
    "query_batch_size": 1,
    "precision": "float32",
    "device": "cuda:0",
    "torch_dtype": "float16",
}


def load_config(config_path):
    path = Path(config_path)

    if not path.is_absolute():
        path = ROOT / path

    if not path.is_file():
        raise FileNotFoundError(path)

    with path.open(encoding="utf-8") as f:
        config = yaml.safe_load(f)

    if not isinstance(config, dict):
        raise ValueError(
            f"Configuration must be a YAML mapping: {path}"
        )

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

    settings = config["settings"]

    if not isinstance(settings, dict):
        raise ValueError("'settings' must be a mapping.")

    embedding = settings.setdefault("embedding", {})

    if not isinstance(embedding, dict):
        raise ValueError(
            "'settings.embedding' must be a mapping."
        )

    for key, value in EMBEDDING_DEFAULTS.items():
        embedding.setdefault(key, value)
        
    models = config["models"]
    if not isinstance(models, dict):
        raise ValueError("'models' must be a mapping.")

    llm = models.get("llm")
    if llm is not None:
        if not isinstance(llm, dict):
            raise ValueError("'models.llm' must be a mapping or null.")

        for key, value in LLM_DEFAULTS.items():
            llm.setdefault(key, value)

    return config

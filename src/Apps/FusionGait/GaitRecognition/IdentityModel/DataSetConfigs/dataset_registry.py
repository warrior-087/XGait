# dataset_registry.py
import functools

# Global dictionary mapping dataset name -> dataset class
DATASET_ZOO = {}

def register_dataset(name: str):
    """Decorator: register a dataset class into DATASET_ZOO under the given name (case-insensitive)."""
    key = name.lower()

    def wrapper(cls):
        if key in DATASET_ZOO:
            old_cls = DATASET_ZOO[key]
            print(
                f"Warning: Dataset '{name}' (key='{key}') is being overwritten: "
                f"{old_cls.__name__} -> {cls.__name__}"
            )
        DATASET_ZOO[key] = cls
        return cls

    return wrapper

def get_dataset_class(name: str):
    """Retrieve a dataset class by name from the registry (case-insensitive)."""
    key = name.lower()
    if key not in DATASET_ZOO:
        raise KeyError(
            f"Dataset '{name}' (key='{key}') not found in registry. "
            f"Available datasets: {list(DATASET_ZOO.keys())}"
        )
    return DATASET_ZOO[key]

                      
                                                                                

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
from typing import Any

import torch


def tensor_digest(value: Any) -> str:
    digest = hashlib.sha256()

    def visit(item: Any, path: str) -> None:
        if torch.is_tensor(item):
            array = item.detach().cpu().contiguous().numpy()
            digest.update(path.encode())
            digest.update(str(array.dtype).encode())
            digest.update(str(array.shape).encode())
            digest.update(array.tobytes())
        elif isinstance(item, dict):
            for key in sorted(item, key=str):
                visit(item[key], f"{path}/{key}")
        elif isinstance(item, (list, tuple)):
            for index, child in enumerate(item):
                visit(child, f"{path}/{index}")

    visit(value, "root")
    return digest.hexdigest()


def replace_strings(value: Any, old: str, new: str) -> Any:
    if isinstance(value, str):
        return value.replace(old, new)
    if isinstance(value, dict):
        return {key: replace_strings(child, old, new) for key, child in value.items()}
    if isinstance(value, list):
        return [replace_strings(child, old, new) for child in value]
    if isinstance(value, tuple):
        return tuple(replace_strings(child, old, new) for child in value)
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--old", required=True)
    parser.add_argument("--new", required=True)
    args = parser.parse_args()

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    before = tensor_digest(checkpoint)
    checkpoint = replace_strings(checkpoint, args.old, args.new)
    after = tensor_digest(checkpoint)
    if before != after:
        raise RuntimeError("Tensor digest changed during metadata sanitization")

    temporary = args.checkpoint.with_suffix(args.checkpoint.suffix + ".sanitized")
    torch.save(checkpoint, temporary)
    reloaded = torch.load(temporary, map_location="cpu", weights_only=False)
    if tensor_digest(reloaded) != before:
        temporary.unlink(missing_ok=True)
        raise RuntimeError("Tensor digest changed after checkpoint serialization")
    temporary.replace(args.checkpoint)
    print(f"Tensor digest preserved: {before}")


if __name__ == "__main__":
    main()

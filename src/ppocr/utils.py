from pathlib import Path

import requests
import torch
from tqdm import tqdm


def _download_file(url: str, output_file: Path) -> Path:
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get("content-length", 0))

    with (
        open(output_file, "wb") as file,
        tqdm(
            desc=output_file.name,
            total=total_size,
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
        ) as bar,
    ):
        for data in response.iter_content(chunk_size=1024):
            size = file.write(data)
            bar.update(size)
    return output_file


def maybe_download_github_asset(file_name: str, output_dir: Path) -> Path:
    output_file = output_dir / file_name
    if output_file.exists():
        return output_file

    url = f"https://github.com/jblemoine/ppocr/releases/download/v0.0.1/{file_name}"
    return _download_file(url, output_file)


def load_module(
    module: torch.nn.Module, weights_file: Path, device: str, eval: bool = True
):
    module.load_state_dict(
        torch.load(weights_file, map_location=device, weights_only=False)
    )
    module.to(device)
    if eval:
        module.eval()
    return module

import argparse
import contextlib
import json
import os
import sys
import types
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm


CSV_DTYPES = {
    "patient_id": str,
    "image_id": str,
    "split": str,
}


def parse_bool(value):
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise ValueError(f"Cannot parse boolean value: {value!r}")


def resolve_path(path_value, base_dir):
    if path_value in (None, ""):
        return None
    path = Path(path_value)
    if not path.is_absolute():
        path = Path(base_dir) / path
    return path


def resolve_data_path(path_value, data_dir):
    path = Path(path_value)
    if not path.is_absolute():
        path = Path(data_dir) / path
    return path


def resolve_image_path(img_root, patient_id, image_id):
    image_id_text = str(image_id)
    base_path = Path(img_root) / str(patient_id) / image_id_text
    candidates = [base_path]
    if base_path.suffix == "":
        candidates.extend(
            [
                base_path.with_suffix(".png"),
                base_path.with_suffix(".jpg"),
                base_path.with_suffix(".jpeg"),
            ]
        )
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def install_lightweight_breastclip_import(repo_root):
    codebase_dir = Path(repo_root) / "src" / "codebase"
    breastclip_dir = codebase_dir / "breastclip"
    if not breastclip_dir.exists():
        raise FileNotFoundError(f"Cannot find breastclip package at {breastclip_dir}")
    if str(codebase_dir) not in sys.path:
        sys.path.insert(0, str(codebase_dir))

    # Avoid importing breastclip.__init__, which pulls trainer/datamodule deps that
    # are not needed for encoder-only inference.
    if "breastclip" not in sys.modules:
        package = types.ModuleType("breastclip")
        package.__path__ = [str(breastclip_dir)]
        sys.modules["breastclip"] = package
    if "breastclip.model" not in sys.modules:
        model_package = types.ModuleType("breastclip.model")
        model_package.__path__ = [str(breastclip_dir / "model")]
        sys.modules["breastclip.model"] = model_package


def torch_load(torch_module, path, map_location="cpu"):
    try:
        return torch_module.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch_module.load(path, map_location=map_location)


def extract_image_encoder_state(state_dict):
    image_encoder_state = {}
    for key, value in state_dict.items():
        normalized_key = key[7:] if key.startswith("module.") else key
        if not normalized_key.startswith("image_encoder."):
            continue
        inner_key = normalized_key[len("image_encoder.") :]
        if inner_key.startswith("module."):
            inner_key = inner_key[len("module.") :]
        image_encoder_state[inner_key] = value
    return image_encoder_state


class EmbeddingDataset:
    def __init__(self, df, image_paths, img_size, mean, std):
        self.df = df.reset_index(drop=True)
        self.image_paths = list(image_paths)
        self.img_size = tuple(img_size)
        self.mean = float(mean)
        self.std = float(std)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        image_path = self.image_paths[index]
        image = Image.open(image_path).convert("RGB")
        if self.img_size is not None:
            height, width = self.img_size
            image = image.resize((width, height), Image.BILINEAR)

        image_array = np.asarray(image).astype("float32")
        image_array -= image_array.min()
        max_value = image_array.max()
        if max_value > 0:
            image_array /= max_value
        image_array = (image_array - self.mean) / self.std
        image_array = np.transpose(image_array, (2, 0, 1))

        return {
            "x": image_array,
            "row": index,
            "img_path": str(image_path),
        }


def collate_batch(batch):
    import torch

    images = np.stack([item["x"] for item in batch], axis=0)
    return {
        "x": torch.from_numpy(images).float(),
        "row": [item["row"] for item in batch],
        "img_path": [item["img_path"] for item in batch],
    }


def load_origin_encoder(repo_root, clip_checkpoint, origin_checkpoint, device):
    import torch
    from torch import nn

    install_lightweight_breastclip_import(repo_root)
    from breastclip.model.modules import load_image_encoder

    base_ckpt = torch_load(torch, clip_checkpoint, map_location="cpu")
    try:
        image_encoder_config = base_ckpt["config"]["model"]["image_encoder"]
        base_model_state = base_ckpt["model"]
    except KeyError as exc:
        raise KeyError(
            "Base checkpoint must contain config['model']['image_encoder'] and model."
        ) from exc

    image_encoder = load_image_encoder(image_encoder_config)
    base_encoder_state = extract_image_encoder_state(base_model_state)
    if not base_encoder_state:
        raise KeyError(f"No image_encoder.* weights found in {clip_checkpoint}")
    image_encoder.load_state_dict(base_encoder_state, strict=True)

    loaded_source = "pretrained"
    if origin_checkpoint is not None:
        origin_ckpt = torch_load(torch, origin_checkpoint, map_location="cpu")
        origin_state = origin_ckpt.get("model", origin_ckpt)
        origin_encoder_state = extract_image_encoder_state(origin_state)
        if not origin_encoder_state:
            raise KeyError(f"No image_encoder.* weights found in {origin_checkpoint}")
        image_encoder.load_state_dict(origin_encoder_state, strict=True)
        loaded_source = "origin_finetuned"

    class OriginImageEncoder(nn.Module):
        def __init__(self, encoder, config):
            super().__init__()
            self.image_encoder = encoder
            self.config = config
            self.image_encoder_type = str(config.get("model_type", "")).lower()
            self.image_encoder_name = str(config.get("name", "")).lower()
            self.out_dim = getattr(encoder, "out_dim", None)

        def encode_image(self, images):
            if self.image_encoder_type == "cnn" and "detect" in self.image_encoder_name:
                output = self.image_encoder(
                    {"image": images, "breast_clip_train_mode": True}
                )
            else:
                output = self.image_encoder(images)

            if isinstance(output, (tuple, list)):
                output = output[0]
            if output.ndim == 3:
                output = output[:, 0]
            return output

        def forward(self, images):
            return self.encode_image(images)

    model = OriginImageEncoder(image_encoder, image_encoder_config)
    model.to(device)
    model.eval()
    return model, image_encoder_config, loaded_source


def build_metadata(df, img_dir, split_name, max_samples, skip_missing):
    df = df.copy()
    df["source_row"] = np.arange(len(df), dtype=np.int64)

    if split_name != "all":
        if "split" not in df.columns:
            raise ValueError("--split was requested but the CSV has no split column.")
        split_values = df["split"].astype(str).str.strip().str.lower()
        df = df[split_values == split_name].copy()

    if max_samples is not None:
        df = df.head(max_samples).copy()

    df = df.reset_index(drop=True)
    image_paths = [
        resolve_image_path(img_dir, row["patient_id"], row["image_id"])
        for _, row in df.iterrows()
    ]

    missing = [path for path in image_paths if not Path(path).exists()]
    if missing and not skip_missing:
        preview = "\n".join(str(path) for path in missing[:10])
        raise FileNotFoundError(
            f"{len(missing)} image(s) are missing. First missing path(s):\n{preview}"
        )

    if missing and skip_missing:
        keep_mask = [Path(path).exists() for path in image_paths]
        df = df.loc[keep_mask].reset_index(drop=True)
        image_paths = [path for path, keep in zip(image_paths, keep_mask) if keep]

    metadata = df.copy()
    metadata.insert(0, "embedding_row", np.arange(len(metadata), dtype=np.int64))
    metadata["resolved_img_path"] = [str(path) for path in image_paths]
    return metadata, image_paths


def export_embeddings(model, dataset, output_path, args, device):
    import torch
    from torch.utils.data import DataLoader

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        drop_last=False,
        collate_fn=collate_batch,
    )

    amp_enabled = parse_bool(args.amp) and device.type == "cuda"
    embeddings = None
    write_pos = 0
    micro_batch_size = max(1, int(args.micro_batch_size or args.batch_size))

    with torch.no_grad():
        for batch in tqdm(loader, desc="Export embeddings"):
            inputs = batch["x"].to(device, non_blocking=True).contiguous()
            batch_features = []
            for start in range(0, inputs.size(0), micro_batch_size):
                end = min(start + micro_batch_size, inputs.size(0))
                autocast = (
                    torch.cuda.amp.autocast(enabled=amp_enabled)
                    if device.type == "cuda"
                    else contextlib.nullcontext()
                )
                with autocast:
                    features = model.encode_image(inputs[start:end])
                batch_features.append(features.detach().float().cpu().numpy())

            batch_features = np.concatenate(batch_features, axis=0).astype(
                np.float32, copy=False
            )
            if batch_features.ndim != 2:
                raise ValueError(
                    f"Expected 2D embeddings, got shape {batch_features.shape}"
                )
            if embeddings is None:
                embeddings = np.lib.format.open_memmap(
                    output_path,
                    mode="w+",
                    dtype=np.float32,
                    shape=(len(dataset), batch_features.shape[1]),
                )
            embeddings[write_pos : write_pos + batch_features.shape[0]] = batch_features
            write_pos += batch_features.shape[0]

    if embeddings is None:
        raise ValueError("No rows were exported; check --split and --max-samples.")
    embeddings.flush()
    return tuple(embeddings.shape)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Export pooled image-encoder embeddings from Mammo-FM origin.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data-dir", default="/mnt/g/data", type=str)
    parser.add_argument("--img-dir", default="images_png", type=str)
    parser.add_argument("--csv-file", default="train_with_test_data.csv", type=str)
    parser.add_argument(
        "--clip-checkpoint",
        "--clip_chk_pt_path",
        dest="clip_checkpoint",
        default="./model/Mammo-FM_BatmanlabTrained_CLIP.tar",
        type=str,
        help="Pre-trained Mammo-FM checkpoint.",
    )
    parser.add_argument(
        "--origin-ckpt",
        default=None,
        type=str,
        help="Optional fine-tuned origin checkpoint, e.g. output/.../checkpoints/best_fold0_seed42.pth.",
    )
    parser.add_argument("--output-dir", default=None, type=str)
    parser.add_argument(
        "--split",
        default="all",
        choices=["all", "train", "val", "test"],
        help="Rows to export. Metadata keeps the original split column.",
    )
    parser.add_argument("--img-size", nargs=2, default=[1520, 912], type=int, metavar=("H", "W"))
    parser.add_argument("--mean", default=0.3089279, type=float)
    parser.add_argument("--std", default=0.25053555408335154, type=float)
    parser.add_argument("--batch-size", default=4, type=int)
    parser.add_argument("--micro-batch-size", default=1, type=int)
    parser.add_argument("--num-workers", default=2, type=int)
    parser.add_argument("--device", default="cuda", type=str)
    parser.add_argument("--gpu-id", default=0, type=int)
    parser.add_argument("--amp", default="y", type=str)
    parser.add_argument("--max-samples", default=None, type=int)
    parser.add_argument(
        "--skip-missing",
        action="store_true",
        help="Skip rows whose image file is missing instead of failing.",
    )
    parser.add_argument(
        "--save-parquet",
        action="store_true",
        help="Also save metadata.parquet when pyarrow is available.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    repo_root = Path(__file__).resolve().parent
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)

    import torch

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    data_dir = Path(args.data_dir)
    csv_path = resolve_data_path(args.csv_file, data_dir)
    img_dir = resolve_data_path(args.img_dir, data_dir)
    clip_checkpoint = resolve_path(args.clip_checkpoint, repo_root)
    origin_checkpoint = resolve_path(args.origin_ckpt, repo_root)

    if args.output_dir in (None, ""):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = repo_root / "output" / "origin_embeddings" / timestamp
    else:
        output_dir = resolve_path(args.output_dir, repo_root)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file does not exist: {csv_path}")
    if not img_dir.exists():
        raise FileNotFoundError(f"Image directory does not exist: {img_dir}")
    if not clip_checkpoint.exists():
        raise FileNotFoundError(f"Base Mammo-FM checkpoint does not exist: {clip_checkpoint}")
    if origin_checkpoint is not None and not origin_checkpoint.exists():
        raise FileNotFoundError(f"Origin checkpoint does not exist: {origin_checkpoint}")

    df = pd.read_csv(csv_path, dtype=CSV_DTYPES)
    required_cols = {"patient_id", "image_id"}
    missing_cols = sorted(required_cols - set(df.columns))
    if missing_cols:
        raise ValueError(f"CSV is missing required column(s): {missing_cols}")

    metadata, image_paths = build_metadata(
        df=df,
        img_dir=img_dir,
        split_name=args.split,
        max_samples=args.max_samples,
        skip_missing=args.skip_missing,
    )
    if len(metadata) == 0:
        raise ValueError("No rows selected for export.")

    print("=" * 60)
    print("Mammo-FM Origin Embedding Export")
    print("=" * 60)
    print(f"Rows:             {len(metadata)}")
    print(f"CSV:              {csv_path}")
    print(f"Image dir:        {img_dir}")
    print(f"Base checkpoint:  {clip_checkpoint}")
    print(f"Origin ckpt:      {origin_checkpoint if origin_checkpoint else '(none; pretrained encoder)'}")
    print(f"Output dir:       {output_dir}")
    print(f"Device:           {device}")
    print(f"Image size H W:   {args.img_size}")
    print("=" * 60)

    model, image_encoder_config, loaded_source = load_origin_encoder(
        repo_root=repo_root,
        clip_checkpoint=clip_checkpoint,
        origin_checkpoint=origin_checkpoint,
        device=device,
    )

    dataset = EmbeddingDataset(
        df=metadata,
        image_paths=image_paths,
        img_size=args.img_size,
        mean=args.mean,
        std=args.std,
    )
    embeddings_path = output_dir / "embeddings.npy"
    embedding_shape = export_embeddings(
        model=model,
        dataset=dataset,
        output_path=embeddings_path,
        args=args,
        device=device,
    )

    metadata_path = output_dir / "metadata.csv"
    metadata.to_csv(metadata_path, index=False)

    parquet_path = None
    if args.save_parquet:
        parquet_path = output_dir / "metadata.parquet"
        metadata.to_parquet(parquet_path, index=False)

    manifest = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "source": loaded_source,
        "repo_root": str(repo_root),
        "csv_file": str(csv_path),
        "img_dir": str(img_dir),
        "clip_checkpoint": str(clip_checkpoint),
        "origin_checkpoint": str(origin_checkpoint) if origin_checkpoint else None,
        "split": args.split,
        "num_rows": int(len(metadata)),
        "embedding_shape": list(embedding_shape),
        "embedding_dtype": "float32",
        "embeddings_file": "embeddings.npy",
        "metadata_file": "metadata.csv",
        "metadata_parquet_file": parquet_path.name if parquet_path else None,
        "img_size": [int(args.img_size[0]), int(args.img_size[1])],
        "mean": float(args.mean),
        "std": float(args.std),
        "batch_size": int(args.batch_size),
        "micro_batch_size": int(args.micro_batch_size),
        "amp": parse_bool(args.amp),
        "image_encoder_config": image_encoder_config,
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )

    print("\nSaved:")
    print(f"  {embeddings_path}")
    print(f"  {metadata_path}")
    print(f"  {manifest_path}")
    if parquet_path is not None:
        print(f"  {parquet_path}")
    print(f"\nEmbedding shape: {embedding_shape}")
    print('Fast read example: np.load("embeddings.npy", mmap_mode="r")')


if __name__ == "__main__":
    main()

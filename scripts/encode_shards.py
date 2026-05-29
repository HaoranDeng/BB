from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
import webdataset as wds
from diffusers import AutoencoderKL
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
from transformers import AutoConfig, AutoModel, AutoTokenizer, T5EncoderModel


IMAGE_KEYS = ("jpg", "jpeg", "png", "webp")
TEXT_KEYS = ("txt", "text", "caption")


def decode_text(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="ignore")
    return str(value)


def extract_sample(sample: dict[str, Any]) -> dict[str, Any]:
    image = None
    for key in IMAGE_KEYS:
        if key in sample:
            image = sample[key]
            break
    if image is None:
        raise ValueError("sample has no image key")
    if not isinstance(image, Image.Image):
        raise ValueError("sample image was not decoded as PIL")

    caption = None
    for key in TEXT_KEYS:
        if key in sample:
            caption = decode_text(sample[key])
            break
    if caption is None and "json" in sample:
        metadata = sample["json"]
        if isinstance(metadata, bytes):
            metadata = json.loads(metadata.decode("utf-8", errors="ignore"))
        if isinstance(metadata, dict):
            caption = metadata.get("caption") or metadata.get("text") or metadata.get("TEXT")
    if not caption:
        raise ValueError("sample has no caption")

    return {"image": image.convert("RGB"), "caption": str(caption)}


def collate_samples(
    samples: list[dict[str, Any]],
    image_transform: transforms.Compose,
) -> dict[str, Any]:
    images = torch.stack([image_transform(sample["image"]) for sample in samples], dim=0)
    captions = [sample["caption"] for sample in samples]
    return {"images": images, "captions": captions}


def load_text_encoder(name: str, dtype: torch.dtype, device: torch.device) -> torch.nn.Module:
    config = AutoConfig.from_pretrained(name)
    if config.model_type == "t5":
        model = T5EncoderModel.from_pretrained(name, torch_dtype=dtype)
    else:
        model = AutoModel.from_pretrained(name, torch_dtype=dtype)
    return model.to(device).eval()


def expand_input_shards(input_shards: str) -> str | list[str]:
    if input_shards.startswith("@"):
        list_path = Path(input_shards[1:])
        with list_path.open("r", encoding="utf-8") as f:
            shards = [line.strip() for line in f if line.strip()]
        if not shards:
            raise ValueError(f"Shard list is empty: {list_path}")
        return shards
    return input_shards


def flush_shard(
    output_dir: Path,
    output_prefix: str,
    shard_index: int,
    latents_buffer: list[torch.Tensor],
    text_buffer: list[torch.Tensor],
) -> None:
    if not latents_buffer:
        return
    latents = torch.cat(latents_buffer, dim=0)
    text_embeds = torch.cat(text_buffer, dim=0)
    path = output_dir / f"{output_prefix}{shard_index:06d}.pt"
    torch.save({"latents": latents, "text_embeds": text_embeds}, path)
    print(f"wrote {path} samples={latents.shape[0]}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-shards", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--output-prefix", default="encoded_")
    parser.add_argument("--vae", default="stabilityai/sd-vae-ft-ema")
    parser.add_argument("--text-encoder", default="google/t5-v1_1-base")
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--max-text-tokens", type=int, default=64)
    parser.add_argument("--samples-per-shard", type=int, default=2048)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--precision", choices=["bf16", "fp16", "fp32"], default="bf16")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float32,
    }[args.precision]

    image_transform = transforms.Compose(
        [
            transforms.Resize(args.image_size, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(args.image_size),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ]
    )

    dataset = (
        wds.WebDataset(
            expand_input_shards(args.input_shards),
            shardshuffle=True,
            handler=wds.warn_and_continue,
        )
        .decode("pil", handler=wds.warn_and_continue)
        .map(extract_sample, handler=wds.warn_and_continue)
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.workers,
        collate_fn=lambda samples: collate_samples(samples, image_transform),
    )

    vae = AutoencoderKL.from_pretrained(args.vae, torch_dtype=dtype).to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained(args.text_encoder)
    text_encoder = load_text_encoder(args.text_encoder, dtype=dtype, device=device)
    scaling_factor = float(getattr(vae.config, "scaling_factor", 0.18215))

    latents_buffer: list[torch.Tensor] = []
    text_buffer: list[torch.Tensor] = []
    shard_index = 0
    buffered = 0
    total = 0

    with torch.no_grad():
        for batch in tqdm(loader, desc="encoding"):
            images = batch["images"].to(device=device, dtype=dtype, non_blocking=True)
            captions = batch["captions"]
            tokens = tokenizer(
                captions,
                padding="max_length",
                truncation=True,
                max_length=args.max_text_tokens,
                return_tensors="pt",
            )
            tokens = {key: value.to(device) for key, value in tokens.items()}
            with torch.autocast(
                device_type="cuda",
                dtype=dtype,
                enabled=device.type == "cuda" and dtype != torch.float32,
            ):
                latent_dist = vae.encode(images).latent_dist
                latents = latent_dist.sample() * scaling_factor
                text_outputs = text_encoder(**tokens)
                text_embeds = text_outputs.last_hidden_state
                if "attention_mask" in tokens:
                    text_embeds = text_embeds * tokens["attention_mask"].unsqueeze(-1)

            latents = latents.detach().cpu().to(torch.float16)
            text_embeds = text_embeds.detach().cpu().to(torch.float16)
            latents_buffer.append(latents)
            text_buffer.append(text_embeds)
            buffered += latents.shape[0]
            total += latents.shape[0]

            if buffered >= args.samples_per_shard:
                flush_shard(
                    output_dir,
                    args.output_prefix,
                    shard_index,
                    latents_buffer,
                    text_buffer,
                )
                shard_index += 1
                latents_buffer.clear()
                text_buffer.clear()
                buffered = 0

            if args.max_samples is not None and total >= args.max_samples:
                break

    flush_shard(output_dir, args.output_prefix, shard_index, latents_buffer, text_buffer)
    print(f"encoded_samples={total} output_dir={output_dir}")


if __name__ == "__main__":
    main()

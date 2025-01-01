import argparse
import json
import os
from typing import Optional

import numpy as np
from datasets import DatasetDict, load_dataset
from huggingface_hub import HfApi
from PIL import Image
from tqdm import tqdm

from src.diffusion.datasets.simulated_dataset import SimulatedDataset


def generate_local_dataset(dataset: SimulatedDataset, output_dir: str) -> None:
    """
    1) Iterates over the SimulatedDataset,
    2) Saves each sample's image as a PNG,
    3) Writes the (file_name, text) pairs to metadata.jsonl
    """
    images_dir: str = os.path.join(output_dir, "images")
    os.makedirs(name=images_dir, exist_ok=True)

    metadata_path = os.path.join(output_dir, "metadata.jsonl")
    with open(file=metadata_path, mode="w", encoding="utf-8") as f:
        for i in tqdm(iterable=range(len(dataset)), desc="Generating samples"):
            image_tensor, text_label = dataset[i]  # (3, H, W), "Caption"

            # Convert Torch tensor -> NumPy -> PIL Image
            img_array = image_tensor.numpy().transpose(1, 2, 0)  # (H, W, 3)
            img_array = (img_array * 255).astype(np.uint8)  # scale up if needed

            pil_img = Image.fromarray(img_array)

            # Write to disk
            filename: str = f"{i:05d}.png"
            file_path: str = os.path.join(images_dir, filename)
            pil_img.save(fp=file_path)

            # Write metadata: file_name, text
            record = {"file_name": f"images/{filename}", "text": text_label}
            f.write(json.dumps(obj=record) + "\n")


def cleanup_local_dataset(output_dir: str) -> None:
    """
    Removes the generated images and metadata file after successful upload.
    """
    images_dir: str = os.path.join(output_dir, "images")
    metadata_path = os.path.join(output_dir, "metadata.jsonl")

    # Remove all images
    if os.path.exists(images_dir):
        for file in os.listdir(images_dir):
            os.remove(os.path.join(images_dir, file))
        os.rmdir(images_dir)

    # Remove metadata file
    if os.path.exists(metadata_path):
        os.remove(metadata_path)

    # Remove parent directory if empty
    if os.path.exists(output_dir) and not os.listdir(output_dir):
        os.rmdir(output_dir)


def push_dataset_to_hub(
    dataset_dir: str, repo_id: str, token: Optional[str] = None, private: bool = False
) -> None:
    """
    Loads the dataset from disk using `imagefolder` + `metadata.jsonl`,
    maps each image to its caption, and then pushes to the Hugging Face Hub.
    """
    if token is not None:
        api = HfApi()
        api.set_access_token(token)

    dataset = load_dataset("imagefolder", data_dir=dataset_dir, split="train")

    metadata_file = os.path.join(dataset_dir, "metadata.jsonl")
    file_to_caption = {}
    with open(metadata_file, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            file_to_caption[item["file_name"]] = item["text"]

    def add_captions(example, idx):
        file_name = f"images/{idx:05d}.png"
        return {"text": file_to_caption[file_name]}

    dataset = dataset.map(add_captions, with_indices=True)

    ds_dict = DatasetDict({"train": dataset})
    ds_dict.push_to_hub(repo_id=repo_id, private=private)


def main():
    parser = argparse.ArgumentParser(
        description="Create and push a dataset to the Hugging Face Hub."
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=50,
        help="Number of samples to generate.",
        required=False,
    )
    parser.add_argument(
        "--image_size",
        type=int,
        default=64,
        help="Size of the generated images.",
        required=False,
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="my_shapes_dataset",
        help="Output directory for the dataset.",
    )
    parser.add_argument(
        "--repo_id",
        type=str,
        default="arminpcm/my_shapes_dataset",
        required=False,
        help="Repository ID on the Hugging Face Hub.",
    )
    parser.add_argument(
        "--token",
        type=str,
        required=False,
        help="Hugging Face token for authentication.",
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Make the dataset private.",
        required=False,
    )

    args = parser.parse_args()

    # Create the in-memory dataset
    dataset = SimulatedDataset(num_samples=args.num_samples, image_size=args.image_size)

    # Generate local files for the dataset
    generate_local_dataset(dataset=dataset, output_dir=args.output_dir)

    # Push the dataset to the Hugging Face Hub
    push_dataset_to_hub(
        dataset_dir=args.output_dir,
        repo_id=args.repo_id,
        token=args.token,
        private=args.private,
    )

    # Clean up local files after successful upload
    cleanup_local_dataset(args.output_dir)


if __name__ == "__main__":
    main()

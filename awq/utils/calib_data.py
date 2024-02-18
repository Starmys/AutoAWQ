import json
import torch
import logging
from typing import List, Union
from datasets import load_dataset
from PIL import Image
# from llava.constants import IMAGE_TOKEN_INDEX  # v1.5
# from llava.model.builder import load_pretrained_model
# from llava.mm_utils import tokenizer_image_token, get_model_name_from_path


def get_calib_dataset(data: Union[str, List[str]] = "pileval",
                      tokenizer=None, n_samples=512, block_size=512,
                      split="train", text_column="text"):
    if isinstance(data, str):
        if data == "pileval":
            dataset = load_dataset("mit-han-lab/pile-val-backup", split="validation")
        elif data == 'textvqa':
            return get_text_vqa(n_samples, block_size)
        else:
            split, data_files = split.split(':')
            data_files = {split: data_files}
            dataset = load_dataset(data, split=split, data_files=data_files)
        
        dataset = dataset.shuffle(seed=42)
    
    samples = []
    n_run = 0
    for data in dataset:
        line = data[text_column]
        line = line.strip()
        line_encoded = tokenizer.encode(line)
        if len(line_encoded) < block_size:
            continue
        sample = torch.tensor([line_encoded[:block_size]])
        if sample.numel() == 0:
            continue
        samples.append(sample)
        n_run += 1
        if n_run == n_samples:
            break
    # now concatenate all samples and split according to block size
    cat_samples = torch.cat(samples, dim=1)
    n_split = cat_samples.shape[1] // block_size
    logging.debug(f" * Split into {n_split} blocks")
    # [n_samples * torch.Tensor[1, block_size]]
    return [cat_samples[:, i*block_size:(i+1)*block_size] for i in range(n_split)]


def get_text_vqa(n_samples=512, block_size=512):
    model_path = '/home/chengzhang/models/llava/llava-v1.5-7b'
    text_path = '/home/chengzhang/Multimodal-Quantization/LLaVA/playground/data/eval/textvqa/llava_textvqa_val_v051_ocr.jsonl'
    img_path = '/home/chengzhang/datasets/TextVQA/images-v0.5/train_images'

    with open(text_path) as f:
        image_ids = [json.loads(line)['image'] for line in f.readlines() if line.startswith('{')]

    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path, None, model_name
    )
    model.seqlen = context_len

    samples = []
    for n_run, image_id in enumerate(image_ids):
        image = Image.open(f'{img_path}/{image_id}.jpg')
        image = image_processor.preprocess(image.convert("RGB"), return_tensors='pt')['pixel_values']
        text = f'<image>'
        input_ids = tokenizer_image_token(text, tokenizer, IMAGE_TOKEN_INDEX)[:512]
        image = image.unsqueeze(0).half()
        input_ids = torch.tensor(input_ids, dtype=torch.long).unsqueeze(0)
        samples.append(input_ids)
        if n_run == n_samples:
            break

    import ipdb; ipdb.set_trace()

    return samples

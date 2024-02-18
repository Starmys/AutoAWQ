from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer

# 4.35

model_path = '/home/chengzhang/models/llava/llava-v1.5-7b'
quant_path = './models/llava-v1.5-7b-c4-128'
quant_config = { "zero_point": True, "q_group_size": 128, "w_bit": 4, "version": "GEMM" }

# Load model
model = AutoAWQForCausalLM.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

# Quantize
model.quantize(
    tokenizer=tokenizer,
    quant_config=quant_config,
    calib_data='allenai/c4',
    # split='validation:en/c4-validation.00000-of-00008.json.gz',
    split='train:en/c4-train.00000-of-01024.json.gz',
)

# Save quantized model
model.save_quantized(quant_path)
tokenizer.save_pretrained(quant_path)

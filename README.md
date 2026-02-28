---
language:
- en
- zh
- ja
- ko
- fr
- de
- es
- pt
- it
- ru
- ar
- nl
- pl
- sv
- da
- fi
- nb
- tr
- cs
- sk
- ro
- hu
- el
- uk
- he
license: apache-2.0
tags:
- text-generation
- reasoning
- zenlm
- zen
- abliterated
- causal-lm
- transformers
- moe
---

# Zen4-Max

Zen4-Max is a 30B Mixture-of-Experts (MoE) language model with approximately 3B
parameters active per forward pass. It delivers high-capability inference at the
compute cost of a much smaller dense model, making it ideal for deployments where
both quality and throughput matter.

## Model Specs

| Property          | Value                              |
|-------------------|------------------------------------|
| Parameters        | 30B total / ~3B active (MoE)       |
| Architecture      | Sparse MoE transformer (causal LM) |
| Context window    | 32,768 tokens                      |
| Format            | SafeTensors (BF16)                 |
| License           | Apache 2.0                         |

## Zen4 Family

This model is part of the Zen4 model family:

| Model                                               | Params       | Architecture | Context   | Use case                        |
|-----------------------------------------------------|--------------|--------------|-----------|--------------------------------|
| [zen4-mini](https://huggingface.co/zenlm/zen4-mini) | 4B           | Dense        | 40,960    | Edge, mobile, low-resource     |
| [zen4-pro](https://huggingface.co/zenlm/zen4-pro)   | 14B          | Dense        | 32,768    | Professional, complex reasoning|
| [zen4-max](https://huggingface.co/zenlm/zen4-max)   | 30B MoE      | MoE (3B act.)| 32,768    | High-capability, efficient     |
| [zen4-coder](https://huggingface.co/zenlm/zen4-coder)| 80B MoE     | MoE          | 32,768    | Advanced code, 100+ languages  |
| [zen4-pro-max](https://huggingface.co/zenlm/zen4-pro-max)| 80B MoE | MoE (3B act.)| 32,768   | Large-scale reasoning          |

## Quick Start

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "zenlm/zen4-max"

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype="auto",
    device_map="auto",
)

messages = [
    {"role": "user", "content": "Design a fault-tolerant microservice architecture for a high-traffic payment system."}
]

text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
)
inputs = tokenizer([text], return_tensors="pt").to(model.device)

output_ids = model.generate(**inputs, max_new_tokens=1024)
generated = output_ids[0][len(inputs.input_ids[0]):]
print(tokenizer.decode(generated, skip_special_tokens=True))
```

## Hardware Requirements

MoE models load all expert weights into memory but only activate a subset per token,
so total VRAM tracks total parameter count while compute tracks active parameters.

| Setup               | Minimum VRAM | Recommended            |
|---------------------|--------------|------------------------|
| BF16 full precision | 64 GB        | 80 GB (A100/H100)      |
| 4-bit quantized     | 20 GB        | 24 GB                  |
| CPU inference       | 64 GB RAM    | 128 GB RAM             |

Multi-GPU configurations with NVLink or high-bandwidth interconnect are recommended
for BF16 full-precision inference.

## Formats Available

| Format           | Location                                  |
|------------------|-------------------------------------------|
| SafeTensors BF16 | `zenlm/zen4-max`                          |
| GGUF             | `zenlm/zen4-max-gguf` (various quants)    |

## Abliteration

Zen4-Max weights are derived from an abliterated base model. Abliteration removes
refusal behavior by identifying and nullifying the "refusal direction" in the model's
residual stream.

**Method**: Directional ablation on the residual stream across all layers
**Implementation**: [hanzoai/remove-refusals](https://github.com/hanzoai/remove-refusals) — Hanzo's production abliteration toolkit
**Technique**: [Refusal in LLMs is mediated by a single direction](https://www.lesswrong.com/posts/jGuXSZgv6qfdhMCuJ/refusal-in-llms-is-mediated-by-a-single-direction) — Arditi et al.
**Effect**: Removes refusal behaviors while preserving all other capabilities
**Identity layer**: Zen identity added via system prompt — full LoRA fine-tuning planned

Abliteration is a feature, not a limitation. It enables unrestricted research,
security testing, and applications where safety guardrails are managed at the
application layer rather than baked into model weights.

## Citation

```bibtex
@misc{zen4max2025,
  title        = {Zen4-Max: 30B Mixture-of-Experts Language Model},
  author       = {{Hanzo AI} and {Zoo Labs Foundation}},
  year         = {2025},
  howpublished = {\url{https://huggingface.co/zenlm/zen4-max}},
  note         = {Apache 2.0 License}
}
```

---

Part of the [Zen model family](https://zenlm.org) by [Hanzo AI](https://hanzo.ai) (Techstars '17) and [Zoo Labs Foundation](https://zoo.ngo).

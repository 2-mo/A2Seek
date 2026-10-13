"""Load Qwen LoRA weights against the model's actual module names."""

import re


def remap_lora_state(state, module_names):
    names = set(module_names)
    mapped, targets, pairs = {}, set(), {}
    for key, tensor in state.items():
        match = re.fullmatch(r"base_model\.model\.(.+)\.lora_([AB])\.(?:default\.)?weight", key)
        if match is None:
            raise ValueError(f"Unsupported LoRA checkpoint key: {key}")
        old_module, side = match.groups()
        candidates = [old_module]
        if old_module.startswith("model.layers."):
            candidates.append(old_module.replace("model.layers.", "model.language_model.layers.", 1))
        if old_module.startswith("visual."):
            candidates.append("model." + old_module)
        matches = [candidate for candidate in candidates if candidate in names]
        if len(matches) != 1:
            raise ValueError(f"Cannot resolve LoRA module: {old_module}")
        module = matches[0]
        new_key = f"base_model.model.{module}.lora_{side}.weight"
        if new_key in mapped:
            raise ValueError(f"Duplicate LoRA weight after mapping: {new_key}")
        mapped[new_key] = tensor
        targets.add(module)
        pairs.setdefault(module, set()).add(side)
    if not mapped or any(sides != {"A", "B"} for sides in pairs.values()):
        raise ValueError("Each LoRA module must have both A and B weights")
    return mapped, sorted(targets)


def load_lora(model, checkpoint):
    from peft import LoraConfig, get_peft_model, set_peft_model_state_dict
    from peft.utils import load_peft_weights

    config = LoraConfig.from_pretrained(checkpoint)
    if config.bias != "none" or config.modules_to_save or config.use_dora or config.rank_pattern or config.alpha_pattern:
        raise ValueError("Use a merged model for LoRA checkpoints with bias, saved modules, DoRA or per-layer rank/alpha")
    weights = load_peft_weights(checkpoint, device="cpu")
    weights, targets = remap_lora_state(weights, (name for name, _ in model.named_modules()))
    # Exact targets prevent an old regular expression from selecting additional
    # visual layers after a Transformers module-layout change.
    config.target_modules = targets
    config.inference_mode = True
    model = get_peft_model(model, config)
    result = set_peft_model_state_dict(model, weights)
    missing = [name for name in result.missing_keys if ".lora_" in name]
    if missing or result.unexpected_keys:
        raise ValueError(f"Incomplete LoRA load: {len(missing)} missing, {len(result.unexpected_keys)} unexpected weights")
    print(f"Loaded {len(weights)} LoRA weights across {len(targets)} modules", flush=True)
    return model.eval()

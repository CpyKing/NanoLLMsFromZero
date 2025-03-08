# import transformers
# from transformers import AutoModelForMaskedLM
# import modelscope
from modelscope.utils.hf_util import AutoModelForCausalLM, AutoTokenizer


if __name__ == '__main__':
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-0.5B-Instruct",
        device_map="cuda",
        torch_dtype="auto",
        trust_remote_code=True,
    )
    
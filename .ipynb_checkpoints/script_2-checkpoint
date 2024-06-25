from transformers import pipeline
import torch


def LLM_output(input):
    model_id = "meta-llama/Meta-Llama-3-8B-Instruct"

    pipe = pipeline(
        "text-generation",
        model=model_id,
        model_kwargs={"torch_dtype": torch.bfloat16},
        device="cuda",
    )

    messages = [
        {"role": "system", "content": "You are a pirate chatbot who always responds in pirate speak!"},
        {"role": "user", "content": "Who are you?"},
    ]

    terminators = [
        pipe.tokenizer.eos_token_id,
        pipe.tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]

    outputs = pipe(
        messages,
        max_new_tokens=256,
        eos_token_id=terminators,
        do_sample=True,
        temperature=10.0,
        top_p=0.1,
    )
    assistant_response = outputs[0]["generated_text"][-1]["content"]
    return(assistant_response)
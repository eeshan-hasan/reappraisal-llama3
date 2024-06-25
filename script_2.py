

def LLM_output(input_text):

    from transformers import pipeline
    import torch

    model_id = "meta-llama/Meta-Llama-3-8B-Instruct"

    pipe = pipeline(
        "text-generation",
        model=model_id,
        model_kwargs={"torch_dtype": torch.bfloat16},
        device=0,  # assuming CUDA device 0
    )

    messages = [
        {"role": "system", "content": "You are trying to help people to do cognitive reappraisal before they tweet.\
        This is especially important when the content might be harmful to themselves or others. If its not, then just say Posted."},
        {"role": "user", "content": input_text},
    ]

    # Ensure eos_token_id is properly set
    eos_token_id = pipe.tokenizer.eos_token_id

    try:
        outputs = pipe(
            messages,
            max_new_tokens=2000,
            eos_token_id=eos_token_id,
            do_sample=True,
            temperature=10.0,
            top_p=0.1,
        )
        assistant_response = outputs[0]["generated_text"][-1]["content"]

        del pipe
        del pipeline
        del outputs
        torch.cuda.empty_cache()



        return assistant_response
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        return "Sorry, my GPU is out of memory!"
    except Exception as e:
        return f"An error occurred: {e}"

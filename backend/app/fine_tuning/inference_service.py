import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)


BASE_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"


class FineTunedLLM:

    def __init__(self):

        print("Loading Qwen base model for RAG...")

        self.tokenizer = AutoTokenizer.from_pretrained(
            BASE_MODEL
        )

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL,
            quantization_config=quantization_config,
            device_map="auto",
        )

        self.model.eval()

        print("Qwen base model loaded.")

    def generate(
        self,
        question: str,
        context: str = "",
        max_new_tokens: int = 120,
    ):

        messages = [
            {
                "role": "system",
                "content": (
                    "You are an enterprise IT support assistant.\n\n"

                    "You MUST answer the user's question using ONLY the "
                    "SUPPORT CONTEXT provided below.\n\n"

                    "STRICT RULES:\n"
                    "1. Answer the exact question asked by the user.\n"
                    "2. Use only facts explicitly present in SUPPORT CONTEXT.\n"
                    "3. Do not use general knowledge or assumptions.\n"
                    "4. Do not invent, infer, or guess missing information.\n"
                    "5. Ignore information in the context that is unrelated "
                    "to the question.\n"
                    "6. If the question asks about a specific fact, return "
                    "only that specific fact.\n"
                    "7. Preserve numbers, durations, names, requirements, "
                    "and times exactly as stated in the context.\n"
                    "8. Answer in English.\n"
                    "9. Answer in 1-2 concise sentences.\n"
                    "10. NEVER copy or reproduce the entire SUPPORT CONTEXT.\n"
                    "11. NEVER answer with headings, document text, or unrelated "
                    "paragraphs.\n"
                    "12. If the answer cannot be found explicitly in the "
                    "SUPPORT CONTEXT, respond EXACTLY with:\n"
                    "\"I don't have enough information in the available "
                    "support documentation to answer that question.\"\n\n"

                    "EXAMPLES:\n"
                    "If the context says 'An account is temporarily locked "
                    "after five unsuccessful login attempts' and the user "
                    "asks 'What happens after five unsuccessful login attempts?', "
                    "answer: 'The account is temporarily locked.'\n\n"

                    "If the context says 'The account automatically unlocks "
                    "after 30 minutes' and the user asks 'How long does the "
                    "account remain locked?', answer: "
                    "'The account automatically unlocks after 30 minutes.'\n\n"

                    "If the context contains no vacation policy and the user "
                    "asks 'What is the vacation policy?', return the exact "
                    "fallback sentence above."
                ),
            },
            {
                "role": "user",
                "content": (
                    "SUPPORT CONTEXT:\n"
                    "====================\n"
                    f"{context}\n"
                    "====================\n\n"
                    "USER QUESTION:\n"
                    f"{question}\n\n"
                    "Provide ONLY the answer to the USER QUESTION."
                ),
            }
        ]

        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
        )

        inputs = {
            key: value.to(self.model.device)
            for key, value in inputs.items()
        }

        with torch.inference_mode():

            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                num_beams=1,
                repetition_penalty=1.05,
                no_repeat_ngram_size=4,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.pad_token_id,
            )

        generated_tokens = outputs[0][
            inputs["input_ids"].shape[1]:
        ]

        answer = self.tokenizer.decode(
            generated_tokens,
            skip_special_tokens=True,
        ).strip()

        return answer
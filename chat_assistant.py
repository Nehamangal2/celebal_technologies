import sys
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer

def main():
    model_name = "Qwen/Qwen2.5-0.5B-Instruct"
    
    print(f"Loading conversational model '{model_name}'...")
    print("Note: If this is the first time running, it will download the model weights (~900MB).")
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype="auto",
            device_map="auto"
        )
    except Exception as e:
        print(f"\nError loading model: {e}")
        print("Please check your internet connection and ensure Hugging Face libraries are installed correctly.")
        return

    print("\n" + "="*60)
    print("               LOCAL MODERN CHATGPT ASSISTANT               ")
    print("="*60)
    print("Type your question and press Enter. Type 'exit' to quit.\n")

    # Conversation history list for Qwen chat templates
    messages = [
        {"role": "system", "content": "You are a helpful, respectful, and honest assistant. Answer in clear, modern English."}
    ]

    while True:
        try:
            user_input = input("\nYou: ")
        except (KeyboardInterrupt, EOFError):
            print("\nExiting. Goodbye!")
            break

        if user_input.strip().lower() == "exit":
            print("Exiting. Goodbye!")
            break

        if not user_input.strip():
            continue

        # Append user message
        messages.append({"role": "user", "content": user_input})

        # Apply chat template
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        # Tokenize inputs
        model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

        print("AI Assistant: ", end="", flush=True)

        # Setup custom streamer to print output word-by-word
        streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

        # Generate response using streaming
        with torch.no_grad():
            generated_ids = model.generate(
                **model_inputs,
                max_new_tokens=512,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                streamer=streamer
            )

        # Retrieve the generated token IDs (excluding inputs) and save to history
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        
        messages.append({"role": "assistant", "content": response})

if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-
import torch
import torch.nn.functional as F
import os
import numpy as np
import evaluate
import scipy.io
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    TrainerCallback
)

# ==========================================
# 1. INITIAL CONFIGURATION
# ==========================================
# Define the target model to be fine-tuned
model_name = "answerdotai/ModernBERT-base"
# Define the output directory for the extracted MATLAB matrices
NOME_CARTELLA_OUTPUT = "matrici_matlab_dirette_rope"
os.makedirs(NOME_CARTELLA_OUTPUT, exist_ok=True)

# Define the specific transformer layer and attention head to inspect
LAYER_DA_SPIARE = 21
TESTA_DA_SPIARE = 3
NUM_EPOCHE = 4

# Set maximum sequence lengths for training and evaluation datasets
MAX_LEN_TRAINING = 4096
MAX_LEN_EVAL = 5000

def main():
    # Download and prepare the LexGLUE (SCOTUS) dataset
    print("1. Scaricamento del dataset LexGLUE (SCOTUS)...")
    dataset = load_dataset("lex_glue", "scotus", split="train")
    # Split the dataset into 90% training and 10% testing sets
    dataset = dataset.train_test_split(test_size=0.1, seed=42)

    # Load Tokenizer, Model, and Evaluation Metrics
    print("2. Caricamento Tokenizer, Modello e Metriche...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # FORCE the use of Scaled Dot-Product Attention (SDPA) to ensure the probing hook functions correctly
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=14,
        attn_implementation="sdpa" 
    )

    # Enable gradient checkpointing to save memory during the backward pass
    model.gradient_checkpointing_enable()

    # Load the accuracy metric from the Hugging Face evaluate library
    metrica_accuracy = evaluate.load("accuracy")

    # Define the metric computation function for the Trainer
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        return metrica_accuracy.compute(predictions=predictions, references=labels)

    # Prepare a fixed input sequence from the test set to use for extracting attention matrices consistently
    testo_lungo_originale = dataset["test"][0]["text"]
    inputs_fissi = tokenizer(
        testo_lungo_originale,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_LEN_EVAL,
        padding="max_length"
    )

    # Set up the computation device (GPU if available, otherwise fallback to CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Dispositivo in uso: {device}")
    
    # Move fixed inputs and the model to the appropriate device
    inputs_fissi = {k: v.to(device) for k, v in inputs_fissi.items()}
    model.to(device)

    # Define the tokenization mapping function applied to dataset batches
    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=MAX_LEN_TRAINING)

    print("3. Tokenizzazione in corso...")
    # Determine the optimal number of CPU cores for parallel dataset processing
    num_cores = max(1, os.cpu_count() - 1) if os.cpu_count() else 1
    tokenized_datasets = dataset.map(tokenize_function, batched=True, num_proc=num_cores)

    # ==========================================
    # 3. SDPA PROBING CALLBACK (RoPE APPLIED)
    # ==========================================
    # Dictionary used to temporarily cache the Query and Key matrices post-RoPE
    q_k_rope_cache = {}

    class ProbingCallback(TrainerCallback):
        def on_step_end(self, args, state, control, model, **kwargs):
            # Retrieve the current epoch and calculate the final planned epoch
            epoca_attuale = int(state.epoch) if state.epoch is not None else 0
            epoca_finale_prevista = int(args.num_train_epochs) - 1

            # Restrict the probing and matrix extraction strictly to the final epoch
            if epoca_attuale != epoca_finale_prevista:
                return

            # Set model to evaluation mode for clean inference during probing
            model.eval()
            
            # Temporarily back up the native PyTorch SDPA function
            original_sdpa = F.scaled_dot_product_attention
            layer_counter = [0] # Counter to keep track of the current transformer layer being executed

            # Define the interceptor function to intercept and extract Query and Key matrices
            def spy_sdpa(query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, **sdpa_kwargs):
                current_layer = layer_counter[0]
                
                if current_layer == LAYER_DA_SPIARE:
                    # BINGO! The matrices intercepted here are POST-RoPE.
                    # Their expected shape is: (batch_size, num_heads, seq_len, head_dim)
                    
                    # Isolate the specific attention head requested for extraction
                    q_testa = query[:, TESTA_DA_SPIARE, :, :]
                    k_testa = key[:, TESTA_DA_SPIARE, :, :]
                    
                    # Detach the tensors from the computation graph and move them to CPU memory
                    q_k_rope_cache[f"Q_rope"] = q_testa.detach().cpu()
                    q_k_rope_cache[f"K_rope"] = k_testa.detach().cpu()

                # Increment the layer counter for the next attention block
                layer_counter[0] += 1
                
                # Execute the original SDPA function so the model's forward pass continues normally
                return original_sdpa(query, key, value, attn_mask, dropout_p, is_causal, **sdpa_kwargs)

            # 1. Activate the interceptor by monkey-patching the PyTorch SDPA function
            F.scaled_dot_product_attention = spy_sdpa

            # Run a forward pass with the fixed inputs (no gradients required)
            with torch.no_grad():
                _ = model(**inputs_fissi)

            # 2. Deactivate the interceptor by restoring the original SDPA function
            F.scaled_dot_product_attention = original_sdpa

            # Convert the cached PyTorch tensors into a dictionary formatted for MATLAB export
            dati_matlab = {}
            for nome, tensor in q_k_rope_cache.items():
                dati_matlab[nome] = tensor.to(torch.float32).numpy()

            step = state.global_step
            file_path = os.path.join(NOME_CARTELLA_OUTPUT, f"step_{step}_ROPE.mat")
            
            # Save the extracted matrices to a .mat file
            scipy.io.savemat(file_path, dati_matlab)

            # Log the successful extraction and save operation every 50 steps
            if step % 50 == 0:
                print(f"[Epoca Finale] Salvato file MATLAB (con RoPE): {file_path}")

            # Clear the cache for the next training step and return the model to train mode
            q_k_rope_cache.clear()
            model.train()

    # ==========================================
    # 4. TRAINING INITIALIZATION
    # ==========================================
    # Determine the optimal mixed-precision format supported by the hardware environment
    use_fp16 = True
    use_bf16 = False
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        use_fp16 = False
        use_bf16 = True
        print("[*] bfloat16 supportato: verra utilizzato per maggiore stabilita.")

    # Define the training arguments, logging strategies, and hyperparameters
    training_args = TrainingArguments(
        output_dir="./risultati_modernbert",
        learning_rate=3e-5,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=4,
        num_train_epochs=NUM_EPOCHE,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        fp16=use_fp16,
        bf16=use_bf16,
        gradient_checkpointing=True,
        save_strategy="no",
        logging_steps=50,
        eval_strategy="steps",
        eval_steps=100,
        report_to="none"
    )

    # Initialize the Hugging Face Trainer with the model, datasets, metrics, and the custom probing callback
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["test"],
        compute_metrics=compute_metrics,
        callbacks=[ProbingCallback()]
    )

    # Execute the fine-tuning process
    print(f"\n[*] Inizio del Fine-Tuning...")
    trainer.train()

if __name__ == "__main__":
    main()
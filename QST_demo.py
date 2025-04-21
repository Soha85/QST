import graphviz
from PIL import Image
import base64
from io import BytesIO
import json
import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, DataCollatorForLanguageModeling
from datasets import load_dataset
from torch.utils.data import DataLoader
import os
import tempfile
import time
import matplotlib.pyplot as plt

# Set page title and layout

model_dir = "saved_models/latest"

# Load from saved paths if available
paths_file = os.path.join(model_dir, "paths_info.json")
if os.path.exists(paths_file):
    with open(paths_file, 'r') as f:
        saved_info = json.load(f)
        model_dir = saved_info.get("model_path", model_dir)
        base_model_name = saved_info.get("base_model_name", "distilgpt2")
else:
    base_model_name = "distilgpt2"
st.set_page_config(page_title="Quantized Side Tuning App", layout="wide")
st.title("Quantized Side Tuning for Large Language Models")
st.markdown("""
This app demonstrates Quantized Side Tuning (QST) - a technique that enables efficient fine-tuning 
of quantized large language models by keeping the base model frozen while training a smaller side network.
""")
def display_qst_architecture():
    # Create a graphviz diagram
    dot = graphviz.Digraph(comment='QST Architecture')
    
    # Add nodes
    dot.attr('node', shape='box', style='filled', fillcolor='lightblue')
    dot.node('Input', 'Input Tokens')
    
    dot.attr('node', shape='box', style='filled', fillcolor='#D6EAF8')
    dot.node('QM', 'Quantized Model\n(Frozen)')
    
    dot.attr('node', shape='box', style='filled', fillcolor='#F9E79F')
    dot.node('DSL', 'Downsampling Layers')
    
    dot.attr('node', shape='box', style='filled', fillcolor='#FCF3CF')
    dot.node('SN', 'Side Network\n(Trainable)')
    
    dot.attr('node', shape='diamond', style='filled', fillcolor='#D5F5E3')
    dot.node('Fusion', 'Fusion\nMechanism')
    
    dot.attr('node', shape='box', style='filled', fillcolor='#D7BDE2')
    dot.node('Output', 'Output Logits')
    
    # Add edges
    dot.edge('Input', 'QM')
    dot.edge('QM', 'DSL', label='hidden states')
    dot.edge('DSL', 'SN')
    dot.edge('QM', 'Fusion', label='output β')
    dot.edge('SN', 'Fusion', label='output (1-β)')
    dot.edge('Fusion', 'Output')
    
    return dot

# Add this function to generate a flow diagram of QST training process
def display_qst_flow():
    # Create a graphviz diagram for the flow
    dot = graphviz.Digraph(comment='QST Flow', graph_attr={'rankdir': 'TB'})
    
    # Define nodes
    dot.attr('node', shape='box', style='filled', fillcolor='#AED6F1')
    dot.node('Start', 'Start')
    
    dot.attr('node', shape='box', style='filled', fillcolor='#D6EAF8')
    dot.node('Load', 'Load Pre-trained Model')
    
    dot.attr('node', shape='box', style='filled', fillcolor='#D6EAF8')
    dot.node('Quant', 'Quantize Model\n(INT8/INT4)')
    
    dot.attr('node', shape='box', style='filled', fillcolor='#D6EAF8')
    dot.node('Freeze', 'Freeze Quantized Model')
    
    dot.attr('node', shape='box', style='filled', fillcolor='#FCF3CF')
    dot.node('Init', 'Initialize Side Network')
    
    dot.attr('node', shape='box', style='filled', fillcolor='#FCF3CF')
    dot.node('Train', 'Train Side Network\n(Full precision)')
    
    dot.attr('node', shape='box', style='filled', fillcolor='#D5F5E3')
    dot.node('Adjust', 'Adjust Fusion Parameter β')
    
    dot.attr('node', shape='box', style='filled', fillcolor='#D7BDE2')
    dot.node('Save', 'Save Model')
    
    dot.attr('node', shape='box', style='filled', fillcolor='#AED6F1')  
    dot.node('End', 'End')
    
    # Add edges to show flow
    dot.edge('Start', 'Load')
    dot.edge('Load', 'Quant')
    dot.edge('Quant', 'Freeze')
    dot.edge('Freeze', 'Init')
    dot.edge('Init', 'Train')
    dot.edge('Train', 'Adjust')
    dot.edge('Adjust', 'Train', label='Iterate')
    dot.edge('Adjust', 'Save')
    dot.edge('Save', 'End')
    
    return dot

# Add this function to create a detailed inference diagram
def display_qst_inference():
    # Create a graphviz diagram for inference
    dot = graphviz.Digraph(comment='QST Inference', graph_attr={'rankdir': 'LR'})
    
    # Subgraph for input
    with dot.subgraph(name='cluster_input') as c:
        c.attr(style='filled', color='lightgrey')
        c.node_attr.update(style='filled', color='white')
        c.node('input', 'Input Text')
        c.node('tokens', 'Tokenized Input')
        c.attr(label='Input Processing')
        c.edge('input', 'tokens')
    
    # Subgraph for base model
    with dot.subgraph(name='cluster_base') as c:
        c.attr(style='filled', color='#D6EAF8')
        c.node_attr.update(style='filled', color='white')
        c.node('q_model', 'Quantized Model\n(Frozen)')
        c.node('q_output', 'Base Model Output')
        c.attr(label='Quantized Base Model')
        c.edge('q_model', 'q_output')
    
    # Subgraph for side network
    with dot.subgraph(name='cluster_side') as c:
        c.attr(style='filled', color='#FCF3CF')
        c.node_attr.update(style='filled', color='white')
        c.node('downsample', 'Downsampling')
        c.node('side_layers', 'Side Network Layers')
        c.node('side_output', 'Side Network Output')
        c.attr(label='Side Network')
        c.edge('downsample', 'side_layers')
        c.edge('side_layers', 'side_output')
    
    # Subgraph for fusion and output
    with dot.subgraph(name='cluster_output') as c:
        c.attr(style='filled', color='#D5F5E3')
        c.node_attr.update(style='filled', color='white')
        c.node('fusion', 'β * base + (1-β) * side')
        c.node('logits', 'Output Logits')
        c.node('tokens_out', 'Output Tokens')
        c.node('text_out', 'Generated Text')
        c.attr(label='Fusion and Output')
        c.edge('fusion', 'logits')
        c.edge('logits', 'tokens_out')
        c.edge('tokens_out', 'text_out')
    
    # Connect the subgraphs
    dot.edge('tokens', 'q_model')
    dot.edge('tokens', 'downsample')
    dot.edge('q_model', 'downsample', style='dashed', label='hidden states')
    dot.edge('q_output', 'fusion')
    dot.edge('side_output', 'fusion')
    
    return dot
# Define model architecture classes
class DownsampleLayer(nn.Module):
    """
    Simple linear downsampler that reduces hidden dimension
    but preserves sequence length. Avoids pooling to ensure
    compatibility with base model outputs.
    """
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.adapter = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        # x: [batch_size, seq_len, input_dim]
        return self.adapter(x)  # -> [batch_size, seq_len, output_dim]

class SideNetwork(nn.Module):
    """Side network that runs alongside the frozen quantized LLM"""
    def __init__(self, base_model_config, hidden_size, num_layers, dropout=0.1):
        super().__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        
        # Get base model dimensions
        self.base_hidden_size = base_model_config.hidden_size
        
        # Create downsampling layers
        self.downsamplers = nn.ModuleList([
            DownsampleLayer(self.base_hidden_size, hidden_size)
            for _ in range(num_layers)
        ])
        
        # Create side network layers
        self.layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_size * 2 if i > 0 else hidden_size, hidden_size),
                nn.LayerNorm(hidden_size),
                nn.GELU(),
                nn.Dropout(dropout)
            )
            for i in range(num_layers)
        ])
        
        # Combination weights (learnable)
        self.alpha = nn.Parameter(torch.ones(num_layers))
        
        # Final projection to match base model's hidden size
        self.final_proj = nn.Linear(hidden_size, self.base_hidden_size)
    
    def forward(self, base_outputs):
        # base_outputs is a list of hidden states from the base model
        
        side_states = []
        current_state = None
        
        for i in range(self.num_layers):
            # Get base model output for this layer
            base_hidden = base_outputs[i]
            
            # Downsample base model output
            downsampled = self.downsamplers[i](base_hidden)
            
            if i == 0:
                # First layer just uses downsampled base output
                current_state = self.layers[i](downsampled)
            else:
                # Combine previous side output with downsampled base output
                combined = torch.cat([current_state, downsampled], dim=-1)
                current_state = self.layers[i](combined)
            
            side_states.append(current_state)
        
        # Project final side output back to base model dimension
        final_side_output = self.final_proj(side_states[-1])
        
        return side_states, final_side_output

class QuantizedSideTuning(nn.Module):
    """Quantized SideTuning main model"""
    def __init__(self, model_name, side_hidden_size=256, device="cuda"):
        super().__init__()
        self.device = device
        self.name_or_path = model_name
        
        # Load base model in 4-bit quantization
        with st.spinner(f"Loading base model {model_name}..."):
            base_model = AutoModelForCausalLM.from_pretrained(model_name)
            self.base_model = torch.quantization.quantize_dynamic(base_model, {torch.nn.Linear}, dtype=torch.qint8)
            self.base_model.to(device)
        
        # Freeze base model parameters
        for param in self.base_model.parameters():
            param.requires_grad = False
        
        # Get configuration
        self.config = self.base_model.config
        
        # Create side network
        self.side_network = SideNetwork(
            self.config,
            hidden_size=side_hidden_size,
            num_layers=self.config.num_hidden_layers
        ).to(device)
        
        # Final combination weight
        self.beta = nn.Parameter(torch.tensor(0.5).to(device))
        
        # Reuse the LLM head
        self.lm_head = self.base_model.get_output_embeddings()
    
    def forward(self, input_ids, attention_mask=None):
        # Run base model in evaluation mode (no gradients)
        with torch.no_grad():
            base_outputs = self.base_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                return_dict=True
            )
            
            base_hidden_states = base_outputs.hidden_states
            base_last_hidden = base_hidden_states[-1]
        
        # Run side network
        side_hidden_states, side_last_hidden = self.side_network(base_hidden_states)
        
        # Combine final representations with learned weight
        beta = torch.sigmoid(self.beta)  # Keep between 0 and 1
        combined_hidden = beta * base_last_hidden + (1 - beta) * side_last_hidden
        
        # Use the LLM head for final prediction
        logits = self.lm_head(combined_hidden)
        
        return logits
    
    def save_pretrained(self, output_dir):
        """Save the side network and combination weights"""
        os.makedirs(output_dir, exist_ok=True)
        
        # Save side network
        side_network_path = os.path.join(output_dir, "side_network.pt")
        torch.save(self.side_network.state_dict(), side_network_path)
        
        # Save beta parameter
        beta_path = os.path.join(output_dir, "beta.pt")
        torch.save(self.beta, beta_path)
        
        # Save config
        config_dict = {
            "base_model_name": self.name_or_path,
            "side_hidden_size": self.side_network.hidden_size
        }
        config_path = os.path.join(output_dir, "config.pt")
        torch.save(config_dict, config_path)
    
    @classmethod
    def from_pretrained(cls, base_model_name, output_dir, device="cuda"):
       
       """Load pretrained side network and combination weights"""
       # Load config first
       config_path = os.path.join(output_dir, "config.pt")
       if not os.path.exists(config_path):
           raise FileNotFoundError(f"Missing config file: {config_path}")
    
       config = torch.load(config_path)
       side_hidden_size = config["side_hidden_size"]

       # Create model with correct side_hidden_size
       model = cls(base_model_name, side_hidden_size=side_hidden_size, device=device)

       # Load side network weights
       side_network_path = os.path.join(output_dir, "side_network.pt")
       model.side_network.load_state_dict(torch.load(side_network_path, map_location=device))

       # Load beta parameter
       beta_path = os.path.join(output_dir, "beta.pt")
       model.beta = torch.load(beta_path, map_location=device)

       return model

# Create sidebar for navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Choose a page", ["Home", "Train Model", "Generate Text", "About QST"])

if page == "Home":
    st.header("Welcome to Quantized Side Tuning Demo")
    st.markdown("""
    ### What is Quantized Side Tuning?
    
    Quantized Side Tuning (QST) is a technique that enables efficient fine-tuning of quantized large language models.
    It works by keeping the base quantized model frozen while training a smaller side network.
    
    ### Key Benefits:
    - **Memory Efficient**: Uses quantized base model (INT8/INT4)
    - **Fast Training**: Only updates parameters in the side network
    - **Flexible Deployment**: Maintains quantization benefits in production
    
    ### How to use this app:
    1. Go to the **Train Model** page to fine-tune a model on your data
    2. Use the **Generate Text** page to create text with your trained model
    3. Learn more in the **About QST** page
    """)
    
    st.image("https://www.researchgate.net/profile/Suhwan-Goh/publication/375669307/figure/fig1/AS:11431281159991461@1707380747825/Comparison-of-fine-tuning-strategies-for-quantized-large-language-models-QST-uses-a.png", 
             caption="Comparison of fine-tuning strategies")

elif page == "Train Model":
    st.header("Train a QST Model")
    
    # Create two columns for input parameters
    col1, col2 = st.columns(2)
    
    with col1:
        model_name = st.selectbox(
            "Select Base Model",
            ["distilgpt2", "gpt2", "gpt2-medium", "facebook/opt-125m"],
            help="Choose a base model to quantize and fine-tune"
        )
        
        dataset_name = st.selectbox(
            "Select Dataset",
            ["wikitext", "imdb", "ag_news", "custom"],
            help="Dataset for fine-tuning"
        )
        
        if dataset_name == "custom":
            st.info("For custom datasets, provide text data below:")
            custom_data = st.text_area(
                "Enter your training data (each line as a separate example):",
                height=200
            )
            
        if dataset_name != "custom":
            dataset_config = st.selectbox(
                "Dataset Configuration",
                {
                    "wikitext": ["wikitext-2-raw-v1", "wikitext-103-raw-v1"],
                    "imdb": ["plain_text"],
                    "ag_news": [""]
                }[dataset_name]
            )
    
    with col2:
        side_hidden_size = st.slider(
            "Side Network Hidden Size",
            min_value=32,
            max_value=512,
            value=128,
            step=32,
            help="Smaller values use less memory but may reduce performance"
        )
        
        batch_size = st.slider(
            "Batch Size",
            min_value=1,
            max_value=30,
            value=5,
            step=1
        )
        
        learning_rate = st.select_slider(
            "Learning Rate",
            options=[1e-5, 2e-5, 5e-5, 1e-4, 2e-4, 5e-4, 1e-3],
            format_func=lambda x: f"{x:.0e}"
        )
        
        num_epochs = st.slider(
            "Number of Epochs",
            min_value=1,
            max_value=5,
            value=1
        )
    
    # Training button
    if st.button("Start Training"):
        # Set device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        st.info(f"Using device: {device}")
        
        # Create progress containers
        progress_bar = st.progress(0)
        status_text = st.empty()
        loss_chart = st.empty()
        
        # Create temp directory for saving
        with tempfile.TemporaryDirectory() as output_dir:
            try:
                # Initialize model
                with st.spinner("Initializing model..."):
                    model = QuantizedSideTuning(model_name, side_hidden_size=side_hidden_size, device=device)
                    tokenizer = AutoTokenizer.from_pretrained(model_name)
                    tokenizer.pad_token = tokenizer.eos_token
                
                # Load and prepare dataset
                with st.spinner("Preparing dataset..."):
                    if dataset_name == "custom":
                        # Create custom dataset from text input
                        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".txt") as f:
                            # Make sure each line is a separate example
                            f.write('\n'.join([line.strip() for line in custom_data.split('\n') if line.strip()]))
                            custom_file = f.name
        
                        dataset = load_dataset("text", data_files=custom_file, split="train")
                    else:
                        if dataset_config:
                            dataset = load_dataset(dataset_name, dataset_config, split="train")
                        else:
                            dataset = load_dataset(dataset_name, split="train")
    
                    # Ensure dataset has 'text' column
                    if 'text' not in dataset.column_names:
                        if 'sentence' in dataset.column_names:
                            dataset = dataset.rename_column('sentence', 'text')
                        elif 'content' in dataset.column_names:
                            dataset = dataset.rename_column('content', 'text')
                        else:
                            st.error(f"Dataset has no 'text' column. Available columns: {dataset.column_names}")
                            st.stop()
                    
                    # Take a small sample for demonstration purposes
                    dataset = dataset.select(range(min(len(dataset), 1000)))
                    
                    # Tokenize dataset
                    def tokenize_function(examples):
                        return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=512)
                    
                    tokenized_dataset = dataset.map(tokenize_function, batched=True, 
                                                    remove_columns=dataset.column_names  # Remove all original columns
                                                    )
                    
                    # Create data collator
                    data_collator = DataCollatorForLanguageModeling(
                        tokenizer=tokenizer, 
                        mlm=False
                    )
                    
                    # Create dataloader
                    dataloader = DataLoader(
                        tokenized_dataset, 
                        batch_size=batch_size, 
                        collate_fn=data_collator
                    )
                
                # Set up optimizer
                optimizer = torch.optim.AdamW(
                    [p for p in model.parameters() if p.requires_grad],
                    lr=learning_rate
                )
                
                # Training loop
                model.train()
                losses = []
                
                for epoch in range(num_epochs):
                    total_loss = 0
                    for batch_idx, batch in enumerate(dataloader):
                        # Update progress
                        progress = (epoch * len(dataloader) + batch_idx) / (num_epochs * len(dataloader))
                        progress_bar.progress(progress)
                        status_text.text(f"Epoch {epoch+1}/{num_epochs}, Batch {batch_idx+1}/{len(dataloader)}")
                        
                        # Move batch to device
                        input_ids = batch["input_ids"].to(device)
                        attention_mask = batch["attention_mask"].to(device)
                        labels = batch["labels"].to(device)
                        
                        # Forward pass
                        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                        
                        # Calculate loss
                        loss_fct = nn.CrossEntropyLoss()
                        loss = loss_fct(outputs.view(-1, outputs.size(-1)), labels.view(-1))
                        
                        # Backward pass and optimize
                        loss.backward()
                        optimizer.step()
                        optimizer.zero_grad()
                        
                        total_loss += loss.item()
                        losses.append(loss.item())
                        
                        
                    avg_loss = total_loss / len(dataloader)
                    status_text.text(f"Epoch {epoch+1}/{num_epochs} completed. Average Loss: {avg_loss:.4f}")
                # Plot loss at the end of training
                fig, ax = plt.subplots()
                ax.plot(losses)
                ax.set_xlabel('Batch')
                ax.set_ylabel('Loss')
                ax.set_title('Training Loss')
                loss_chart.pyplot(fig)
                plt.close(fig)
                
                # Save final model
                status_text.text("Saving model...")
                #model.save_pretrained(output_dir)
                
                # Save final model to a persistent folder, not temp directory
                # model_dir = os.path.join("saved_models", f"qst_{int(time.time())}")
                model_dir = "saved_models/latest"
                os.makedirs(model_dir, exist_ok=True)
                model.save_pretrained(model_dir)
                
                # Create download button for the saved model
                model_bytes = {}
                for filename in os.listdir(output_dir):
                    with open(os.path.join(output_dir, filename), 'rb') as f:
                        model_bytes[filename] = f.read()
                
                for filename, file_bytes in model_bytes.items():
                    st.download_button(
                        label=f"Download {filename}",
                        data=file_bytes,
                        file_name=filename,
                        mime="application/octet-stream"
                    )
                
                status_text.text("✅ Training completed successfully!")
                st.session_state['model_trained'] = True
                #st.session_state['model_path'] = output_dir
                #st.session_state['base_model_name'] = model_name
                
                # Store model path in session state and also save to a file for persistence
                st.session_state['model_path'] = model_dir
                st.session_state['base_model_name'] = model_name

                # Save paths information to disk for persistence
                paths_info = {'model_path': model_dir,
                              'base_model_name': model_name
                              }
                # Show beta value
                beta_value = torch.sigmoid(model.beta).item()
                st.write(f"Learned mixture weight (beta): {beta_value:.4f}")
                st.write(f"This means the model combines {beta_value*100:.1f}% from quantized base model and {(1-beta_value)*100:.1f}% from the side network.")
                with open(os.path.join(model_dir, "paths_info.json"), 'w') as f:
                    json.dump(paths_info, f)

                status_text.text(f"✅ Training completed successfully! Model saved to {model_dir}")
                
            except Exception as e:
                st.error(f"Error during training: {str(e)}")
                raise e
            finally:
                if dataset_name == "custom" and 'custom_file' in locals():
                    os.unlink(custom_file)

elif page == "Generate Text":
    st.header("Generate Text with QST Model")
    
    # Check if model is trained
    if not st.session_state.get('model_trained', False):
        st.warning("No trained model found. Please train a model first.")
        if st.button("Go to Training Page"):
            st.session_state['page'] = "Train Model"
            st.experimental_rerun()
    else:
        st.success("Using model trained in this session.")
        
        # Model settings
        prompt = st.text_area("Enter your prompt:", "Once upon a time,")
        max_length = st.slider("Maximum generation length", 10, 200, 50)
        
        if st.button("Generate Text"):
            with st.spinner("Generating text..."):
                try:
                    # Set device
                    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                    # Verify the model files exist before loading
                    side_network_path = os.path.join(model_dir, "side_network.pt")
                    beta_path = os.path.join(model_dir, "beta.pt")
            
                    if not os.path.exists(side_network_path):
                        st.error(f"Model file not found: {side_network_path}")
                        st.stop()
                
                    if not os.path.exists(beta_path):
                        st.error(f"Model file not found: {beta_path}")
                        st.stop()
            
                    st.info(f"Loading model files from {model_dir}")
            
                    # Load model and tokenizer
                    model = QuantizedSideTuning.from_pretrained(
                        base_model_name, 
                        model_dir,
                        device=device
                        )
                    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
                    # Load model and tokenizer
                    #model_name = st.session_state.get('base_model_name', "distilgpt2")
                    #output_dir = st.session_state.get('model_path')
                    
                    
                    # Generate text
                    model.eval()
                    inputs = tokenizer(prompt, return_tensors="pt").to(device)
                    
                    with torch.no_grad():
                        input_ids = inputs.input_ids
                        attention_mask = inputs.attention_mask
                        generated_ids = []
                        
                        for _ in range(max_length):
                            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                            next_token_logits = outputs[:, -1, :]
                            next_token = torch.argmax(next_token_logits, dim=-1).unsqueeze(-1)
                            
                            generated_ids.append(next_token.item())
                            input_ids = torch.cat([input_ids, next_token], dim=-1)
                            attention_mask = torch.cat([
                                attention_mask, 
                                attention_mask.new_ones((attention_mask.shape[0], 1))
                            ], dim=-1)
                            
                            if next_token.item() == tokenizer.eos_token_id:
                                break
                    
                    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
                    
                    # Display results
                    st.subheader("Generated Text:")
                    st.write(f"{prompt}{generated_text}")
                    
                    # Show the full text in an expander
                    with st.expander("View complete text"):
                        st.write(f"{prompt}{generated_text}")
                    
                except Exception as e:
                    st.error(f"Error during generation: {str(e)}")

elif page == "About QST":
    st.header("About Quantized Side Tuning")
     # Add tabs for different visualizations
    tab1, tab2, tab3, tab4 = st.tabs(["Overview", "Architecture", "Training Flow", "Inference Flow"])
    
    with tab1:
        st.markdown("""
        ### Quantized Side Tuning: Technical Overview
        
        Quantized Side Tuning (QST) is an approach to adapt pre-trained language models while maintaining their quantized state to save memory and improve inference speed.
        
        #### How It Works:
        
        1. **Quantized Base Model**: 
           - Keeps the original LLM in a quantized format (INT8/INT4)
           - Freezes all parameters
           - Uses its hidden states as input to the side network
        
        2. **Side Network**:
           - Smaller network trained in full precision
           - Processes the hidden states from the base model
           - Produces representations that complement the base model
        
        3. **Fusion Mechanism**:
           - Combines outputs from both networks
           - Uses learned parameters to balance contributions
        
        #### Advantages over Other Methods:
        
        | Method | Memory | Speed | Performance | Flexibility |
        |--------|--------|-------|-------------|------------|
        | Full Fine-tuning | 🔴 High | 🔴 Slow | 🟢 Best | 🟡 Medium |
        | LoRA | 🟡 Medium | 🟢 Fast | 🟡 Good | 🟡 Medium |
        | QAT | 🟡 Medium | 🔴 Very Slow | 🟡 Good | 🔴 Low |
        | **QST** | 🟢 Low | 🟢 Fast | 🟡 Good | 🟢 High |
        
        #### Best Use Cases:
        
        - Resource-constrained environments
        - Edge deployment scenarios
        - Rapid prototyping and experimentation
        - Applications requiring quantized inference
        """)
    
    with tab2:
        st.subheader("QST Architecture")
        st.graphviz_chart(display_qst_architecture())
        
        st.markdown("""
        ### Architecture Components:
        
        #### 1. Quantized Base Model
        - Pre-trained LLM converted to INT8/INT4 precision
        - All parameters frozen during training
        - Provides feature-rich representations but with quantization artifacts
        
        #### 2. Downsampling Layers
        - Reduce dimensionality of hidden states from the base model
        - Make computation more efficient in the side network
        - Preserve sequence length for alignment with base model
        
        #### 3. Side Network
        - Small, trainable network in full FP32/FP16 precision
        - Learns to complement the base model's capabilities
        - Compensates for quantization errors and adapts to new tasks
        
        #### 4. Fusion Mechanism
        - Learns optimal weighting (β) between base and side outputs
        - Balances contributions based on task requirements
        - Simple weighted sum: `output = β * base_output + (1-β) * side_output`
        """)
    
    with tab3:
        st.subheader("QST Training Flow")
        st.graphviz_chart(display_qst_flow())
        
        st.markdown("""
        ### Training Process:
        
        1. **Model Preparation**
           - Load pre-trained model
           - Apply quantization (INT8/INT4)
           - Freeze all quantized parameters
        
        2. **Side Network Training**
           - Initialize small side network
           - Forward pass: Extract hidden states from quantized model
           - Process hidden states through side network
           - Combine outputs with fusion mechanism
           - Calculate loss and update only side network parameters
        
        3. **Fusion Parameter Adjustment**
           - β parameter is learned during training
           - Automatically finds optimal balance between models
           - Lower β values indicate stronger reliance on side network
        
        4. **Evaluation and Saving**
           - Save only side network parameters and fusion weights
           - Quantized base model remains unchanged
           - Significantly smaller storage footprint than full fine-tuning
        """)
    
    with tab4:
        st.subheader("QST Inference Flow")
        st.graphviz_chart(display_qst_inference())
        
        st.markdown("""
        ### Inference Process:
        
        1. **Input Processing**
           - Text input is tokenized
           - Tokens are fed to both model components
        
        2. **Parallel Processing**
           - Base model processes input at INT8/INT4 precision
           - Side network receives base model hidden states
           - Both produce complementary representations
        
        3. **Fusion and Generation**
           - Outputs are combined using learned β parameter
           - Final logits determine next token probabilities
           - Tokens are decoded back to text
        
        ### Code Architecture:
        
        This implementation follows a modular design with:
        - `DownsampleLayer`: Reduces dimensions for efficient processing
        - `SideNetwork`: Processes and learns from base model states
        - `QuantizedSideTuning`: Main model integrating quantized base and side network
        """)
# Initialize session state for model path
if 'model_path' not in st.session_state:
    st.session_state['model_path'] = None

if 'model_trained' not in st.session_state:
    st.session_state['model_trained'] = False
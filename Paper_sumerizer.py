import os
import streamlit as st
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain.prompts import PromptTemplate
from langchain.schema import HumanMessage

# Load environment variables
load_dotenv()
hf_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")

if not hf_token:
    st.error("❌ HUGGINGFACEHUB_API_TOKEN not found in .env")
    st.stop()

# Page config
st.set_page_config(
    page_title="Research Paper Summarizer - LangChain",
    page_icon="📚",
    layout="wide"
)

st.title("📚 Paper Summarizer with LangChain")

# Simple model configs - fixed parameter issue
MODELS = {
    "Zephyr-7B": "HuggingFaceH4/zephyr-7b-beta",
    "Flan-T5": "google/flan-t5-large",
    "Mistral-7B": "mistralai/Mistral-7B-Instruct-v0.1"
}

# Model selection
selected_model = st.selectbox("Choose Model:", list(MODELS.keys()))
model_id = MODELS[selected_model]

# Initialize model - simplified parameters
@st.cache_resource
def get_model(model_name):
    try:
        # Create HuggingFaceEndpoint with explicit parameters (not in model_kwargs)
        llm = HuggingFaceEndpoint(
            repo_id=MODELS[model_name],
            huggingfacehub_api_token=hf_token,
            temperature=0.7,
            max_new_tokens=400,
            timeout=60
        )
        
        # Use ChatHuggingFace for better conversation
        if "flan-t5" not in model_name.lower():
            chat_llm = ChatHuggingFace(llm=llm)
            return chat_llm, True
        else:
            return llm, False
            
    except Exception as e:
        st.error(f"Error initializing model: {e}")
        return None, False

# Get the model
model, is_chat = get_model(selected_model)

if model is None:
    st.stop()

# Simple prompt template
prompt_template = PromptTemplate(
    input_variables=["paper_title", "style"],
    template="""
Summarize the research paper titled: {paper_title}

Style: {style}

Provide a clear summary covering:
1. Main contribution
2. Key methodology 
3. Important results
4. Significance

Summary:
"""
)

# Paper selection
RESEARCH_PAPERS = [
    "Attention Is All You Need",
    "BERT: Pre-training of Deep Bidirectional Transformers",
    "GPT-3: Language Models are Few-Shot Learners", 
    "ResNet: Deep Residual Learning for Image Recognition",
    "YOLO: Real-Time Object Detection",
    "Diffusion Models Beat GANs on Image Synthesis",
    "AlexNet: ImageNet Classification with Deep CNNs",
    "Custom Paper (Enter Below)"
]

selected_paper = st.selectbox("Select Research Paper:", RESEARCH_PAPERS)

# Custom paper input
if selected_paper == "Custom Paper (Enter Below)":
    paper_title = st.text_input("Enter Custom Paper Title:", placeholder="Type your paper title here...")
else:
    paper_title = selected_paper

style = st.selectbox("Style:", ["Simple", "Technical", "Beginner-friendly"])

# Generate summary
if st.button("Generate Summary"):
    if paper_title:
        # Create prompt
        prompt = prompt_template.format(paper_title=paper_title, style=style)
        
        try:
            # Generate based on model type
            if is_chat:
                # Use ChatHuggingFace
                message = HumanMessage(content=prompt)
                response = model.invoke([message])
                result = response.content
            else:
                # Use regular HuggingFaceEndpoint
                result = model.invoke(prompt)
            
            st.success("✅ Summary Generated!")
            st.write("### Summary:")
            
            # Format output with bold headings
            formatted_result = result
            headings_to_bold = [
                "Main contribution", "Main Contribution", "MAIN CONTRIBUTION",
                "Key methodology", "Key Methodology", "KEY METHODOLOGY", 
                "Important results", "Important Results", "IMPORTANT RESULTS",
                "Significance", "SIGNIFICANCE",
                "1.", "2.", "3.", "4."
            ]
            
            for heading in headings_to_bold:
                formatted_result = formatted_result.replace(heading, f"**{heading}**")
            
            st.markdown(formatted_result)
            
        except Exception as e:
            st.error(f"Error generating summary: {e}")
    else:
        st.warning("Please enter a paper title")

# Show the prompt being used
with st.expander("View Prompt Template"):
    if paper_title:
        st.code(prompt_template.format(paper_title=paper_title, style=style))
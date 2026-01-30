import streamlit as st
from query_rag import answer_question

# 🎨 Streamlit App UI
st.set_page_config(page_title="GPT-5 RAG Teaching Assistant", page_icon="🤖", layout="wide")
st.title("🎓 GPT-5 RAG Teaching Assistant")
st.caption("Ask normal, fill-in-the-blank, or multiple-choice questions using your PDF knowledge base.")

# Select mode
mode = st.selectbox(
    "Select Question Type:",
    ["Normal Q&A", "Fill in the Blank", "Multiple Choice Question"]
)

query = st.text_area("Enter your question:", placeholder="Type your question here...")

# If MCQ mode, dynamically add options
options = {}
if mode == "Multiple Choice Question":
    st.subheader("Add Multiple Choice Options")
    num_options = st.number_input("Number of Options:", min_value=2, max_value=10, value=4, step=1)
    
    for i in range(num_options):
        label = chr(65 + i)  # 'A', 'B', 'C', etc.
        value = st.text_input(f"Option {label}:", placeholder=f"Enter option {label}")
        if value:
            options[label] = value

# Submit button
if st.button("Get Answer"):
    if not query.strip():
        st.warning("Please enter a question first.")
    else:
        with st.spinner("🤔 Thinking..."):
            try:
                # Determine mode for backend
                mode_key = (
                    "mcq" if mode == "Multiple Choice Question"
                    else "fill" if mode == "Fill in the Blank"
                    else "normal"
                )

                answer = answer_question(query, options=options if options else None, mode=mode_key)

                st.markdown("### 🧩 **Answer:**")
                st.success(answer)

            except Exception as e:
                st.error(f"⚠️ Error: {str(e)}")

st.markdown("---")
st.markdown("**Powered by GPT-5 RAG | Built with Flask + FAISS + Streamlit** 🚀")

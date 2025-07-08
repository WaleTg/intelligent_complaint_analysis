import gradio as gr
from rag_pipeline import retrieve_and_generate  # your existing function

def chat_interface(question):
    if not question.strip():
        return "Please enter a question.", []
    
    answer, sources = retrieve_and_generate(question)
    
    # Format sources nicely for display
    formatted_sources = "\n\n---\n\n".join(sources)
    
    return answer, formatted_sources

with gr.Blocks() as demo:
    gr.Markdown("# CrediTrust Complaint Analysis Chatbot")
    chatbot = gr.Textbox(label="Ask your question about customer complaints:")
    
    output_answer = gr.Textbox(label="Answer", interactive=False)
    output_sources = gr.Textbox(label="Source Contexts", interactive=False)
    
    submit_btn = gr.Button("Ask")
    clear_btn = gr.Button("Clear")

    submit_btn.click(fn=chat_interface, inputs=chatbot, outputs=[output_answer, output_sources])
    clear_btn.click(fn=lambda: ("", ""), inputs=None, outputs=[output_answer, output_sources, chatbot])

demo.launch()

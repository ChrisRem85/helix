import gradio as gr
import os

# Function to generate UI for uploaded images
def generate_time_inputs(files):
    if files is None:
        return "", "No files uploaded"
    
    # Generate HTML for drag-and-drop reordering
    components_html = """
    <ul id='sortable' style='list-style: none; padding: 0;'>
    """
    for f in files:
        file_name = os.path.basename(f.name)
        components_html += f"""
        <li style='margin: 10px 0; padding: 10px; border: 1px solid #ccc; cursor: grab;' data-file='{f.name}'>
            <img src='{f.name}' alt='{file_name}' style='max-width: 100px; height: auto; display: block;' />
            <span>{file_name}</span>
        </li>
        """
    components_html += "</ul>"
    
    # Add JavaScript for drag-and-drop functionality
    components_html += """
    <script src="https://cdnjs.cloudflare.com/ajax/libs/Sortable/1.15.0/Sortable.min.js"></script>
    <script>
        const sortable = document.getElementById('sortable');
        new Sortable(sortable, {
            animation: 150,
            ghostClass: 'sortable-ghost',
            onEnd: function (evt) {
                const items = Array.from(sortable.children);
                const order = items.map(item => item.getAttribute('data-file'));
                document.getElementById('file_order').value = JSON.stringify(order);
            }
        });
    </script>
    """
    
    return components_html, "Drag and drop to rearrange the files."

# Process results based on reordered files
def process(files, file_order_json):
    if files is None:
        return "No images uploaded."
    
    # Parse the reordered file list
    file_order = []
    if file_order_json:
        file_order = eval(file_order_json)
    
    results = []
    for f in file_order:
        results.append(f"File: {os.path.basename(f)}")
    
    return "\n".join(results)


with gr.Blocks() as demo:
    # Title
    gr.Markdown("# 🧬 SkinStager")
    
    # Upload section
    gr.Markdown("## Upload Images")
    file_input = gr.File(file_count="multiple", file_types=["image"])
    
    # Intermediate message
    upload_message = gr.Markdown("")
    
    # Dynamic container for drag-and-drop
    gr.Markdown("## Reorder Images Based on Time Points")
    dynamic_area = gr.HTML()  # Use gr.HTML for custom drag-and-drop content
    
    # Hidden input to store reordered file list
    file_order = gr.Textbox(visible=False, elem_id="file_order")
    
    # Results
    gr.Markdown("## Results")
    output = gr.Textbox(lines=10)
    
    btn = gr.Button("Run Analysis")
    
    # When files uploaded → generate UI and show message
    file_input.change(
        fn=generate_time_inputs,
        inputs=file_input,
        outputs=[dynamic_area, upload_message]
    )
    
    # Button click → process reordered files
    btn.click(
        fn=process,
        inputs=[file_input, file_order],
        outputs=output
    )

demo.launch()
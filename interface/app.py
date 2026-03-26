import gradio as gr
import json
import os

# Injected once at page load via gr.Blocks(js=...).
# Loads SortableJS, then uses a MutationObserver to detect when #sortable appears
# or is replaced (happens every time new files are uploaded), and re-initialises
# Sortable on it.  On drag-end it syncs the new order into the hidden Gradio
# textbox so Python sees it when the button is clicked.
PAGE_JS = """
(function () {
    var script = document.createElement('script');
    script.src = 'https://cdnjs.cloudflare.com/ajax/libs/Sortable/1.15.0/Sortable.min.js';
    script.onload = function () {
        function pushOrder(list) {
            var order = Array.from(list.querySelectorAll('li')).map(function (li) {
                return li.getAttribute('data-file');
            });
            var ta = document.querySelector('#file_order textarea');
            if (ta) {
                // Use the native setter so Svelte's two-way binding notices the change
                var nativeSetter = Object.getOwnPropertyDescriptor(
                    window.HTMLTextAreaElement.prototype, 'value'
                ).set;
                nativeSetter.call(ta, JSON.stringify(order));
                ta.dispatchEvent(new Event('input', { bubbles: true }));
            }
        }

        function attachSortable(list) {
            if (list._sortableReady) return;
            list._sortableReady = true;
            new Sortable(list, {
                animation: 150,
                ghostClass: 'sortable-ghost',
                chosenClass: 'sortable-chosen',
                onEnd: function () { pushOrder(list); }
            });
        }

        // Watch for #sortable being added or replaced in the DOM
        var observer = new MutationObserver(function () {
            var list = document.getElementById('sortable');
            if (list) attachSortable(list);
        });
        observer.observe(document.body, { childList: true, subtree: true });

        // Also handle the case where the list is already present on load
        var existing = document.getElementById('sortable');
        if (existing) attachSortable(existing);
    };
    document.head.appendChild(script);
})();
"""


def generate_time_inputs(files):
    if files is None:
        return "", "No files uploaded.", ""

    file_entries = [{"name": os.path.basename(f.name), "path": f.name} for f in files]
    initial_order = json.dumps([e["path"] for e in file_entries])

    items_html = ""
    for entry in file_entries:
        items_html += (
            f"<li style='margin:8px 0;padding:10px;border:1px solid #ccc;"
            f"border-radius:6px;cursor:grab;display:flex;align-items:center;"
            f"gap:12px;background:#fff;user-select:none;' data-file='{entry['path']}'>"
            f"<img src='/file={entry['path']}' alt='{entry['name']}' "
            f"style='width:80px;height:80px;object-fit:cover;border-radius:4px;flex-shrink:0;' />"
            f"<span style='font-weight:500'>{entry['name']}</span></li>"
        )

    components_html = (
        "<style>"
        "#sortable{list-style:none;padding:0;margin:0;}"
        ".sortable-ghost{opacity:0.25;}"
        ".sortable-chosen{box-shadow:0 4px 14px rgba(0,0,0,0.18);}"
        "</style>"
        f"<ul id='sortable'>{items_html}</ul>"
    )

    return components_html, "Drag and drop images to set the time series order, then click Run Analysis.", initial_order


def process(files, file_order_json):
    if files is None:
        return "No images uploaded."

    if file_order_json:
        try:
            ordered_paths = json.loads(file_order_json)
        except json.JSONDecodeError:
            ordered_paths = [f.name for f in files]
    else:
        ordered_paths = [f.name for f in files]

    lines = ["Time series order:"]
    for i, path in enumerate(ordered_paths, 1):
        lines.append(f"  {i}. {os.path.basename(path)}")

    return "\n".join(lines)


with gr.Blocks(js=PAGE_JS) as demo:
    gr.Markdown("# SkinStager")

    gr.Markdown("## Upload Images")
    file_input = gr.File(file_count="multiple", file_types=["image"])

    upload_message = gr.Markdown("")

    gr.Markdown("## Reorder Images Based on Time Points")
    dynamic_area = gr.HTML()

    # Stores the JSON-encoded ordered file list; updated by JS on drag-and-drop
    file_order = gr.Textbox(visible=False, elem_id="file_order")

    gr.Markdown("## Results")
    output = gr.Textbox(lines=10)

    btn = gr.Button("Run Analysis")

    file_input.change(
        fn=generate_time_inputs,
        inputs=file_input,
        outputs=[dynamic_area, upload_message, file_order],
    )

    btn.click(
        fn=process,
        inputs=[file_input, file_order],
        outputs=output,
    )

demo.launch()
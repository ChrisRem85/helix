import gradio as gr
import json
import os
import re
from datetime import datetime

_DE_MONTHS = {
    "januar": 1, "jan": 1,
    "februar": 2, "feb": 2,
    "maerz": 3, "mrz": 3,
    "april": 4, "apr": 4,
    "mai": 5,
    "juni": 6, "jun": 6,
    "juli": 7, "jul": 7,
    "august": 8, "aug": 8,
    "september": 9, "sep": 9, "sept": 9,
    "oktober": 10, "okt": 10, "oct": 10,
    "november": 11, "nov": 11,
    "dezember": 12, "dez": 12, "dec": 12,
}

def extract_date_from_filename(filename):
    name = os.path.splitext(filename)[0]
    patterns = [
        (r"(\d{1,2})\.(\d{1,2})\.(\d{4})",
         lambda m: datetime(int(m.group(3)), int(m.group(2)), int(m.group(1)))),
        (r"(\d{1,2})\.(\d{1,2})\.(\d{2})",
         lambda m: datetime(2000+int(m.group(3)), int(m.group(2)), int(m.group(1)))),
        (r"(\d{4})[_\-](\d{2})[_\-](\d{2})",
         lambda m: datetime(int(m.group(1)), int(m.group(2)), int(m.group(3)))),
        (r"(?<!\d)(20\d{2})(0[1-9]|1[0-2])(0[1-9]|[12]\d|3[01])(?!\d)",
         lambda m: datetime(int(m.group(1)), int(m.group(2)), int(m.group(3)))),
        (r"(\d{1,2})[_\-](\d{1,2})[_\-](\d{4})",
         lambda m: datetime(int(m.group(3)), int(m.group(2)), int(m.group(1)))),
    ]
    for pattern, parser in patterns:
        m = re.search(pattern, name)
        if m:
            try:
                return parser(m).strftime("%Y-%m-%d")
            except ValueError:
                continue
    name_lower = name.lower()
    for month_name, month_num in _DE_MONTHS.items():
        sep = r"[.\s_\-]?"
        m = re.search(rf"(\d{{1,2}}){sep}{re.escape(month_name)}{sep}(\d{{4}})", name_lower)
        if m:
            try:
                return datetime(int(m.group(2)), month_num, int(m.group(1))).strftime("%Y-%m-%d")
            except ValueError:
                pass
        m = re.search(rf"(\d{{4}}){sep}{re.escape(month_name)}", name_lower)
        if m:
            return datetime(int(m.group(1)), month_num, 1).strftime("%Y-%m-%d")
        m = re.search(rf"{re.escape(month_name)}{sep}(\d{{4}})", name_lower)
        if m:
            return datetime(int(m.group(1)), month_num, 1).strftime("%Y-%m-%d")
    return ""

PAGE_JS = """
(function () {
    function parseGermanDate(str) {
        var m = (str || "").match(/^(\\d{1,2})\\.(\\d{1,2})\\.(\\d{4})$/);
        if (!m) return "";
        var d = m[1].padStart(2,"0"), mo = m[2].padStart(2,"0"), y = m[3];
        var dt = new Date(y + "-" + mo + "-" + d);
        return isNaN(dt.getTime()) ? "" : y + "-" + mo + "-" + d;
    }
    function isChronological(list) {
        var dates = Array.from(list.querySelectorAll("li")).map(function (li) {
            return li.getAttribute("data-iso") || "";
        });
        for (var i = 1; i < dates.length; i++) {
            if (dates[i] && dates[i-1] && dates[i] < dates[i-1]) return false;
        }
        return true;
    }
    function updateWarning(list) {
        var warn = document.getElementById("order-warning");
        if (!warn) return;
        var hasDate = Array.from(list.querySelectorAll("li")).some(function (li) {
            return !!li.getAttribute("data-iso");
        });
        warn.style.display = (hasDate && !isChronological(list)) ? "block" : "none";
    }
    function pushOrder(list) {
        var order = Array.from(list.querySelectorAll("li")).map(function (li) {
            return { path: li.getAttribute("data-file"), date: li.getAttribute("data-iso") || "" };
        });
        var ta = document.querySelector("#file_order textarea");
        if (ta) {
            var nativeSetter = Object.getOwnPropertyDescriptor(window.HTMLTextAreaElement.prototype, "value").set;
            nativeSetter.call(ta, JSON.stringify(order));
            ta.dispatchEvent(new Event("input", { bubbles: true }));
        }
        updateWarning(list);
    }
    function sortByDate(list) {
        var items = Array.from(list.querySelectorAll("li"));
        items.sort(function (a, b) {
            var da = a.getAttribute("data-iso") || "9999-99-99";
            var db = b.getAttribute("data-iso") || "9999-99-99";
            return da < db ? -1 : da > db ? 1 : 0;
        });
        items.forEach(function (item) { list.appendChild(item); });
        pushOrder(list);
    }
    function attachSortable(list) {
        if (list._sortableReady) return;
        list._sortableReady = true;
        var script = document.createElement("script");
        script.src = "https://cdnjs.cloudflare.com/ajax/libs/Sortable/1.15.0/Sortable.min.js";
        script.onload = function () {
            new Sortable(list, {
                animation: 150, ghostClass: "sortable-ghost", chosenClass: "sortable-chosen", handle: ".drag-handle",
                onEnd: function () { pushOrder(list); }
            });
        };
        document.head.appendChild(script);
        list.addEventListener("change", function (e) {
            if (e.target.classList.contains("date-input")) {
                var iso = parseGermanDate(e.target.value);
                var li = e.target.closest("li");
                li.setAttribute("data-iso", iso);
                var ind = li.querySelector(".date-indicator");
                if (ind) { ind.textContent = iso ? "\u2713" : "?"; ind.style.color = iso ? "#4caf50" : "#ff9800"; }
                sortByDate(list);
            }
        });
    }
    var observer = new MutationObserver(function () { var list = document.getElementById("sortable"); if (list && !list._sortableReady) attachSortable(list); });
    observer.observe(document.body, { childList: true, subtree: true });
    var existing = document.getElementById("sortable");
    if (existing) attachSortable(existing);
})();
"""

def _iso_to_german(iso_date):
    """Convert YYYY-MM-DD to DD.MM.YYYY"""
    m = re.match(r'^(\d{4})-(\d{2})-(\d{2})$', iso_date or "")
    return f"{m.group(3)}.{m.group(2)}.{m.group(1)}" if m else ""


def generate_time_inputs(files):
    if files is None:
        return "", "No files uploaded.", ""
    file_entries = [
        {"name": os.path.basename(f.name), "path": f.name, "date": extract_date_from_filename(os.path.basename(f.name))}
        for f in files
    ]
    file_entries.sort(key=lambda e: e["date"] if e["date"] else "9999-99-99")
    initial_order = json.dumps([{"path": e["path"], "date": e["date"]} for e in file_entries])
    items_html = ""
    for entry in file_entries:
        date_iso = entry["date"]
        date_de = _iso_to_german(date_iso)
        indicator = "&#10003;" if date_iso else "?"
        ind_color = "#4caf50" if date_iso else "#ff9800"
        items_html += (
            f"<li style='margin:8px 0;padding:10px;border:1px solid #ccc;border-radius:6px;"
            f"display:flex;align-items:center;gap:12px;background:#fff;user-select:none;' "
            f"data-file='{entry['path']}' data-iso='{date_iso}'>"
            f"<span class='drag-handle' title='Drag to reorder' style='cursor:grab;font-size:20px;color:#bbb;padding:0 4px;flex-shrink:0;'>&#9776;</span>"
            f"<img src='/file={entry['path']}' alt='{entry['name']}' style='width:80px;height:80px;object-fit:cover;border-radius:4px;flex-shrink:0;pointer-events:none;' />"
            f"<div style='flex:1;min-width:0;'>"
            f"<div style='font-weight:500;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;margin-bottom:6px;' title='{entry['name']}'>{entry['name']}</div>"
            f"<div style='display:flex;align-items:center;gap:6px;'>"
            f"<span class='date-indicator' style='color:{ind_color};font-weight:bold;font-size:16px;'>{indicator}</span>"
            f"<input type='text' class='date-input' value='{date_de}' placeholder='TT.MM.JJJJ' "
            f"style='border:1px solid #ccc;border-radius:4px;padding:3px 6px;font-size:13px;width:110px;' />"
            f"</div></div></li>"
        )
    components_html = (
        "<style>#sortable{list-style:none;padding:0;margin:0;} .sortable-ghost{opacity:0.25;} .sortable-chosen{box-shadow:0 4px 14px rgba(0,0,0,0.18);}</style>"
        "<div id='order-warning' style='display:none;background:#fff3cd;border:1px solid #ffc107;"
        "border-radius:6px;padding:10px 14px;margin-bottom:10px;color:#856404;font-weight:500;'>"
        "&#9888; Die Reihenfolge entspricht nicht der chronologischen Datumsreihenfolge.</div>"
        f"<ul id='sortable'>{items_html}</ul>"
    )
    n_found = sum(1 for e in file_entries if e["date"])
    msg = f"Dates extracted for **{n_found}/{len(file_entries)}** images. Edit any date manually — the list re-sorts automatically. Drag &#9776; to reorder manually."
    return components_html, msg, initial_order

def process(files, file_order_json):
    if files is None:
        return "No images uploaded."
    ordered_entries = []
    if file_order_json:
        try:
            parsed = json.loads(file_order_json)
            if parsed and isinstance(parsed[0], str):
                ordered_entries = [{"path": p, "date": ""} for p in parsed]
            else:
                ordered_entries = parsed
        except (json.JSONDecodeError, IndexError):
            pass
    if not ordered_entries:
        ordered_entries = [{"path": f.name, "date": ""} for f in files]
    dates = [e.get("date", "") for e in ordered_entries]
    is_chronological = all(
        not dates[i] or not dates[i - 1] or dates[i - 1] <= dates[i]
        for i in range(1, len(dates))
    )
    lines = ["Time series order:"]
    if any(dates) and not is_chronological:
        lines.append("  \u26a0 WARNUNG: Bilder sind nicht in chronologischer Datumsreihenfolge!")
    for i, entry in enumerate(ordered_entries, 1):
        name = os.path.basename(entry["path"])
        date = entry.get("date", "")
        lines.append(f"  {i}. {name} [{date if date else 'Datum unbekannt'}]")
    return "\n".join(lines)

with gr.Blocks(js=PAGE_JS) as demo:
    gr.Markdown("# SkinStager")
    gr.Markdown("## Upload Images")
    file_input = gr.File(file_count="multiple", file_types=["image"])
    upload_message = gr.Markdown("")
    gr.Markdown("## Time Series Order")
    dynamic_area = gr.HTML()
    file_order = gr.Textbox(visible=False, elem_id="file_order")
    gr.Markdown("## Results")
    output = gr.Textbox(lines=10)
    btn = gr.Button("Run Analysis")
    file_input.change(fn=generate_time_inputs, inputs=file_input, outputs=[dynamic_area, upload_message, file_order])
    btn.click(fn=process, inputs=[file_input, file_order], outputs=output)

demo.launch()

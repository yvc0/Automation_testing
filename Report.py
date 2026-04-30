import json
from datetime import datetime

def save_json(data, path):
    with open(path, "w") as f:
        json.dump(data, f, indent=2)

def save_html(summary, path, title):
    html = f"<h2>{title}</h2><table border='1'>"
    html += "<tr><th>Metric</th><th>Value</th></tr>"

    for k, v in summary.items():
        html += f"<tr><td>{k}</td><td>{round(v,4)}</td></tr>"

    html += f"</table><p>{datetime.now()}</p>"

    with open(path, "w") as f:
        f.write(html)

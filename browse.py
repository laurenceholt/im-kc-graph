#!/usr/bin/env python3
"""
Simple web viewer for the questions database.
Browse all questions with inline image thumbnails.

Usage:
    python browse.py
    # Then open http://localhost:5111
"""
import os
import base64
from flask import Flask, Response
import psycopg2
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)
DATABASE_URL = os.environ["DATABASE_URL"]


def get_conn():
    return psycopg2.connect(DATABASE_URL)


@app.route("/")
def index():
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("""
        SELECT id, filename, grade, unit, section,
               assessment_number, question_number,
               image_width, image_height,
               length(image_data)/1024 as size_kb
        FROM questions
        ORDER BY grade, unit, section NULLS FIRST,
                 filename, assessment_number, question_number
    """)
    rows = cur.fetchall()
    conn.close()

    html = """<!DOCTYPE html>
<html><head>
<title>IM Questions Browser</title>
<style>
  body { font-family: -apple-system, system-ui, sans-serif; margin: 20px; background: #f5f5f5; }
  h1 { color: #333; }
  .stats { color: #666; margin-bottom: 20px; }
  table { border-collapse: collapse; background: white; box-shadow: 0 1px 3px rgba(0,0,0,0.1); width: 100%; }
  th { background: #e8552e; color: white; padding: 10px 14px; text-align: left; position: sticky; top: 0; }
  td { padding: 8px 14px; border-bottom: 1px solid #eee; vertical-align: top; }
  tr:hover { background: #fef7f5; }
  .topic { background: #e8f4fd; color: #1a73e8; padding: 2px 8px; border-radius: 10px; font-size: 0.85em; }
  .no-topic { color: #aaa; font-style: italic; }
  .thumb { max-width: 350px; max-height: 250px; border: 1px solid #ddd; border-radius: 4px; cursor: pointer; transition: transform 0.2s; }
  .thumb:hover { transform: scale(1.02); box-shadow: 0 2px 8px rgba(0,0,0,0.15); }
  .meta { font-size: 0.85em; color: #888; }
  /* Modal for full-size view */
  .modal { display:none; position:fixed; top:0; left:0; width:100%; height:100%; background:rgba(0,0,0,0.8); z-index:100; justify-content:center; align-items:center; cursor:pointer; }
  .modal.active { display:flex; }
  .modal img { max-width:95vw; max-height:95vh; border-radius:8px; }
</style>
</head><body>
<h1>IM Assessments</h1>
<div class="stats">""" + f"{len(rows)} questions" + """</div>
<table>
<tr><th>Grade</th><th>Unit</th><th>Section</th><th>Assessment</th><th>Q#</th><th>Image</th><th>Details</th></tr>
"""
    for row in rows:
        id_, fname, grade, unit, section, assess, qnum, w, h, size_kb = row
        section_html = f'<span class="topic">{section}</span>' if section else '<span class="no-topic">full unit</span>'
        html += f"""<tr>
  <td><strong>{grade}</strong></td>
  <td>{unit}</td>
  <td>{section_html}</td>
  <td>{assess}</td>
  <td><strong>{qnum}</strong></td>
  <td><img class="thumb" src="/image/{id_}" onclick="showModal(this.src)" loading="lazy"></td>
  <td class="meta">{w}&times;{h}<br>{size_kb} KB<br>{fname}</td>
</tr>"""

    html += """</table>
<div class="modal" id="modal" onclick="this.classList.remove('active')">
  <img id="modal-img" src="">
</div>
<script>
function showModal(src) {
  document.getElementById('modal-img').src = src;
  document.getElementById('modal').classList.add('active');
}
</script>
</body></html>"""
    return html


@app.route("/image/<int:question_id>")
def image(question_id):
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("SELECT image_data FROM questions WHERE id = %s", (question_id,))
    row = cur.fetchone()
    conn.close()
    if not row:
        return "Not found", 404
    return Response(bytes(row[0]), mimetype="image/png")


if __name__ == "__main__":
    print("Starting browser at http://localhost:5111")
    app.run(port=5111, debug=False)

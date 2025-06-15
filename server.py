from flask import Flask, request, send_file, jsonify
from flask_cors import CORS
from queue import Queue
from threading import Thread
import subprocess
import os
import time

app = Flask(__name__)
CORS(app)

job_queue = Queue()
OUTPUT_FOLDER = os.path.join(os.path.dirname(__file__), "outputs")
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

def worker():
    while True:
        job = job_queue.get()
        if job is None:
            break
        prompt, output_path = job
        subprocess.run([
            "python", "ai.py",
            "--prompt", prompt,
            "--output", output_path,
            "--model", "realisticStockPhoto_v20.safetensors",
            "--lora", "SDXL_FILM_PHOTOGRAPHY_STYLE_V1.safetensors",
            "--width", "1024",
            "--height", "1024",
            "--device", "cuda"
        ])
        job_queue.task_done()

Thread(target=worker, daemon=True).start()

@app.route("/ping", methods=["GET"])
def ping():
    return "pong"

@app.route("/generate", methods=["POST"])
def generate():
    prompt = request.form.get("prompt")
    if not prompt:
        return "Prompt lipsă", 400

    timestamp = int(time.time())
    output_path = os.path.join(OUTPUT_FOLDER, f"output_{timestamp}.png")

    job_queue.put((prompt, output_path))
    position = job_queue.qsize()
    return jsonify({
        "message": f"Imaginea ta este în coadă. Revino în câteva secunde: /outputs/output_{timestamp}.png",
        "url": f"/outputs/output_{timestamp}.png",
        "position": position
    }), 202

@app.route("/outputs/<filename>")
def get_image(filename):
    safe_filename = os.path.basename(filename)  # prevenție dacă vine path nasol
    path = os.path.abspath(os.path.join(OUTPUT_FOLDER, safe_filename))
    print(f"[GET] Caut: {path}")
    if os.path.exists(path):
        return send_file(path, mimetype="image/png")
    return "Imaginea nu e gata încă", 404

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5050)

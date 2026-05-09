import os
import sqlite3
from datetime import datetime
from uuid import uuid4
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO
from pydantic import BaseModel
from PIL import Image


# ================================================================
# FASTAPI APP
# ================================================================
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ================================================================
# KONFIGURASI
# ================================================================
UPLOAD_DIR = "chili_uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

DB_PATH = "chili.db"


# ================================================================
# DATABASE SETUP
# ================================================================
def db_connect():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = db_connect()
    cur = conn.cursor()

    cur.execute("""
        CREATE TABLE IF NOT EXISTS pot_detection (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            pot INTEGER,
            ripe INTEGER,
            unripe INTEGER,
            total INTEGER,
            timestamp TEXT
        )
    """)

    conn.commit()
    conn.close()


init_db()   # create table on startup


def save_pot_detection(pot, ripe, unripe, total):
    conn = db_connect()
    cur = conn.cursor()

    cur.execute("""
        INSERT INTO pot_detection (pot, ripe, unripe, total, timestamp)
        VALUES (?, ?, ?, ?, ?)
    """, (pot, ripe, unripe, total, datetime.now().isoformat()))

    conn.commit()
    conn.close()


# ================================================================
# YOLO MODEL & STATE
# ================================================================
chili_model = YOLO("bestchili.pt")

chili_state = {
    "last_image": None,
    "last_pred": None,
    "count_total": 0,
    "count_ripe": 0,
    "count_unripe": 0
}

logs_state = []
current_pot = {"pot": None}
pot_result = {}


# ================================================================
# LOGGING
# ================================================================
def add_log(text):
    print(text)
    logs_state.append(text)
    if len(logs_state) > 200:
        logs_state.pop(0)


# ================================================================
# IMAGE PROCESSING
# ================================================================
def preprocess_and_rotate(image_path):
    img = Image.open(image_path)
    img = img.rotate(-90, expand=True)
    img.save(image_path)
    return image_path


# ================================================================
# YOLO RENDER
# ================================================================
def save_detected_image(result, src_path):
    try:
        out_path = src_path.replace(".jpg", "_det.jpg")
        result.save(filename=out_path)
        return out_path
    except Exception as e:
        print("Render error:", e)
        return src_path


# ================================================================
# YOLO ANALYZE
# ================================================================
def analyze_chili_boxes(result):
    boxes = result.boxes
    if len(boxes) == 0:
        return -1, 0, 0, 0

    total = len(boxes)
    ripe = sum(1 for b in boxes if int(b.cls) == 0)
    unripe = sum(1 for b in boxes if int(b.cls) == 1)
    best = max(boxes, key=lambda x: float(x.conf))
    best_class = int(best.cls)

    return best_class, total, ripe, unripe


# ================================================================
# AUTO CLEANUP
# ================================================================
def cleanup_uploads(max_files=100):
    files = sorted(
        [os.path.join(UPLOAD_DIR, f) for f in os.listdir(UPLOAD_DIR)],
        key=os.path.getctime
    )
    if len(files) > max_files:
        for f in files[:-max_files]:
            os.remove(f)


# ================================================================
# ENDPOINT POT
# ================================================================
class PotData(BaseModel):
    pot: int


@app.post("/pot")
def post_pot(data: PotData):
    current_pot["pot"] = data.pot
    add_log(f"Scanning POT ({data.pot})")
    return {"status": "ok", "pot": data.pot}


# ================================================================
# UPLOAD DETEKSI CABAI
# ================================================================
@app.post("/chili/upload")
async def upload_chili(file: UploadFile = File(...)):

    # LOGGING
    add_log("memulai")
    if current_pot["pot"] is not None:
        add_log(f"Processing POT {current_pot['pot']}")
    add_log("Processing")

    filename = f"{uuid4()}.jpg"
    filepath = os.path.join(UPLOAD_DIR, filename)

    with open(filepath, "wb") as f:
        f.write(await file.read())

    cleanup_uploads()
    preprocess_and_rotate(filepath)

    try:
        result = chili_model.predict(filepath, verbose=False)[0]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")

    ripeness, total, ripe, unripe = analyze_chili_boxes(result)
    detected_path = save_detected_image(result, filepath)

    chili_state["last_image"] = detected_path
    chili_state["last_pred"] = ripeness
    chili_state["count_total"] = total
    chili_state["count_ripe"] = ripe
    chili_state["count_unripe"] = unripe

    pot_id = current_pot["pot"]

    # ================================================================
    # SIMPAN DATA PER POT (RAM + SQLite)
    # ================================================================
    if pot_id is not None:

        # update RAM
        if pot_id not in pot_result:
            pot_result[pot_id] = {"ripe": 0, "unripe": 0}

        pot_result[pot_id]["ripe"] += ripe
        pot_result[pot_id]["unripe"] += unripe

        add_log(f"POT {pot_id} → ripe={pot_result[pot_id]['ripe']} unripe={pot_result[pot_id]['unripe']}")

        # simpan ke SQLite
        save_pot_detection(pot_id, ripe, unripe, total)

    return {
        "status": "ok",
        "ripeness": ripeness,
        "total_detected": total,
        "ripe": ripe,
        "unripe": unripe,
        "pot": current_pot["pot"],
        "note": "0=ripe, 1=unripe, -1=no chili"
    }


# ================================================================
# GET STATUS
# ================================================================
@app.get("/chili/status")
def chili_status():
    return {
        "last_pred": chili_state["last_pred"],
        "total_detected": chili_state["count_total"],
        "ripe": chili_state["count_ripe"],
        "unripe": chili_state["count_unripe"],
        "note": "0=ripe, 1=unripe, -1=no chili"
    }


# ================================================================
# GET POT RESULT (RAM)
# ================================================================
@app.get("/pot/result")
def get_pot_result():
    return pot_result


# ================================================================
# GET LOG
# ================================================================
@app.get("/log")
def get_log():
    return logs_state


# ================================================================
# GET IMAGE
# ================================================================
@app.get("/chili/image")
def chili_image():
    if chili_state["last_image"] is None:
        raise HTTPException(status_code=404, detail="No image yet")
    return FileResponse(chili_state["last_image"])


# ================================================================
# SQLITE READ & RESET
# ================================================================
@app.get("/pot/db")
def pot_db():
    conn = db_connect()
    cur = conn.cursor()
    cur.execute("SELECT * FROM pot_detection ORDER BY id DESC")
    rows = [dict(row) for row in cur.fetchall()]
    conn.close()
    return rows


@app.delete("/pot/db/reset")
def pot_db_reset():
    conn = db_connect()
    cur = conn.cursor()
    cur.execute("DELETE FROM pot_detection")
    conn.commit()
    conn.close()
    return {"status": "cleared"}


# ================================================================
# DEVICE TIME API
# ================================================================
class TimeData(BaseModel):
    device: str
    datetime: str


time_state = {"last_device": None, "last_datetime": None}


@app.post("/device/time")
def post_time(data: TimeData):
    time_state.update(data.dict())
    return {"status": "ok", **data.dict()}


@app.get("/device/time")
def get_time():
    return time_state


# ================================================================
# DHT SENSOR API
# ================================================================
class DHTData(BaseModel):
    device: str
    temperature: float
    humidity: float


dht_state = {"device": None, "temperature": None, "humidity": None}


@app.post("/sensor/dht")
def post_dht(data: DHTData):
    dht_state.update(data.dict())
    return {"status": "ok", **data.dict()}


@app.get("/sensor/dht")
def get_dht():
    return dht_state

# ============================================
# Halaman Web Viewer
# ============================================
@app.get("/chili/view", response_class=HTMLResponse)
def chili_view():
    return """
    <html>
    <head><title>Chili Viewer</title>
    <style>
        body { font-family: Arial; }
        .container {
            display: flex;
            flex-direction: row;
            gap: 30px;
            align-items: flex-start;
        }
        .card {
            padding: 15px;
            border: 1px solid #ccc;
            border-radius: 10px;
            width: 300px;
            background: #f8f8f8;
        }
        img {
            border-radius: 10px;
            border: 1px solid #ccc;
        }
    </style>
    </head>

    <body>
        <h2>Chili Ripeness Detection</h2>

        <div class="container">

            <div class="card">
                <img id="img" src="/chili/image" width="280" />
                <p>Status: <span id="status">Loading...</span></p>
                <p>Total Detected: <span id="total">0</span></p>
                <p>Ripe (Merah): <span id="ripe">0</span></p>
                <p>Unripe (Hijau): <span id="unripe">0</span></p>
            </div>

            <div class="card">
                <h3>📌 Device Time</h3>
                <p>Device: <span id="t_device">-</span></p>
                <p>Datetime: <span id="t_datetime">-</span></p>

                <h3>🌡️ Sensor DHT</h3>
                <p>Temperature: <span id="s_temp">-</span> °C</p>
                <p>Humidity: <span id="s_hum">-</span> %</p>
                <p>Sensor Device: <span id="s_dev">-</span></p>
            </div>

        </div>

        <script>
            async function update() {
                try {
                    const res = await fetch('/chili/status');
                    const data = await res.json();

                    let text = "";
                    if (data.last_pred === 0) text = "RIPE (Merah)";
                    else if (data.last_pred === 1) text = "UNRIPE (Hijau)";
                    else text = "No chili detected";

                    document.getElementById("status").innerHTML = text;
                    document.getElementById("total").innerHTML = data.total_detected;
                    document.getElementById("ripe").innerHTML = data.ripe;
                    document.getElementById("unripe").innerHTML = data.unripe;

                    document.getElementById("img").src = '/chili/image?t=' + Date.now();

                    const t = await fetch('/device/time');
                    const tData = await t.json();
                    document.getElementById("t_device").innerHTML = tData.last_device ?? "-";
                    document.getElementById("t_datetime").innerHTML = tData.last_datetime ?? "-";

                    const d = await fetch('/sensor/dht');
                    const dData = await d.json();
                    document.getElementById("s_temp").innerHTML = dData.temperature ?? "-";
                    document.getElementById("s_hum").innerHTML = dData.humidity ?? "-";
                    document.getElementById("s_dev").innerHTML = dData.device ?? "-";

                } catch (err) {
                    document.getElementById("status").innerHTML = "No Image";
                }
            }
            setInterval(update, 2000);
            update();
        </script>

    </body>
    </html>
    """


import os
import sqlite3
import hashlib
from collections import Counter
import csv

import numpy as np
import librosa
from scipy.ndimage import maximum_filter


# ===================== CONFIG =====================
SONGS_FOLDER = "songs"
QUERIES_FOLDER = "audio"
DB_PATH = "database/shazam.db"

N_FFT = 2048
HOP_LENGTH = 512
NEIGHBORHOOD_SIZE = 20
PEAK_THRESHOLD_DB = -30

FAN_VALUE = 10
MAX_DT = 200

TOP_N_HASHES = 2000   # speed up matching
MIN_MATCHES = 10      # minimum votes to accept match


# ===================== DATABASE =====================
def init_db():
    os.makedirs("database", exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    cur.execute("""
        CREATE TABLE IF NOT EXISTS songs (
            song_id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL UNIQUE
        )
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS fingerprints (
            hash TEXT NOT NULL,
            song_id INTEGER NOT NULL,
            time_offset INTEGER NOT NULL,
            FOREIGN KEY(song_id) REFERENCES songs(song_id)
        )
    """)

    cur.execute("CREATE INDEX IF NOT EXISTS idx_hash ON fingerprints(hash)")
    conn.commit()
    return conn


def reset_db(conn):
    cur = conn.cursor()
    cur.execute("DELETE FROM fingerprints")
    cur.execute("DELETE FROM songs")
    conn.commit()


def add_song(conn, song_name):
    cur = conn.cursor()
    cur.execute("INSERT OR IGNORE INTO songs(name) VALUES (?)", (song_name,))
    conn.commit()
    cur.execute("SELECT song_id FROM songs WHERE name=?", (song_name,))
    return cur.fetchone()[0]


def store_fingerprints(conn, song_id, fingerprints):
    cur = conn.cursor()
    cur.executemany(
        "INSERT INTO fingerprints(hash, song_id, time_offset) VALUES (?,?,?)",
        [(h, song_id, int(t)) for (h, t) in fingerprints]
    )
    conn.commit()


def lookup_hash(conn, h):
    cur = conn.cursor()
    cur.execute("SELECT song_id, time_offset FROM fingerprints WHERE hash=?", (h,))
    return cur.fetchall()


def get_song_name(conn, song_id):
    cur = conn.cursor()
    cur.execute("SELECT name FROM songs WHERE song_id=?", (song_id,))
    row = cur.fetchone()
    return row[0] if row else "UNKNOWN"


# ===================== FINGERPRINTING =====================
def compute_spectrogram_db(y):
    S = np.abs(librosa.stft(y, n_fft=N_FFT, hop_length=HOP_LENGTH))
    return librosa.amplitude_to_db(S, ref=np.max)


def detect_peaks(S_db):
    local_max = maximum_filter(S_db, size=NEIGHBORHOOD_SIZE) == S_db
    coords = np.where((local_max) & (S_db > PEAK_THRESHOLD_DB))
    peak_freqs = coords[0]
    peak_times = coords[1]

    peaks = list(zip(peak_times, peak_freqs))
    peaks.sort()
    return peaks


def make_hash(f1, f2, dt):
    raw = f"{f1}|{f2}|{dt}".encode("utf-8")
    return hashlib.sha1(raw).hexdigest()[:20]


def generate_fingerprints(peaks):
    fingerprints = []
    for i in range(len(peaks)):
        t1, f1 = peaks[i]
        for j in range(1, FAN_VALUE + 1):
            if i + j >= len(peaks):
                break
            t2, f2 = peaks[i + j]
            dt = t2 - t1
            if 0 < dt <= MAX_DT:
                fingerprints.append((make_hash(f1, f2, dt), t1))
    return fingerprints


def fingerprint_file(path):
    y, sr = librosa.load(path)
    S_db = compute_spectrogram_db(y)
    peaks = detect_peaks(S_db)
    return generate_fingerprints(peaks)


# ===================== BUILD DATABASE =====================
def build_db_from_songs():
    conn = init_db()
    reset_db(conn)

    song_files = [f for f in os.listdir(SONGS_FOLDER)
                  if f.lower().endswith((".mp3", ".wav"))]

    if not song_files:
        print("❌ No songs found in songs/ folder.")
        return None

    print(f"\n✅ Found {len(song_files)} songs. Building DB...\n")

    for f in song_files:
        song_path = os.path.join(SONGS_FOLDER, f)
        song_id = add_song(conn, f)

        fps = fingerprint_file(song_path)
        store_fingerprints(conn, song_id, fps)

        print(f"✅ Stored {len(fps)} fingerprints for {f}")

    print("\n✅ Database build complete!")
    print("DB saved at:", DB_PATH)
    return conn


# ===================== MATCHING =====================
def identify_song(conn, query_path):
    query_fps = fingerprint_file(query_path)

    # speed-up
    if TOP_N_HASHES and len(query_fps) > TOP_N_HASHES:
        query_fps = query_fps[:TOP_N_HASHES]

    votes = Counter()

    for h, t_query in query_fps:
        matches = lookup_hash(conn, h)
        for song_id, t_song in matches:
            delta = t_song - t_query
            votes[(song_id, delta)] += 1

    if not votes:
        return None, 0

    (best_song_id, _best_delta), best_votes = votes.most_common(1)[0]
    return best_song_id, best_votes


# ===================== TEST ALL QUERIES =====================
def test_all_queries(conn):
    query_files = [f for f in os.listdir(QUERIES_FOLDER)
                   if f.lower().endswith((".mp3", ".wav"))]

    if not query_files:
        print("❌ No query clips found in audio/ folder.")
        return

    print(f"\n🎤 Found {len(query_files)} query clips. Testing...\n")

    results = []
    for q in query_files:
        qpath = os.path.join(QUERIES_FOLDER, q)

        song_id, votes = identify_song(conn, qpath)
        if song_id is None:
            predicted = "NO MATCH"
            confidence = "LOW"
        else:
            predicted = get_song_name(conn, song_id)
            confidence = "GOOD" if votes >= MIN_MATCHES else "LOW"

        results.append([q, predicted, votes, confidence])
        print(f"Query: {q:20s}  →  Predicted: {predicted:20s}  Votes: {votes:5d}  Confidence: {confidence}")

    # Save results for paper
    with open("results.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Query Clip", "Predicted Song", "Votes", "Confidence"])
        writer.writerows(results)

    print("\n✅ Saved results table to results.csv (use in your report!)")


# ===================== MAIN =====================
if __name__ == "__main__":
    conn = build_db_from_songs()
    if conn:
        test_all_queries(conn)

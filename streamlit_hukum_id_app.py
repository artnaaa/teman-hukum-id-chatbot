# -*- coding: utf-8 -*-
# Chatbot Teman Hukum ID - Streamlit App
# Catatan: Aplikasi ini bersifat edukatif dan bukan pengganti nasihat hukum profesional.

import streamlit as st
from google import genai
from typing import List, Dict
import numpy as np
from pypdf import PdfReader
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import altair as alt
from database_hukum_tools import (
    hukum_init_database,
    hukum_add_message,
    hukum_get_messages,
    hukum_clear_conversation,
    hukum_upsert_pdf_chunk,
    hukum_list_conversations,
    hukum_create_conversation,
    hukum_delete_conversation,
)

# =============================
# 1) Konfigurasi Halaman
# =============================
st.set_page_config(page_title="Chatbot Teman Hukum ID", page_icon="⚖️", layout="centered")
st.title("⚖️ Chatbot Teman Hukum ID")
st.caption("Teman Konsultasi Hukum Berbasis AI yang Siap Membantu Kapan Saja, Dimana Saja")
_ = hukum_init_database()

# =============================
# 2) Sidebar: Parameter Kreatif & API Key
# =============================
with st.sidebar:
    st.subheader("Pengaturan")
    google_api_key = st.text_input("Google AI API Key", type="password")

    # Kelompokkan parameter prompt & RAG dalam satu bagian
    with st.expander("Parameter Prompt & RAG", expanded=True):
        model = st.selectbox(
            "Model",
            options=[
                "gemini-2.5-flash",
                "gemini-1.5-flash",
                "gemini-1.5-pro",
            ],
            index=0,
            help="Pilih model sesuai kebutuhan respons (flash cepat, pro lebih cermat).",
        )

        # Preferensi tepat di bawah Model
        gaya_bahasa = st.selectbox(
            "Gaya Bahasa",
            options=["Santai", "Asik", "Keren", "Professional", "Formal"],
            index=0,
            help="Nada respons asisten.",
        )

        domain_fokus = st.selectbox(
            "Fokus Domain",
            options=[
                "Umum",
                "Ketenagakerjaan",
                "Perlindungan Konsumen",
                "Lalu Lintas",
                "Perdata (Kontrak)",
                "Pidana (Ringan)",
            ],
            index=0,
            help="Fokus area pembahasan agar respons lebih relevan.",
        )

        # Chart Biru (Prompting) tepat DI BAWAH Model
        _cur_temperature = float(st.session_state.get("temperature", 0.3))
        _cur_top_p = float(st.session_state.get("top_p", 0.9))
        _cur_top_k = float(st.session_state.get("decoding_top_k", 40))
        _cur_max_tokens = float(st.session_state.get("max_output_tokens", 1024))
        st.caption("Visual Prompting")
        _prompt_items = [
            ("Temperature", _cur_temperature),
            ("Top-p", _cur_top_p),
            ("Top-K", _cur_top_k / 50.0),
            ("Max Tokens", _cur_max_tokens / 4096.0),
        ]
        _chart_prompting = (
            alt.Chart(alt.Data(values=[{"Parameter": k, "Nilai": v} for k, v in _prompt_items]))
            .mark_bar(color="#6366F1")
            .encode(x=alt.X("Parameter:N", sort=None), y=alt.Y("Nilai:Q", scale=alt.Scale(domain=[0, 1])))
            .properties(height=260)
        )
        st.altair_chart(_chart_prompting, use_container_width=True)

        temperature = st.slider(
            "Temperature",
            min_value=0.0,
            max_value=1.0,
            value=0.3,
            step=0.1,
            help="Lebih rendah = lebih deterministik dan aman untuk domain hukum.",
            key="temperature",
        )

        # Kontrol decoding tambahan
        top_p = st.slider(
            "Top-p",
            min_value=0.0,
            max_value=1.0,
            value=0.9,
            step=0.05,
            help="Sampling nucleus: 1.0 = tanpa pembatasan.",
            key="top_p",
        )
        decoding_top_k = st.slider(
            "Decoding Top-K",
            min_value=1,
            max_value=50,
            value=40,
            step=1,
            help="Pertimbangkan K token teratas saat sampling.",
            key="decoding_top_k",
        )
        max_output_tokens = st.slider(
            "Max Output Tokens",
            min_value=256,
            max_value=4096,
            value=1024,
            step=64,
            help="Batas panjang jawaban model.",
            key="max_output_tokens",
        )
        
        

        aktifkan_rag_pdf = st.checkbox("Aktifkan RAG dari PDF", value=False, help="Gunakan konten PDF sebagai sumber konteks tambahan.")

        # Chart Hijau (RAG) tepat DI BAWAH toggle Aktifkan RAG
        _cur_top_k_docs = float(st.session_state.get("rag_top_k", 4))
        _cur_threshold = float(st.session_state.get("score_threshold", 0.2))
        _cur_ctx_chars = float(st.session_state.get("max_context_chars", 4000))
        st.caption("Visual RAG")
        _rag_items = [
            ("Top-K Docs", _cur_top_k_docs / 8.0),
            ("Threshold", _cur_threshold),
            ("Ctx Chars", _cur_ctx_chars / 20000.0),
        ]
        _chart_rag = (
            alt.Chart(alt.Data(values=[{"Parameter": k, "Nilai": v} for k, v in _rag_items]))
            .mark_bar(color="#10B981")
            .encode(x=alt.X("Parameter:N", sort=None), y=alt.Y("Nilai:Q", scale=alt.Scale(domain=[0, 1])))
            .properties(height=260)
        )
        st.altair_chart(_chart_rag, use_container_width=True)
        top_k = st.slider("Top-K Dokumen Relevan", 1, 8, 4, help="Jumlah potongan konteks yang disuntikkan ke prompt.", key="rag_top_k")
        score_threshold = st.slider(
            "Ambang Skor Kesesuaian (threshold)",
            min_value=0.0,
            max_value=1.0,
            value=0.2,
            step=0.05,
            help="Abaikan potongan dengan kesesuaian di bawah ambang ini.",
            key="score_threshold",
        )
        max_context_chars = st.number_input(
            "Batas Panjang Konteks (karakter)",
            min_value=500,
            max_value=20000,
            value=4000,
            step=250,
            help="Memotong konteks RAG agar prompt tidak terlalu panjang.",
            key="max_context_chars",
        )
        uploaded_pdfs = st.file_uploader("Unggah PDF (bisa lebih dari satu)", type=["pdf"], accept_multiple_files=True, help="PDF akan diproses lokal dan tidak diunggah ke server lain.")
        bangun_indeks = st.button("Bangun/Perbarui Indeks PDF", use_container_width=True)
        hapus_indeks = st.button("Bersihkan Indeks PDF", use_container_width=True, help="Hapus indeks RAG dari memori aplikasi.")
        # Memori profil dihapus; selalu nonaktif
        memori_diaktifkan = False


    # Reset percakapan dipindah ke bagian 'Kelola Percakapan'
    reset_click = False

    # Manajemen Percakapan (seperti ChatGPT)
    with st.expander("Kelola Percakapan", expanded=True):
        try:
            convs = hukum_list_conversations(limit=100)
        except Exception:
            convs = []

        # Pastikan ada percakapan terpilih di session
        if "conv_id" not in st.session_state:
            if convs:
                st.session_state.conv_id = convs[0]["id"]
            else:
                try:
                    st.session_state.conv_id = hukum_create_conversation("Percakapan")
                    convs = hukum_list_conversations(limit=100)
                except Exception:
                    st.session_state.conv_id = None

        # Build opsi untuk selectbox
        conv_options = [f"#{c['id']} - {c.get('title') or 'Percakapan'}" for c in convs]
        conv_map = {f"#{c['id']} - {c.get('title') or 'Percakapan'}": c["id"] for c in convs}
        current_label = None
        for label, cid in conv_map.items():
            if cid == st.session_state.get("conv_id"):
                current_label = label
                break
        selected_label = st.selectbox("Daftar Percakapan", options=conv_options, index=conv_options.index(current_label) if current_label in conv_options else 0)
        st.session_state.conv_id = conv_map.get(selected_label)

        # CRUD Percakapan (buat percakapan saat Enter di input)
        def _on_new_conv_change():
            title = (st.session_state.get("new_conv_title") or "").strip()
            if title:
                try:
                    cid = hukum_create_conversation(title)
                    st.session_state.conv_id = cid
                    st.session_state["new_conv_title"] = ""
                    # Tidak memanggil st.rerun() di dalam callback; perubahan state akan memicu rerun otomatis
                except Exception as e:
                    st.warning(f"Gagal membuat percakapan: {e}")

        st.text_input(
            "Percakapan Baru",
            value="",
            key="new_conv_title",
            on_change=_on_new_conv_change,
            placeholder="Ketik judul lalu tekan enter",
        )
        

        # Reset percakapan yang dipilih (rapat dengan input judul baru)
        if st.button("Reset Percakapan", help="Kosongkan histori percakapan yang sedang dipilih", use_container_width=True):
            reset_click = True

        # Hapus percakapan (diletakkan di bawah reset dan full width)
        if st.button("Hapus Percakapan", use_container_width=True):
            try:
                cid = st.session_state.get("conv_id")
                if cid:
                    hukum_delete_conversation(cid)
                    # pilih percakapan lain atau buat baru
                    remaining = hukum_list_conversations(limit=100)
                    if remaining:
                        st.session_state.conv_id = remaining[0]["id"]
                    else:
                        st.session_state.conv_id = hukum_create_conversation("Percakapan")
                    st.rerun()
            except Exception as e:
                st.warning(f"Gagal menghapus percakapan: {e}")


# =============================
# 3) Validasi API Key & Inisialisasi Client
# =============================
if not google_api_key:
    st.info("Tambahkan Google AI API Key di sidebar untuk mulai menggunakan chatbot.", icon="🗝️")
    st.stop()

# Inisialisasi client sekali saat kunci berubah
if ("genai_client" not in st.session_state) or (getattr(st.session_state, "_last_key", None) != google_api_key):
    try:
        st.session_state.genai_client = genai.Client(api_key=google_api_key)
        st.session_state._last_key = google_api_key
        # Bersihkan sesi lama ketika kunci berubah
        st.session_state.pop("chat", None)
        st.session_state.pop("messages", None)
        st.session_state.pop("rag_index", None)
    except Exception as e:
        st.error(f"API Key tidak valid: {e}")
        st.stop()

# Hapus indeks RAG dari memori jika diminta
if 'hapus_indeks' in locals() and hapus_indeks:
    st.session_state.pop("rag_index", None)
    st.toast("Indeks PDF dibersihkan dari memori.")

# =============================
# 3a) Visualisasi Parameter (Chart Design)
# =============================
# 4) Inisialisasi Chat & State
# =============================
if "chat" not in st.session_state:
    st.session_state.chat = st.session_state.genai_client.chats.create(model=model)

if "messages" not in st.session_state:
    # Inisialisasi dari DB untuk percakapan terpilih
    try:
        db_msgs = hukum_get_messages(limit=100, conversation_id=st.session_state.get("conv_id"))
        st.session_state.messages = [
            {"role": m["role"], "content": m["content"]} for m in db_msgs
        ]
    except Exception:
        st.session_state.messages = []  # List[Dict[str, str]] dengan key: role, content
else:
    # Jika user berganti percakapan, muat ulang historinya
    if st.session_state.get("_last_loaded_conv") != st.session_state.get("conv_id"):
        try:
            db_msgs = hukum_get_messages(limit=100, conversation_id=st.session_state.get("conv_id"))
            st.session_state.messages = [
                {"role": m["role"], "content": m["content"]} for m in db_msgs
            ]
        except Exception:
            st.session_state.messages = []
        st.session_state._last_loaded_conv = st.session_state.get("conv_id")

if reset_click:
    try:
        hukum_clear_conversation(st.session_state.get("conv_id"))
    except Exception:
        pass
    st.session_state.pop("chat", None)
    st.session_state.pop("messages", None)
    st.rerun()

    

# =============================
# 5) Utilitas Prompt
# =============================
DISKLAIMER = (
    "Catatan: Informasi berikut bersifat edukatif, bukan nasihat hukum. "
    "Untuk keputusan penting, konsultasikan dengan advokat berlisensi."
)

RUJUKAN_UMUM = {
    "Ketenagakerjaan": [
        "UU No. 13 Tahun 2003 tentang Ketenagakerjaan (sebagaimana diubah UU Cipta Kerja)",
        "PP terkait PKWT/PKWTT, PHK, Pesangon",
    ],
    "Perlindungan Konsumen": [
        "UU No. 8 Tahun 1999 tentang Perlindungan Konsumen",
        "Peraturan pelaksana BPKN/BSN terkait SNI",
    ],
    "Lalu Lintas": [
        "UU No. 22 Tahun 2009 tentang Lalu Lintas dan Angkutan Jalan",
        "PP/Perkapolri terkait tilang dan kecelakaan",
    ],
    "Perdata (Kontrak)": [
        "KUH Perdata (BW) Buku III Perikatan",
        "Yurisprudensi terkait wanprestasi dan PMH",
    ],
    "Pidana (Ringan)": [
        "KUHP (terbaru)",
        "KUHAP terkait prosedur acara pidana",
    ],
}


def bangun_preamble(gaya: str, domain: str, memori: str) -> str:
    rujukan = RUJUKAN_UMUM.get(domain, [])
    rujukan_txt = "\n".join(f"- {r}" for r in rujukan) if rujukan else "- (sesuaikan jika relevan)"
    persona = f"Anda adalah 'Teman Hukum ID', asisten hukum edukatif berbahasa Indonesia dengan gaya {gaya}."
    fokus = f"Fokus bahasan: {domain}. Jawab ringkas, akurat, dan mudah dipahami orang awam."
    mem = f"Konteks Pengguna: {memori}" if memori else "Konteks Pengguna: (tidak tersedia)"
    instruksi = (
        "Pedoman: 1) Jelaskan konsep dengan contoh praktis. 2) Hindari bahasa terlalu teknis tanpa definisi. "
        "3) Jika butuh data spesifik kasus, ajukan pertanyaan klarifikasi. 4) Cantumkan rujukan regulasi jika relevan. "
        "5) Jangan memberikan nasihat hukum pasti; tekankan edukasi dan opsi langkah umum."
    )
    keluaran = (
        "Format keluaran: \n- Ringkasan poin\n- Penjelasan singkat\n- Rujukan regulasi (jika ada)\n- Saran tindak lanjut"
    )
    return (
        f"{persona}\n{fokus}\n{mem}\n{instruksi}\nRujukan umum yang sering relevan:\n{rujukan_txt}\n{keluaran}"
    )


def rangkai_prompt(user_input: str, gaya: str, domain: str, memori_diaktifkan: bool) -> str:
    memori_text = st.session_state.get("profil_pengguna", "") if memori_diaktifkan else ""
    preamble = bangun_preamble(gaya, domain, memori_text)
    return (
        f"{preamble}\n\nPertanyaan/cerita pengguna:\n" + user_input + "\n\n"
        "Jawablah dalam Bahasa Indonesia sesuai pedoman di atas."
    )


# =============================
# 5a) Fitur RAG PDF: ekstraksi, embedding, retrieval
# =============================
def _extract_pdfs(files) -> List[Dict[str, str]]:
    hasil = []
    for f in files:
        try:
            reader = PdfReader(f)
            for i, page in enumerate(reader.pages, start=1):
                text = page.extract_text() or ""
                if text.strip():
                    hasil.append({"filename": f.name, "page": i, "text": text})
        except Exception:
            continue
    return hasil


def _chunk_text(t: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
    t = " ".join(t.split())
    chunks = []
    start = 0
    while start < len(t):
        end = min(len(t), start + chunk_size)
        chunks.append(t[start:end])
        start = end - overlap
        if start < 0:
            start = 0
        if start >= len(t):
            break
        if end == len(t):
            break
    return chunks


def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


def _ensure_embeddings(api_key: str):
    if "embeddings" not in st.session_state:
        try:
            st.session_state.embeddings = GoogleGenerativeAIEmbeddings(
                model="models/text-embedding-004",
                google_api_key=api_key,
            )
        except Exception:
            st.session_state.embeddings = GoogleGenerativeAIEmbeddings(
                model="models/embedding-001",
                google_api_key=api_key,
            )


def build_pdf_index(files, api_key: str):
    try:
        _ensure_embeddings(api_key)
    except Exception as e:
        st.error("Gagal inisialisasi embeddings. Pastikan API key Google AI valid dan Generative Language API aktif.")
        return None

    docs = _extract_pdfs(files)
    chunks = []
    metas = []
    for d in docs:
        for ch in _chunk_text(d["text"]):
            chunks.append(ch)
            metas.append({"filename": d["filename"], "page": d["page"]})
    if not chunks:
        st.warning("Tidak ada teks yang dapat diambil dari PDF yang diunggah.")
        return None

    try:
        vectors = st.session_state.embeddings.embed_documents(chunks)
    except Exception as e:
        st.error("Gagal membuat embedding dokumen. Periksa kembali API key Anda (valid/aktif) atau pembatasan key.")
        return None

    st.session_state.rag_index = {"chunks": chunks, "metas": metas, "vectors": np.array(vectors, dtype=float)}
    return st.session_state.rag_index


def retrieve_context(query: str, top_k: int, score_threshold: float, max_context_chars: int) -> Dict[str, any]:
    if not st.session_state.get("rag_index"):
        return {"context": "", "sources": [], "scores": []}
    _ensure_embeddings(google_api_key)
    q_vec = np.array(st.session_state.embeddings.embed_query(query), dtype=float)
    vecs = st.session_state.rag_index["vectors"]
    sims = vecs @ q_vec / (np.linalg.norm(vecs, axis=1) * (np.linalg.norm(q_vec) + 1e-10) + 1e-10)
    order = np.argsort(-sims)
    # Ambil kandidat awal lebih banyak lalu filter threshold
    top_idxs = order[: max(top_k * 3, top_k)]
    filtered = [(i, float(sims[i])) for i in top_idxs if float(sims[i]) >= float(score_threshold)]
    filtered = filtered[:top_k]
    ctx_parts = []
    sources = []
    scores = []
    for i, sc in filtered:
        ctx_parts.append(st.session_state.rag_index["chunks"][i])
        sources.append(st.session_state.rag_index["metas"][i])
        scores.append(sc)
    context = "\n\n".join(ctx_parts)
    if len(context) > int(max_context_chars):
        context = context[: int(max_context_chars)]
    return {"context": context, "sources": sources, "scores": scores}


def buat_rekomendasi_tindak_lanjut(jawaban_asisten: str, prompt_user: str, domain_fokus: str) -> List[str]:
    """Meminta model membuat 3 rekomendasi pertanyaan tindak lanjut singkat."""
    try:
        prompt = (
            "Berdasarkan jawaban berikut, buatkan 3 pertanyaan tindak lanjut ringkas, numerik, relevan, "
            "tanpa penjelasan tambahan, dalam Bahasa Indonesia.\n\nJawaban:\n" + jawaban_asisten
        )
        resp = st.session_state.genai_client.models.generate_content(
            model=model,
            contents=prompt,
            config={
                "temperature": max(0.2, min(temperature + 0.1, 0.8)),
                "top_p": float(top_p),
                "top_k": int(decoding_top_k),
                "max_output_tokens": int(max_output_tokens/2),
            },
        )
        text = getattr(resp, "text", str(resp))
        # Ekstrak 3 baris pertama yang bermakna
        lines = [l.strip(" -•\t") for l in text.splitlines() if l.strip()]
        hasil: List[str] = []
        for l in lines:
            if len(hasil) >= 3:
                break
            # buang penomoran jika ada
            l = l.split(". ", 1)[-1] if ". " in l[:4] else l
            hasil.append(l)
        # fallback bila kurang dari 3
        while len(hasil) < 3:
            hasil.append("Apakah ada dokumen/kontrak terkait yang bisa Anda bagi detailnya?")
        return hasil[:3]
    except Exception:
        return [
            "Apa kronologi singkat dan tanggal kejadian utama?",
            "Apakah ada dokumen/kontrak atau surat resmi terkait?",
            "Apa tujuan Anda (klarifikasi, negosiasi, laporan, gugatan)?",
        ]

# =============================
# 6) Tampilkan Histori
# =============================
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# =============================
# 7) Input Pengguna & Alur Chat
# =============================
prompt_user = st.chat_input("Tulis pertanyaan atau ceritakan masalah Anda...")

if prompt_user:
    # Simpan dan tampilkan pesan user
    st.session_state.messages.append({"role": "user", "content": prompt_user})
    try:
        hukum_add_message("user", prompt_user, st.session_state.get("conv_id"))
    except Exception:
        pass
    with st.chat_message("user"):
        st.markdown(prompt_user)

    # Susun prompt dengan persona + memori
    composed_prompt = rangkai_prompt(prompt_user, gaya_bahasa, domain_fokus, memori_diaktifkan)

    # Jika RAG aktif, ambil konteks dari PDF dan suntikkan
    retrieved = {"context": "", "sources": [], "scores": []}
    if 'aktifkan_rag_pdf' in locals() and aktifkan_rag_pdf:
        if st.session_state.get("rag_index"):
            retrieved = retrieve_context(prompt_user, top_k, float(score_threshold), int(max_context_chars))
            if retrieved["context"]:
                composed_prompt = (
                    composed_prompt
                    + "\n\nKonteks dari dokumen yang diunggah (gunakan hanya jika relevan):\n"
                    + retrieved["context"]
                )

    try:
        # Kirim ke chat (mempertahankan context di server sisi Google dalam sesi chat)
        # Untuk kontrol yang lebih eksplisit, kita juga menambahkan temperature via models.generate_content untuk fitur rekomendasi.
        resp = st.session_state.chat.send_message(composed_prompt)
        if hasattr(resp, "text"):
            answer = resp.text
        else:
            answer = str(resp)
    except Exception as e:
        answer = f"Terjadi kesalahan saat memproses: {e}"

    # Tampilkan jawaban asisten
    with st.chat_message("assistant"):
        st.markdown(answer)
        st.caption(DISKLAIMER)

        # Rekomendasi tindak lanjut (konteks-aware)
        rekomendasi = buat_rekomendasi_tindak_lanjut(answer, prompt_user, domain_fokus)
        with st.expander("Rekomendasi pertanyaan tindak lanjut"):
            for i, r in enumerate(rekomendasi, start=1):
                st.markdown(f"{i}. {r}")

        # Tampilkan sumber PDF yang dipakai
        if retrieved.get("sources"):
            with st.expander("Sumber konteks PDF yang digunakan"):
                for s in retrieved["sources"]:
                    st.markdown(f"- {s['filename']} (hal. {s['page']})")

    # Simpan jawaban ke histori
    st.session_state.messages.append({"role": "assistant", "content": answer})
    try:
        hukum_add_message("assistant", answer, st.session_state.get("conv_id"))
    except Exception:
        pass

# =============================
# 8) Footer kecil
# =============================
st.markdown("---")
st.markdown(
    "Privasi: Memori dan profil hanya tersimpan lokal di sesi Streamlit Anda. "
    "Jangan membagikan data sensitif."
)

# Bangun/perbarui indeks PDF jika diminta
if 'bangun_indeks' in locals() and bangun_indeks and uploaded_pdfs:
    with st.status("Memproses PDF dan membangun indeks..."):
        idx = build_pdf_index(uploaded_pdfs, google_api_key)
        if idx:
            # Simpan potongan ke database untuk jejak
            try:
                for meta, chunk in zip(idx["metas"], idx["chunks"]):
                    hukum_upsert_pdf_chunk(meta.get("filename", "unknown.pdf"), int(meta.get("page", 0)), chunk)
            except Exception:
                pass
            st.success(f"Indeks PDF siap. Potongan: {len(idx['chunks'])}")
        else:
            st.warning("Tidak ada teks dapat diambil dari PDF yang diunggah.")

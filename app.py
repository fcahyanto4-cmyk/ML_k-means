from flask import Flask, render_template, request, redirect, url_for, flash, session
from flask_sqlalchemy import SQLAlchemy
import pandas as pd
import os
from datetime import datetime
import numpy as np
import re
from util.ml_utils import KMeans, silhouette_score
from functools import wraps
from sqlalchemy import text
import calendar

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql+mysqlconnector://root:@localhost:3306/uts'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['SECRET_KEY'] = 'secret'
app.config['UPLOAD_FOLDER'] = 'uploads'
 
db = SQLAlchemy(app)
 
# ======================
# MODEL
# ======================
class PajakData(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    nama_wajib_pajak = db.Column(db.String(255))
    npwp = db.Column(db.String(255))
    masa_pajak = db.Column(db.String(50))
    bulan_pajak = db.Column(db.Integer)
    jenis_pajak = db.Column(db.String(255))
    tanggal_bayar = db.Column(db.Date)
    jumlah_bayar = db.Column(db.Float)
 
    __tablename__ = 'pajak_data'
 
 
# ======================
# LOGIN
# ======================
HARDCODED_USERNAME = "admin"
HARDCODED_PASSWORD = "password"
 
def login_required(f):
    @wraps(f)
    def wrap(*args, **kwargs):
        if session.get("logged_in"):
            return f(*args, **kwargs)
        return redirect(url_for("login"))
    return wrap
 
 
@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        if request.form["username"] == HARDCODED_USERNAME and request.form["password"] == HARDCODED_PASSWORD:
            session["logged_in"] = True
            session["username"] = HARDCODED_USERNAME
            return redirect(url_for("index"))
        flash("Login gagal")
    return render_template("login.html")
 
 
@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))
 # sesuaikan dengan nama filenya
  
@app.route("/")
@login_required
def index():
    res = _hitung_kmeans()
    if res is None:
        return render_template(
            "index.html",
            has_data=False,
            data=[],
            inertia=[],
            k_range=[],
            silhouette=0.0,
            insight=[],
            k_optimal=0,
            centroids=[],
            purity=None
        )
    return render_template(
        "index.html",
        has_data=True,
        data=res['data'],
        inertia=res['inertia'],
        k_range=res['k_range'],
        silhouette=res['silhouette'],
        insight=res['insight'],
        k_optimal=res['k_optimal'],
        centroids=res['centroids'],
        purity=res['purity']
    )
 
 
# ======================
# HELPER FUNCTIONS
# ======================
def clean_npwp(npwp):
    return str(npwp).replace('.', '').replace('-', '').strip()
 
def clean_jenis(jenis):
    return str(jenis).lower().replace(" ", "").strip()
 
def clean_bulan(bulan, default=1):
    if pd.isna(bulan) or bulan is None:
        return default
    bulan_map = {
        "januari": 1, "februari": 2, "maret": 3, "april": 4,
        "mei": 5, "juni": 6, "juli": 7, "agustus": 8,
        "september": 9, "oktober": 10, "november": 11, "desember": 12,
        "jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
        "jul": 7, "aug": 8, "sep": 9, "okt": 10, "oct": 10, "nov": 11, "des": 12, "dec": 12
    }
    if isinstance(bulan, (int, np.integer)):
        return int(bulan)
    try:
        val = int(float(bulan))
        if 1 <= val <= 12:
            return val
    except (ValueError, TypeError):
        pass
    return bulan_map.get(str(bulan).strip().lower(), default)
 
def hitung_jatuh_tempo(jenis, bulan, tahun):
    """
    Hitung tanggal jatuh tempo penyetoran berdasarkan aturan DJP
    yang berlaku sejak PMK-9/PMK.03/2022 (berlaku mulai 1 April 2022):

    - PPN / PPnBM      : Tanggal 15 bulan berikutnya  (PMK-9/2022 Pasal 4)
    - PPh 25 (angsuran): Tanggal 15 bulan berikutnya  (PMK-242/2014)
    - PPh 21, 23,
      PPh Badan, dll   : Tanggal 10 bulan berikutnya  (PMK-242/2014)

    Catatan: Aturan lama PPN = akhir bulan berikutnya SUDAH TIDAK BERLAKU
    sejak PMK-9/PMK.03/2022 efektif 1 April 2022.
    """
    bulan_next = bulan + 1 if bulan < 12 else 1
    tahun_next = tahun if bulan < 12 else tahun + 1

    if 'ppn' in jenis or 'ppnbm' in jenis:
        # PMK-9/2022: PPN disetor paling lambat tanggal 15 bulan berikutnya
        return datetime(tahun_next, bulan_next, 15).date()
    elif '25' in jenis:
        # PPh 25 angsuran: tgl 15 bulan berikutnya
        return datetime(tahun_next, bulan_next, 15).date()
    else:
        # PPh 21, 23, Badan, dll: tgl 10 bulan berikutnya
        return datetime(tahun_next, bulan_next, 10).date()


# ======================
# DETEKSI K OPTIMAL DARI ELBOW METHOD (OTOMATIS)
# ======================
def _deteksi_k_elbow(K_range, inertia):
    """
    Deteksi titik elbow (siku) secara otomatis dari kurva inertia
    menggunakan metode jarak maksimum terhadap garis lurus
    yang menghubungkan titik pertama dan titik terakhir kurva.

    Semakin besar jarak tegak lurus suatu titik terhadap garis
    (titik_awal -> titik_akhir), semakin besar kemungkinan titik
    tersebut adalah "siku" dari kurva elbow.
    """
    if not K_range or not inertia or len(K_range) != len(inertia):
        return None
    if len(K_range) == 1:
        return K_range[0]

    x = np.array(K_range, dtype=float)
    y = np.array(inertia, dtype=float)

    # Normalisasi supaya skala sumbu x dan y sebanding
    x_norm = (x - x.min()) / (x.max() - x.min() + 1e-9)
    y_norm = (y - y.min()) / (y.max() - y.min() + 1e-9)

    p1 = np.array([x_norm[0], y_norm[0]])
    p2 = np.array([x_norm[-1], y_norm[-1]])
    line_vec = p2 - p1
    line_len = np.linalg.norm(line_vec)
    line_unit = line_vec / (line_len + 1e-9)

    jarak = []
    for i in range(len(x_norm)):
        p = np.array([x_norm[i], y_norm[i]])
        proj_len = np.dot(p - p1, line_unit)
        proj_point = p1 + proj_len * line_unit
        jarak.append(np.linalg.norm(p - proj_point))

    idx_elbow = int(np.argmax(jarak))
    return int(x[idx_elbow])


# ======================
# KELAS AKTUAL (GROUND TRUTH) UNTUK PERHITUNGAN PURITY
# Aturan bisnis independen dari hasil clustering, dipakai sebagai
# pembanding "kelas sebenarnya" saat menghitung Purity.
# ======================
def klasifikasi_aktual(tepat, telat, belum):
    """
    Klasifikasi kelas aktual (ground truth) per wajib pajak berdasarkan
    proporsi tepat waktu / belum bayar, TANPA melihat hasil clustering.
    - Patuh        : proporsi tepat waktu >= 60%
    - Tidak Patuh  : proporsi belum bayar >= 50%
    - Kurang Patuh : selain dua kondisi di atas
    """
    if tepat >= 0.60:
        return 'Patuh'
    elif belum >= 0.50:
        return 'Tidak Patuh'
    else:
        return 'Kurang Patuh'


def _hitung_purity(hasil, k_optimal):
    """
    Hitung nilai Purity dari hasil clustering K-Means.

    Purity = (1/N) * sum_k max_j |w_k ∩ c_j|

    Langkah:
    1. Untuk tiap cluster, hitung jumlah anggota per kelas aktual.
    2. Tentukan kelas mayoritas pada cluster tersebut.
    3. Jumlahkan anggota kelas mayoritas dari seluruh cluster.
    4. Bagi dengan total data (N).

    Return: dict berisi detail langkah perhitungan + nilai akhir purity.
    """
    N = len(hasil)
    if N == 0:
        return {
            'detail_cluster': [],
            'total_majority': 0,
            'N': 0,
            'nilai': 0.0,
            'interpretasi': 'Tidak ada data untuk dihitung.'
        }

    detail_cluster = []
    total_majority = 0

    for cid in range(k_optimal):
        anggota = [h for h in hasil if h['cluster'] == cid]
        if not anggota:
            continue

        # 1) Jumlah anggota tiap kelas aktual pada cluster ini
        distribusi_kelas = {}
        for a in anggota:
            kelas = a['kelas_aktual']
            distribusi_kelas[kelas] = distribusi_kelas.get(kelas, 0) + 1

        # 2) Kelas mayoritas pada cluster ini
        kelas_mayoritas = max(distribusi_kelas, key=distribusi_kelas.get)
        jumlah_mayoritas = distribusi_kelas[kelas_mayoritas]
        total_majority += jumlah_mayoritas

        total_cluster = len(anggota)
        purity_k = round(jumlah_mayoritas / total_cluster, 4) if total_cluster > 0 else 0.0
        persen_mayoritas = round((jumlah_mayoritas / total_cluster) * 100, 2) if total_cluster > 0 else 0.0
        jumlah_non_mayoritas = total_cluster - jumlah_mayoritas
        persen_non_mayoritas = round((jumlah_non_mayoritas / total_cluster) * 100, 2) if total_cluster > 0 else 0.0

        detail_cluster.append({
            'cluster': cid,
            'jumlah_anggota': total_cluster,
            'distribusi_kelas': distribusi_kelas,
            'kelas_mayoritas': kelas_mayoritas,
            'jumlah_mayoritas': jumlah_mayoritas,
            'persen_mayoritas': persen_mayoritas,
            'jumlah_non_mayoritas': jumlah_non_mayoritas,
            'persen_non_mayoritas': persen_non_mayoritas,
            'purity_k': purity_k
        })

    # 3) & 4) Total anggota kelas mayoritas dibagi total data
    nilai_purity = total_majority / N
    nilai_purity_round = round(nilai_purity, 4)

    interpretasi = (
        f"Nilai Purity sebesar {nilai_purity_round} menunjukkan bahwa "
        f"{nilai_purity_round * 100:.2f}% data berada pada cluster yang "
        f"sesuai dengan kelas aktualnya."
    )

    return {
        'detail_cluster': detail_cluster,
        'total_majority': total_majority,
        'N': N,
        'nilai': nilai_purity_round,
        'interpretasi': interpretasi
    }
 
 
# ======================
# INPUT DATA
# ======================
@app.route("/data_input", methods=["GET", "POST"])
@login_required
def data_input():
    if request.method == "POST":
        file = request.files.get("file")
 
        if file and file.filename:
            path = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
            file.save(path)
 
            df = pd.read_excel(path)
 
            # Normalize column names
            df.columns = [str(c).strip().lower() for c in df.columns]
 
            mapping = {
                "nama": "nama_wajib_pajak",
                "npwp": "npwp",
                "masa": "masa_pajak",
                "bulan": "bulan_pajak",
                "jenis": "jenis_pajak",
                "tanggal": "tanggal_bayar",
                "jumlah": "jumlah_bayar"
            }
 
            new_cols = []
            for c in df.columns:
                found = False
                for k, v in mapping.items():
                    if k in c:
                        new_cols.append(v)
                        found = True
                        break
                if not found:
                    new_cols.append(c)
 
            df.columns = new_cols
 
            required = [
                "nama_wajib_pajak", "npwp", "masa_pajak",
                "bulan_pajak", "jenis_pajak",
                "tanggal_bayar", "jumlah_bayar"
            ]
 
            if not all(col in df.columns for col in required):
                flash("Format Excel salah", "danger")
                return redirect(url_for("data_input"))
 
            success_count = 0
            for _, r in df.iterrows():
                try:
                    tgl = pd.to_datetime(r["tanggal_bayar"], dayfirst=True).date()
                    bln = clean_bulan(r.get("bulan_pajak"), default=tgl.month if tgl else 1)
                    db.session.add(PajakData(
                        nama_wajib_pajak=str(r["nama_wajib_pajak"]),
                        npwp=str(r["npwp"]),
                        masa_pajak=str(r["masa_pajak"]),
                        bulan_pajak=bln,
                        jenis_pajak=str(r["jenis_pajak"]),
                        tanggal_bayar=tgl,
                        jumlah_bayar=float(r["jumlah_bayar"])
                    ))
                    success_count += 1
                except Exception as e:
                    print(f"Baris gagal: {e}")
                    continue
 
            db.session.commit()
            flash(f"Upload sukses: {success_count} data berhasil diimpor", "success")
            return redirect(url_for("preprocessing"))
 
    return render_template("data_input.html")
 
@app.route("/delete_all_pajak", methods=["POST"])
@login_required
def delete_all_pajak():
    try:
        PajakData.query.delete()
        db.session.commit()
        flash('Seluruh data pembayaran berhasil dihapus.', 'success')
    except Exception as e:
        db.session.rollback()
        flash(f'Gagal menghapus seluruh data: {str(e)}', 'danger')
        
    return redirect(url_for('data_input'))
 

 
# ======================
# PREPROCESSING
# ======================
@app.route('/preprocessing')
@login_required
def preprocessing():
    all_data = PajakData.query.all()
    unique_entries = {}
    removed_count = 0
 
    for d in all_data:
        try:
            bulan = clean_bulan(d.bulan_pajak)
            if bulan == 0:
                continue
 
            tahun = int(d.masa_pajak)
            jenis = clean_jenis(d.jenis_pajak)
 
            key = (
                clean_npwp(d.npwp),
                jenis,
                bulan,
                tahun
            )
 
            tanggal_bayar = d.tanggal_bayar
            jatuh_tempo = hitung_jatuh_tempo(jenis, bulan, tahun)
            selisih = abs((tanggal_bayar - jatuh_tempo).days)
 
            if key not in unique_entries:
                unique_entries[key] = (d, selisih)
            else:
                # Simpan yang paling dekat jatuh tempo (lebih valid)
                if selisih < unique_entries[key][1]:
                    db.session.delete(unique_entries[key][0])
                    unique_entries[key] = (d, selisih)
                else:
                    db.session.delete(d)
                    removed_count += 1
 
        except Exception as e:
            flash(f"Error preprocessing baris: {e}", 'danger')
            continue
 
    db.session.commit()
 
    final_data = [v[0] for v in unique_entries.values()]
    flash(
        f'Total data awal: {len(all_data)} | '
        f'Data unik: {len(final_data)} | '
        f'Duplikat dibuang: {removed_count}',
        'info'
    )
 
    return render_template('preprocessing.html', data=final_data)
 
 
@app.route('/delete_pajak/<int:id>', methods=['POST'])
@login_required
def delete_pajak(id):
    try:
        pajak = PajakData.query.get(id)
        if pajak:
            db.session.delete(pajak)
            db.session.commit()
            flash('Data pembayaran berhasil dihapus.', 'success')
        else:
            flash('Data tidak ditemukan.', 'danger')
    except Exception as e:
        db.session.rollback()
        flash(f'Gagal menghapus data: {str(e)}', 'danger')
        
    return redirect(url_for('preprocessing'))


# ======================
# HELPER: BANGUN DICT PEMBAYARAN
# Dipakai oleh kewajiban, agregasi, dan kmeans
# Key: (npwp_bersih, jenis_bersih, bulan_int, tahun_int)
# Value: tanggal_bayar (date)
# ======================
def build_pembayaran_dict(data_bayar):
    pembayaran = {}
    for d in data_bayar:
        try:
            bulan = clean_bulan(d.bulan_pajak)
            if bulan == 0:
                continue
            key = (
                clean_npwp(d.npwp),
                clean_jenis(d.jenis_pajak),
                bulan,
                int(d.masa_pajak)
            )
            # Jika duplikat, pakai yang paling awal (sudah dipreprocess, tapi aman)
            if key not in pembayaran:
                pembayaran[key] = d.tanggal_bayar
        except Exception as e:
            print(f"Skip pembayaran: {e}")
            continue
    return pembayaran
 
 
# ======================
# GENERATE KEWAJIBAN
# ======================
@app.route('/generate-kewajiban')
@login_required
def generate_kewajiban():
    search = request.args.get("search", "").strip()

    data_bayar = PajakData.query.all()
    pembayaran = build_pembayaran_dict(data_bayar)

    # Peta nama WP dari data pembayaran
    nama_map = {}
    for d in data_bayar:
        k = clean_npwp(d.npwp)
        if k not in nama_map:
            nama_map[k] = d.nama_wajib_pajak

    nama_bulan = {
        1:'Januari', 2:'Februari', 3:'Maret',    4:'April',
        5:'Mei',     6:'Juni',     7:'Juli',      8:'Agustus',
        9:'September',10:'Oktober',11:'November',12:'Desember'
    }

    # Generate kewajiban OTOMATIS dari kombinasi unik (npwp, jenis, tahun)
    # yang ada di data Excel — fleksibel sesuai jumlah perusahaan di data
    kewajiban_set = {}  # (npwp_bersih, jenis_bersih, tahun) → (npwp_raw, jenis_raw)
    # bulan_global: (jenis_bersih, tahun) → set of bulan yang ada di SELURUH data
    # Acuan kewajiban bulan untuk semua NPWP supaya "belum bayar" bisa terdeteksi
    bulan_global = {}

    for d in data_bayar:
        try:
            bulan = clean_bulan(d.bulan_pajak)
            if bulan == 0:
                continue
            npwp_bersih  = clean_npwp(d.npwp)
            jenis_bersih = clean_jenis(d.jenis_pajak)
            tahun = int(d.masa_pajak.split()[-1]) if d.masa_pajak else datetime.now().year
            key_kewajiban = (npwp_bersih, jenis_bersih, tahun)
            if key_kewajiban not in kewajiban_set:
                kewajiban_set[key_kewajiban] = (d.npwp, d.jenis_pajak)
            # Catat bulan ini ke bulan global jenis+tahun
            key_global = (jenis_bersih, tahun)
            if key_global not in bulan_global:
                bulan_global[key_global] = set()
            bulan_global[key_global].add(bulan)
        except Exception as e:
            print(f"Skip build kewajiban: {e}")
            continue

    hasil_detail = []   # detail per baris (untuk tabel lengkap)
    ringkasan    = {}   # ringkasan per NPWP

    for (npwp, jenis, tahun), (npwp_raw, jenis_raw) in kewajiban_set.items():
        # Search bisa pakai format asli (76.532) atau bersih (76532)
        if search and search not in npwp_raw and search not in npwp:
            continue

        if npwp not in ringkasan:
            ringkasan[npwp] = {
                'npwp'     : npwp_raw,
                'nama'     : nama_map.get(npwp, '-'),
                'tepat'    : 0,
                'terlambat': 0,
                'belum'    : 0,
            }

        # Gunakan bulan GLOBAL per jenis+tahun sebagai acuan kewajiban
        # Jika NPWP tidak bayar di bulan tertentu → status "Belum Bayar"
        for bulan in sorted(bulan_global.get((jenis, tahun), [])):
            key = (npwp, jenis, bulan, tahun)
            jt  = hitung_jatuh_tempo(jenis, bulan, tahun)

            if key in pembayaran:
                tgl = pembayaran[key]
                if isinstance(tgl, datetime):
                    tgl = tgl.date()
                selisih = (tgl - jt).days

                if selisih <= 0:
                    status = 'Tepat Waktu'
                    ringkasan[npwp]['tepat'] += 1
                else:
                    status = 'Terlambat'
                    ringkasan[npwp]['terlambat'] += 1

                selisih_tampil = abs(selisih)
            else:
                status         = 'Belum Bayar'
                selisih_tampil = None
                tgl            = None
                ringkasan[npwp]['belum'] += 1

            hasil_detail.append({
                'npwp'       : npwp_raw,
                'nama'       : nama_map.get(npwp, '-'),
                'jenis'      : jenis_raw,
                'bulan'      : nama_bulan[bulan],
                'tahun'      : tahun,
                'jatuh_tempo': jt.strftime('%d/%m/%Y'),
                'tgl_bayar'  : tgl.strftime('%d/%m/%Y') if tgl else '-',
                'status'     : status,
                'selisih'    : selisih_tampil,
            })

    return render_template(
        'kewajiban.html',
        data=hasil_detail,
        ringkasan=list(ringkasan.values()),
        search=search
    )
 
 
# ======================
# AGREGASI PAGE
# ======================
@app.route('/agregasi')
@login_required
def agregasi():
    dataset = _hitung_agregasi()
    return render_template('agregasi.html', data=dataset)
 
 
def _hitung_agregasi():
    """
    Fungsi internal: hitung agregasi per NPWP.
    Dipakai oleh route /agregasi dan route /kmeans.

    Kewajiban di-generate OTOMATIS dari data Excel yang diinput (PajakData),
    sehingga jumlah wajib pajak fleksibel mengikuti data yang ada —
    tidak terpaku pada isi tabel kewajiban_pajak di database.

    Return: list of dict {npwp, nama, tepat, telat, belum, avg_telat}
    """
    # Ambil semua data pembayaran dari Excel yang sudah diinput
    data_bayar = PajakData.query.all()

    if not data_bayar:
        return []

    # Bangun dict pembayaran: key=(npwp_bersih, jenis_bersih, bulan_int, tahun_int)
    pembayaran = build_pembayaran_dict(data_bayar)

    # Nama map: npwp_bersih → nama wajib pajak
    nama_map = {}
    for d in data_bayar:
        npwp_bersih = clean_npwp(d.npwp)
        if npwp_bersih not in nama_map:
            nama_map[npwp_bersih] = d.nama_wajib_pajak

    # Generate kewajiban otomatis dari kombinasi unik (npwp, jenis, tahun)
    # yang ada di data Excel — bukan dari tabel kewajiban_pajak
    # Sekaligus rekam rentang bulan GLOBAL per (jenis, tahun)
    # supaya "belum bayar" bisa terdeteksi dengan benar
    kewajiban_set = set()
    # bulan_global: (jenis_bersih, tahun) → set of bulan yang ada di SELURUH data
    # Ini dipakai sebagai acuan kewajiban bulan untuk semua NPWP
    bulan_global = {}

    for d in data_bayar:
        try:
            bulan = clean_bulan(d.bulan_pajak)
            if bulan == 0:
                continue
            npwp  = clean_npwp(d.npwp)
            jenis = clean_jenis(d.jenis_pajak)
            tahun = int(d.masa_pajak.split()[-1]) if d.masa_pajak else datetime.now().year
            kewajiban_set.add((npwp, jenis, tahun))
            # Catat bulan ini sebagai bulan kewajiban global untuk jenis+tahun ini
            key_global = (jenis, tahun)
            if key_global not in bulan_global:
                bulan_global[key_global] = set()
            bulan_global[key_global].add(bulan)
        except Exception as e:
            print(f"Skip generate kewajiban: {e}")
            continue

    hasil = {}   # key: npwp_bersih → dict counter

    for (npwp, jenis, tahun) in kewajiban_set:
        if npwp not in hasil:
            hasil[npwp] = {
                'tepat'        : 0,
                'telat'        : 0,
                'belum'        : 0,
                'total_selisih': 0,
                'count_bayar'  : 0
            }

        # Gunakan rentang bulan GLOBAL untuk jenis+tahun ini sebagai kewajiban
        # Sehingga jika NPWP tidak bayar di bulan tertentu → terdeteksi "Belum Bayar"
        bulan_kewajiban = sorted(bulan_global.get((jenis, tahun), []))

        for bulan in bulan_kewajiban:
            key = (npwp, jenis, bulan, tahun)
            jt  = hitung_jatuh_tempo(jenis, bulan, tahun)

            if key in pembayaran:
                tgl = pembayaran[key]
                if isinstance(tgl, datetime):
                    tgl = tgl.date()
                selisih = (tgl - jt).days

                if selisih <= 0:
                    hasil[npwp]['tepat'] += 1
                else:
                    hasil[npwp]['telat'] += 1

                hasil[npwp]['total_selisih'] += selisih
                hasil[npwp]['count_bayar']   += 1
            else:
                hasil[npwp]['belum'] += 1
 
    # Hitung persentase
    dataset = []
    for npwp, val in hasil.items():
        total = val['tepat'] + val['telat'] + val['belum']
        if total == 0:
            continue
 
        dataset.append({
            'npwp'     : npwp,
            'nama'     : nama_map.get(npwp, '-'),
            'tepat'    : val['tepat']  / total,
            'telat'    : val['telat']  / total,
            'belum'    : val['belum']  / total,
            'avg_telat': val['total_selisih'] / val['count_bayar'] if val['count_bayar'] > 0 else 0
        })
 
    return dataset


def _hitung_kmeans():
    """
    Fungsi internal: hitung clustering K-Means, Elbow, dan Silhouette.
    Return: dict atau None jika data tidak cukup
    """
    data = _hitung_agregasi()

    if not data or len(data) < 2:
        return None

    # ======================
    # FITUR
    # ======================
    fitur_names = ['Tepat Waktu', 'Terlambat', 'Belum Bayar', 'Rata-rata Keterlambatan']

    X = np.array([
        [d['tepat'], d['telat'], d['belum'], d['avg_telat']]
        for d in data
    ])

    # ======================
    # NORMALISASI
    # ======================
    X_min  = X.min(axis=0)
    X_max  = X.max(axis=0)
    X_norm = (X - X_min) / (X_max - X_min + 1e-9)

    # ======================
    # ELBOW METHOD
    # ======================
    max_k   = min(8, len(X_norm))
    K_range = list(range(2, max_k))
    inertia = []

    for k in K_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        km.fit(X_norm)
        inertia.append(round(float(km.inertia_), 4))

    # ======================
    # K OPTIMAL (otomatis dari elbow method)
    # ======================
    k_optimal = _deteksi_k_elbow(K_range, inertia)

    # Fallback jika elbow tidak bisa dihitung (data terlalu sedikit)
    if k_optimal is None or k_optimal < 2:
        k_optimal = 2 if len(X_norm) >= 2 else 1

    # ======================
    # TRAINING FINAL
    # ======================
    model  = KMeans(n_clusters=k_optimal, random_state=42, n_init=10)
    labels = model.fit_predict(X_norm)

    # ======================
    # SILHOUETTE SCORE
    # ======================
    if len(set(labels)) > 1:
        silhouette = round(float(silhouette_score(X_norm, labels)), 3)
    else:
        silhouette = 0.0

    # ======================
    # LABEL CLUSTER DINAMIS
    # Tentukan mana Patuh/Kurang Patuh/Tidak Patuh
    # berdasarkan rata-rata 'tepat' per cluster
    # (cluster dengan tepat tertinggi = Patuh)
    # ======================
    cluster_means = {}
    for cid in range(k_optimal):
        mask = (labels == cid)
        if mask.sum() == 0:
            continue
        cluster_means[cid] = {
            'avg_tepat': float(X_norm[mask, 0].mean()),
            'avg_belum': float(X_norm[mask, 2].mean()),
        }

    sorted_by_tepat = sorted(cluster_means.keys(),
                             key=lambda c: cluster_means[c]['avg_tepat'],
                             reverse=True)

    label_map = {}
    if k_optimal >= 3:
        label_map[sorted_by_tepat[0]]  = 'Patuh'
        label_map[sorted_by_tepat[-1]] = 'Tidak Patuh'
        for cid in sorted_by_tepat[1:-1]:
            label_map[cid] = 'Kurang Patuh'
    elif k_optimal == 2:
        label_map[sorted_by_tepat[0]]  = 'Patuh'
        label_map[sorted_by_tepat[-1]] = 'Tidak Patuh'
    else:
        # k_optimal == 1 (fallback ekstrem, data sangat sedikit)
        for cid in cluster_means.keys():
            label_map[cid] = 'Patuh'
    
    # ======================
    # CENTROID HISTORY (riwayat per iterasi)
    # Bangun dua variabel:
    #   centroid_history : list lengkap untuk JS/API
    #   iterasi          : list untuk template Jinja2 (sesuai format yang dipakai tabel Tahap 3)
    # ======================
    centroid_history = []
    iterasi_template = []
    prev_centers_norm = None

    if model.centroid_history_ is not None:
        label_hist = model.label_history_ or []
        n_hist = len(model.centroid_history_)

        for iter_idx, centers_norm in enumerate(model.centroid_history_):
            centers_orig = centers_norm * (X_max - X_min + 1e-9) + X_min
            is_last = (iter_idx == n_hist - 1)

            # Hitung pergeseran centroid dari iterasi sebelumnya
            if prev_centers_norm is not None:
                shift = float(np.sum((centers_norm - prev_centers_norm) ** 2))
                shift_str = f"{shift:.6f}"
            else:
                shift_str = None  # iterasi 0 tidak ada pergeseran
            prev_centers_norm = centers_norm.copy()

            # Jumlah anggota per cluster (dari label_hist iterasi ini)
            lbl_arr = label_hist[iter_idx] if iter_idx < len(label_hist) else None
            anggota_per_cluster = {}
            if lbl_arr is not None:
                for cid in range(k_optimal):
                    anggota_per_cluster[cid] = int(np.sum(lbl_arr == cid))

            # --- Format untuk centroid_history (API/JS) ---
            iter_entry = {
                'iterasi': iter_idx,
                'stabil': is_last,
                'clusters': []
            }
            # --- Format untuk iterasi_template (Jinja2) ---
            centroid_rows = []
            for cid in range(k_optimal):
                row = {
                    'cluster': cid,
                    'label': label_map.get(cid, f'Cluster {cid}'),
                    'tepat': round(float(centers_orig[cid, 0]), 4),
                    'telat': round(float(centers_orig[cid, 1]), 4),
                    'belum': round(float(centers_orig[cid, 2]), 4),
                    'avg_telat': round(float(centers_orig[cid, 3]), 2),
                    'norm': [round(float(v), 4) for v in centers_norm[cid]],
                    'jumlah_anggota': anggota_per_cluster.get(cid, '-'),
                }
                iter_entry['clusters'].append(row)
                centroid_rows.append(row)

            centroid_history.append(iter_entry)
            iterasi_template.append({
                'iter': iter_idx,
                'konvergen': is_last and iter_idx > 0,
                'shift': shift_str,
                'centroid_rows': centroid_rows,
            })

    total_iterasi = model.n_iter_
    # Tampilkan maks 5 iterasi awal + iterasi terakhir agar halaman tidak terlalu panjang
    MAX_SHOW = 6
    if len(iterasi_template) > MAX_SHOW + 1:
        iterasi_dipotong = True
        iterasi_tampil = iterasi_template[:MAX_SHOW] + [iterasi_template[-1]]
    else:
        iterasi_dipotong = False
        iterasi_tampil = iterasi_template

    # ======================
    # CENTROID CLUSTER
    # ======================
    centroids_norm = model.cluster_centers_
    centroids_orig = centroids_norm * (X_max - X_min + 1e-9) + X_min
    centroids_list = []
    for cid in range(k_optimal):
        centroids_list.append({
            'cluster': cid,
            'label': label_map.get(cid, f'Cluster {cid}'),
            'tepat': round(float(centroids_orig[cid, 0]), 3),
            'telat': round(float(centroids_orig[cid, 1]), 3),
            'belum': round(float(centroids_orig[cid, 2]), 3),
            'avg_telat': round(float(centroids_orig[cid, 3]), 2),
            'norm': [round(float(v), 4) for v in centroids_norm[cid]],
        })

    # ======================
    # JARAK KE SETIAP CENTROID (menjelaskan alasan alokasi cluster)
    # K-Means menempatkan setiap titik ke cluster dengan CENTROID TERDEKAT
    # (Euclidean distance pada ruang fitur yang sudah dinormalisasi).
    # ======================
    jarak_centroid_list = []
    for i in range(len(X_norm)):
        titik = X_norm[i]
        jarak_per_cluster = []
        for cid in range(k_optimal):
            centroid = centroids_norm[cid]
            selisih = titik - centroid
            dist = float(np.sqrt(np.sum(selisih ** 2)))
            jarak_per_cluster.append({
                'cluster': cid,
                'label': label_map.get(cid, f'Cluster {cid}'),
                'centroid_norm': [round(float(v), 4) for v in centroid],
                'selisih': [round(float(v), 4) for v in selisih],
                'jarak': round(dist, 4),
            })
        # Urutkan dari yang terdekat -> terjauh
        jarak_per_cluster.sort(key=lambda x: x['jarak'])
        jarak_centroid_list.append(jarak_per_cluster)

    # ======================
    # HASIL TABEL
    # ======================
    hasil = []
    for i, d in enumerate(data):
        cid = int(labels[i])
        hasil.append({
            'npwp'         : d['npwp'],
            'nama'         : d['nama'],
            'tepat'        : round(d['tepat'], 3),
            'telat'        : round(d['telat'], 3),
            'belum'        : round(d['belum'], 3),
            'avg_telat'    : round(d['avg_telat'], 2),
            'cluster'      : cid,
            'label'        : label_map.get(cid, f'Cluster {cid}'),
            'kelas_aktual' : klasifikasi_aktual(d['tepat'], d['telat'], d['belum']),
            'norm'         : [round(float(v), 4) for v in X_norm[i]],
            'jarak_centroid': jarak_centroid_list[i],
        })

    # ======================
    # INSIGHT OTOMATIS
    # Interpretasi berdasarkan label dinamis dari label_map
    # ======================
    insight = []
    for cluster_id in range(k_optimal):
        anggota = [h for h in hasil if h['cluster'] == cluster_id]
        if not anggota:
            continue

        avg_tepat = float(np.mean([a['tepat'] for a in anggota]))
        avg_telat = float(np.mean([a['telat'] for a in anggota]))
        avg_belum = float(np.mean([a['belum'] for a in anggota]))
        avg_hari  = float(np.mean([a['avg_telat'] for a in anggota]))

        label = label_map.get(cluster_id, f'Cluster {cluster_id}')

        # Karakter berdasarkan dominasi nilai
        karakter = []
        if avg_belum >= 0.50:
            karakter.append(f"Belum bayar sangat tinggi ({avg_belum*100:.1f}%)")
        elif avg_belum >= 0.30:
            karakter.append(f"Belum bayar cukup tinggi ({avg_belum*100:.1f}%)")

        if avg_tepat >= 0.60:
            karakter.append(f"Mayoritas bayar tepat waktu ({avg_tepat*100:.1f}%)")
        elif avg_tepat >= 0.30:
            karakter.append(f"Separuh bayar tepat waktu ({avg_tepat*100:.1f}%)")

        if avg_telat >= 0.30:
            karakter.append(f"Terlambat cukup sering ({avg_telat*100:.1f}%)")

        if avg_hari > 10:
            karakter.append(f"Rata-rata keterlambatan {avg_hari:.1f} hari")
        elif avg_hari < 0:
            karakter.append(f"Rata-rata bayar {abs(avg_hari):.1f} hari sebelum jatuh tempo")

        # Interpretasi berdasarkan label dinamis
        if label == 'Patuh':
            interpretasi = (
                f"Wajib pajak sangat disiplin. Sebagian besar ({avg_tepat*100:.1f}%) "
                f"membayar tepat atau sebelum jatuh tempo dengan rata-rata "
                f"{abs(avg_hari):.1f} hari lebih awal. Tingkat kepatuhan tertinggi."
            )
        elif label == 'Tidak Patuh':
            interpretasi = (
                f"Wajib pajak tidak aktif membayar kewajiban pajak. "
                f"Sebesar {avg_belum*100:.1f}% kewajiban belum dibayar, "
                f"dan jika membayar rata-rata terlambat {avg_hari:.1f} hari. "
                f"Perlu tindak lanjut prioritas dari otoritas pajak."
            )
        else:
            interpretasi = (
                f"Wajib pajak dalam kategori menengah. Aktif membayar namun "
                f"konsistensi masih perlu ditingkatkan. Sekitar {avg_belum*100:.1f}% "
                f"kewajiban belum terpenuhi dan rata-rata terlambat {avg_hari:.1f} hari."
            )

        insight.append({
            'cluster'      : cluster_id,
            'label'        : label,
            'avg_tepat'    : round(avg_tepat, 3),
            'avg_telat'    : round(avg_telat, 3),
            'avg_belum'    : round(avg_belum, 3),
            'avg_hari'     : round(avg_hari,  2),
            'karakter'     : karakter,
            'interpretasi' : interpretasi,
            'contoh'       : anggota[:5],
            'jumlah'       : len(anggota),
        })

    # Urutkan insight: Patuh → Kurang Patuh → Tidak Patuh
    urutan = {'patuh': 0, 'kurang patuh': 1, 'tidak patuh': 2}
    insight.sort(key=lambda x: urutan.get(x['label'].lower(), 9))

    # ======================
    # PURITY
    # ======================
    purity = _hitung_purity(hasil, k_optimal)

    return {
        'data': hasil,
        'inertia': inertia,
        'k_range': K_range,
        'silhouette': silhouette,
        'insight': insight,
        'k_optimal': k_optimal,
        'centroids': centroids_list,
        'purity': purity,
        'fitur_names': fitur_names,
        'centroid_history': centroid_history,
        'n_iter': model.n_iter_,
        'iterasi': iterasi_tampil,
        'total_iterasi': total_iterasi,
        'iterasi_dipotong': iterasi_dipotong,
    }


# ======================
# KMEANS
# ======================
@app.route('/kmeans')
@login_required
def kmeans():
    res = _hitung_kmeans()
    if res is None:
        return render_template(
            'Kmeans.html',
            data=[],
            inertia=[],
            k_range=[],
            silhouette=0,
            insight=[],
            centroids=[],
            purity=None,
            fitur_names=[],
            centroid_history=[],
            n_iter=0,
            iterasi=[],
            total_iterasi=0,
            iterasi_dipotong=False,
            error="Data tidak cukup untuk clustering (minimal 2 wajib pajak)"
        )

    # RENDER
    return render_template(
        'Kmeans.html',
        data=res['data'],
        inertia=res['inertia'],
        k_range=res['k_range'],
        silhouette=res['silhouette'],
        insight=res['insight'],
        centroids=res['centroids'],
        purity=res['purity'],
        fitur_names=res['fitur_names'],
        centroid_history=res['centroid_history'],
        n_iter=res['n_iter'],
        iterasi=res['iterasi'],
        total_iterasi=res['total_iterasi'],
        iterasi_dipotong=res['iterasi_dipotong'],
        error=None
    )
 


@app.route('/silhouette')
@login_required
def silhouette_endpoint():
    """Return the silhouette coefficient for the current optimal clustering as JSON."""
    res = _hitung_kmeans()
    if res is None:
        return {"silhouette": None, "error": "Data not sufficient"}
    return {"silhouette": res['silhouette']}





@app.route('/api/purity')
@login_required
def purity_api():
    """Return the Purity value (and detail per cluster) for the current clustering as JSON."""
    res = _hitung_kmeans()
    if res is None:
        return {"purity": None, "error": "Data not sufficient"}
    return {
        "purity": res['purity']['nilai'],
        "total_majority": res['purity']['total_majority'],
        "N": res['purity']['N'],
        "detail_cluster": res['purity']['detail_cluster'],
        "interpretasi": res['purity']['interpretasi']
    }

# ======================
# RUN APP
# ======================
if __name__ == "__main__":
    with app.app_context():
        db.create_all()

    app.run(
        host="127.0.0.1",
        port=5000,
        debug=True
    )
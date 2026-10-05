# Research Notes: Next-Generation Computing Substrate + AI Architecture

Status: eksplorasi awal. Semua klaim di bawah ini dipisah jelas jadi **(F) Fakta dari literatur**, **(E) Hasil eksperimen sendiri**, dan **(H) Hipotesis belum teruji** — jangan campur ketiganya saat nulis ulang/sitasi.

---

## 0. RINGKASAN EKSEKUTIF (baca ini dulu, detail lengkap di §1-24 di bawah)

**PERINGATAN BACA DULU**: versi awal ringkasan ini (dan §23) udah diaudit Opus ronde 2 dan **dikoreksi keras** — lihat §24. Tabel di bawah ini versi SETELAH koreksi, bukan versi asli.

**Pertanyaan awal**: komputer/AI generasi baru apa yang bisa dukung AI jauh lebih maju (arah ASI)?

**Arah resmi saat ini** (§15, disetujui user 2026-10-02): **BUKAN** bangun arsitektur baru. Kita jadi **auditor metodologi** — bangun protokol evaluasi yang rigorous (collapse-diagnostic + downstream-probe, bukan raw loss), pakai itu buat ngaudit klaim yang udah published di bidang SNN/SSL self-supervised learning.

### Temuan Paling Penting (urut dari paling kuat buktinya, POST-KOREKSI §24)

| # | Temuan | Kekuatan bukti | Detail |
|---|---|---|---|
| 1 | Eksperimen SpikingJEPA kita sendiri **collapse total**, bukan menang | **KUAT** — 45 run, 3 seed × 12 kombinasi hyperparameter, 0% lolos. Catatan: sumbu yang disweep (beta, LR) BUKAN sumbu yang paling mungkin kausal (threshold/readout/EMA-decay gak pernah divariasikan) — "robust terhadap hyperparameter" itu klaim yang terlalu luas, lihat §24 | §16-17 |
| 2 | **EventCLR: "collapse" yang diukur sepanjang §21g-26c SEBAGIAN BESAR artefak mismatch statistik BatchNorm, BUKAN murni representasi gagal belajar** — dibuktikan via 3 kondisi A/B/C terkontrol (angka TERKOREKSI di §28 setelah Opus audit ronde 3 nemu bug `reset_net` di script kita: A=26.95%, B=71.48%, C=77.73%) | **KUAT untuk BN-sebagai-confound-dominan**, TAPI implikasinya BUKAN "EventCLR representasinya bagus" — implikasinya "protokol audit KITA SENDIRI punya blind spot baru". Angka §26 awal SUPERSEDED, pakai §28b. | §26e-26f-bis, **§28 (angka final)** |
| 3 | 3 paper lain yang dicek (LPL/ESPP/PC-SNN) **TIDAK** rentan ke masalah yang sama kayak kita, masing-masing beda alasan | **SEDANG** — baca kode/paper langsung, bukan cuma abstract | §21a-21d |
| 4 | Von Neumann vs neuromorphic = sama-sama Turing-complete (ideal, unbounded memory); Transformer vs Mamba = sama-sama bounded TC0 (satu forward-pass, presisi terbatas) | **KUAT secara individual, TAPI "pola yang sama 2x" itu SALAH** — dua hasil diukur di rezim resource yang beda, perbandingannya nyaris tautologi (lihat §24a) | §12, §22d, §24a |
| 5 | Chain-of-thought bisa Turing-complete di model ideal | **Benar tapi BUKAN sintesis baru** (restatement Turing 1936), dan BUKAN satu-satunya mekanisme (§13 kita sendiri udah punya mekanisme kedua: rational-weight RNN) | §22e, §24a |
| 6 | **PredNext paper (ICLR 2026) nyebut collapse 3x sebagai risiko nyata, ablation SENDIRI hasilnya 5.03% (collapse), tapi nol diagnostic rank/variance + nol frozen linear-probe di seluruh paper** | **KUAT** — audit full-paper (bukan abstract), argumen kuantitatif d/N=16, anomali Table 3 (SimSiam di bawah random-init floor), gap frozen-vs-finetuned 3.7x, error matematis di Appendix G. **Scope ketat: collapse TOTAL udah ditolak buktinya sendiri (Table 5 R@1 jauh di atas chance) — yang masih hidup cuma hipotesis PARTIAL/dimensional collapse** | §25 |
| 7 | **SynOps energy model: klaim "spiking/sparse = hemat energi" TIDAK terbukti buat arsitektur kita** — angka TERKOREKSI (§28, Opus audit ronde 3 nemu bug unit 1000x + inkonsistensi harga energi): GPU 29.49 µJ/sampel, neuromorphic dgn pricing KONSISTEN **140.33 µJ/sampel → 4.8x LEBIH MAHAL**, bukan "nyaris impas" kayak klaim awal §27 | **KUAT untuk arsitektur SPESIFIK kita, arah kesimpulan malah lebih tegas negatif setelah koreksi** — layer pertama (92% compute) gak sparse sama sekali. **Jangan generalize ke "neuromorphic gak pernah hemat"** — LTC di Loihi-2 asli (§6) terbukti hemat (213µJ/frame, sekarang ORDE BESARAN SEBANDING setelah unit-bug diperbaiki) | §27 (angka awal, SUPERSEDED), **§28c (final)** |
| 8 | **Phase B FINAL (§34-35): Dense (51.89%) > GRU param-matched (26.46%) > Spiking-v2 (23.76%) > trivial. Sparsity murni (parameter disamain) cuma ongkos ~2.7pp, bukan fatal** | **KUAT, angka final yang dipakai ke depan** — semua confound besar (BN, state-leak, eval-length, probe-convergence, parameter-count) udah dibersihin satu-satu | ~~§30-32~~ SUPERSEDED, **§35 (final, pakai ini)** |
| 9 | **Lava (framework Loihi 2) resmi mati (archived Mei 2026); INRC butuh PI (gak bisa individual student) — EBRAINS jadi jalur hardware-asli utama (nol syarat PI)** | **KUAT** — diverifikasi langsung ke GitHub API + halaman resmi INRC, bukan dari ingatan/asumsi | §29 |

### Sintesis §23 — **DITARIK, lihat §24a buat lubang logikanya**
Versi asli §23 ("lever-nya cuma WAKTU, bukan substrat") **ditarik substansial**, bukan dikasih hedging lebih berat — Opus (dan kita, setelah cross-check) nemu itu lubang logika struktural: (a) perbandingan "pola sama 2x" itu artefak definisional complexity-class, bukan temuan, (b) frame complexity-class ini udah divonis salah di §14c SENDIRI dan ditutup di §15, dibuka lagi tanpa sadar, (c) ada minimal 2 mekanisme lain buat nyebrang kelas yang ada DI DOKUMEN KITA SENDIRI (§13 rational-weight RNN, dan "CoT itu mekanisme SPACE berpakaian waktu") yang gak diperiksa karena cerita "WAKTU" udah kerasa pas duluan. **Pelajaran prosesnya lebih berharga dari klaimnya**: kita (dan Opus ronde 1) sama-sama gak nyadar lagi buka frame yang udah ditutup sendiri — itu persis failure mode yang proyek audit ini ada buat nangkep, kejadian ke penulisnya sendiri.

### Yang BELUM Selesai / Masih Kebuka
- ~~PredNext audit berbasis paper~~ — **SELESAI 2026-10-02, lihat §25**. Hasil: temuan kuat & well-calibrated, collapse total ditolak, partial collapse masih hidup sebagai hipotesis belum terbukti. Belum ada aksi publik (OpenReview comment dll) — nunggu keputusan user.
- File 4 bug EventCLR (O(N²) OOM di diagnostic mereka, lmdb hilang, cupy hardcoded, nol checkpoint) sebagai GitHub issue/PR — kontribusi ter-atribusi, **butuh konfirmasi eksplisit dulu** (aksi publik ke repo orang lain)
- SpikeMamba (kode publik, SNN+Mamba) — ketemu, **provenance belum diverifikasi** (org publish banyak repo wrapper tipis), jangan masuk daftar kandidat sebelum dicek
- Neurosymbolic AI & Active Inference/FEP — cuma disebut nama di §3, belum pernah di-deep-dive
- ~~Linear-probe + trivial-baseline buat EventCLR~~ — **SELESAI 2026-10-02, lihat §26a-26d**.
- ~~Verifikasi hipotesis BN-unfrozen EventCLR~~ — **SELESAI & DIKONFIRMASI KUAT, lihat §26e-26f**. Temuan terbesar session ini: sebagian besar sinyal "collapse" EventCLR adalah artefak BatchNorm, bukan representasi — dan ini kelemahan baru di PROTOKOL AUDIT KITA SENDIRI, bukan cuma soal EventCLR doang.
- **BARU**: cek apakah PredNext (SimSiam/BYOL/BarlowTwins, standar pakai BN di projector) kena confound yang sama — gak ada kode publik buat dicek langsung, tapi layak dicatat sebagai kelemahan tambahan di kritik §25
- Gap 23pp yang masih belum kejelasin di eksperimen BN (67.8% vs 90.4% EventCLR) — kandidat: lebih banyak batch rekalibrasi, atau beda regime optimisasi probe (mini-batch SGD multi-epoch asli mereka vs full-batch 300-step kita)
- ~~SynOps energy model~~ — **SELESAI 2026-10-02, lihat §27**. Hasil sobering: rasio cuma 1.1x buat arsitektur kita, bukan karena idenya salah tapi karena layer pertama (92% compute) gak sparse — insight arsitektural buat kerjaan selanjutnya kalau mau beneran kejar efisiensi: input layer juga harus sparse-native, bukan cuma layer dalam

### Kelemahan yang Harus Diingat Siapa Pun yang Baca Ini
Dokumen ini udah 2x dikoreksi Opus (§14, §24) dan beberapa kali dikoreksi user langsung di tengah sesi — itu bagian dari proses, bukan kelemahan yang disembunyikan. **Pola spesifik yang udah kejadian 4x**: nulis "sama persis"/"identik" antara dua hal yang cuma sama LABEL-nya, bukan sama substansinya (§12, §21g, §22d, §23 — detail di §24a). Kalibrasi: sebelum nulis "sama persis", wajib tunjukin dua objeknya berdampingan.

---

## 1. Kenapa Komputasi Sekarang Ada Batasnya

**(F) Landauer's Principle (1961)** — batas energi minimum teoretis buat menghapus 1 bit informasi secara ireversibel: `kT ln 2` ≈ 2.85×10⁻²¹ J di suhu ruang. Komputer modern pakai energi ~10⁹x lebih banyak dari batas ini per operasi. Operasi yang **reversibel** (gak menghapus informasi) gak kena batas ini sama sekali.
Sumber: [Landauer's principle (Wikipedia)](https://en.wikipedia.org/wiki/Landauer%27s_principle), [Landauer Bound: Meaning, Experimental Verification](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC11119825/)

**(F) Von Neumann bottleneck** — di 2026, biaya "kecerdasan" AI ditentukan seberapa cepat data bisa dipindah dari memory ke compute unit, bukan raw FLOPS. Semua GPU/TPU/accelerator AI yang shipped 2026 masih hidup di arsitektur ini.
Sumber: [The von Neumann bottleneck: AI's real constraint is memory](https://aibottlenecks.app/research/von-neumann-bottleneck)

**(F) Moore's Law mentok fisika** — transistor mendekati ukuran atom, muncul electron tunneling & leakage current. Prediksi mentok total sekitar 2040-2050.
Sumber: [Is Moore's law dead? (imec)](https://www.imec-int.com/en/semiconductor-education-and-workforce-development/microchips/moores-law/moores-law-dead)

## 2. Kandidat Substrat Hardware Generasi Baru

| Paradigma | Mekanisme | Status 2026 | Sumber |
|---|---|---|---|
| **Reversible/adiabatic computing** | Setiap operasi bisa dibalik, gak hapus informasi → secara teori energi dissipation bisa mendekati 0 | Vaire Computing (UK) chip "Ice River" recover 40-70% energi; Sandia National Labs (Michael Frank) nembus batas Landauer lewat hardware reversible. Masih "laboratory curiosity" 30 tahun. | [EE Times - Vaire](https://www.eetimes.com/vaire-demos-energy-recovery-with-reversible-computing-test-chip/), [Reversibility and energy dissipation in adiabatic superconductor logic](https://www.nature.com/articles/s41598-017-00089-9) |
| **Superconducting** | Elektron tanpa resistansi (diketahui sejak 1911), tapi butuh suhu cryogenic sub-30mK | 2026: qubit + kontrol elektronik terintegrasi di 2 chip terpisah terhubung (Nature Electronics); Modular Cryogenic Platform mulai pengiriman multi-module akhir 2026 | [Superconducting computing (Wikipedia)](https://en.wikipedia.org/wiki/Superconducting_computing) |
| **Neuromorphic** | Niru struktur otak, event-driven (spiking), address-event routing | **Paling matang** — Intel Loihi 2 & BrainChip Akida transisi riset→produk. Dipakai Mercedes-Benz: 1000x hemat energi, respons 200ms lebih cepat dari GPU | [Intel Loihi 2 brief](https://download.intel.com/newsroom/2021/new-technologies/neuromorphic-computing-loihi-2-brief.pdf) |
| **Photonic/optical** | Cahaya (bukan elektron) buat matrix multiplication lewat interferensi (Mach-Zehnder Interferometer mesh) | Breakthrough 2026: chip photonic AI dites diagnosis medis setara radiolog senior; hybrid light-matter particle (Penn) | [Photonic matrix multiplication (Nature)](https://www.nature.com/articles/s41377-022-00717-8) |
| **Memristor/in-memory** | Compute dan memory jadi satu komponen fisik (resistor yang "ingat" nilai) | Memristor Bi₂Se₃ (2D material) — pertama gabung long-term retention + analog tuning + operasi tanpa regulator | [Memristor (Michigan Engineering)](https://news.engin.umich.edu/2026/03/memristor-demonstrates-use-in-fully-analog-hardware-based-neural-network/) |
| **DNA/molecular** | Sirkuit untai DNA, compute+storage dalam sistem yang sama | KAIST 2026: bio-transistor DNA, circuit <2nm, proses real-time tanpa reset eksternal | [DNA molecular computer (phys.org)](https://phys.org/news/2026-04-dna-molecular-combines-memory-scales.html) |
| **Quantum** | Qubit/superposisi | **(F) PALING SPEKULATIF buat AI** — konsensus 2026: belum ada quantum advantage terbukti buat deep learning di data dunia nyata. Quantum gak lebih cepat buat bulk linear algebra (inti training neural network). Jangan jadi fokus utama riset ini. | [Quantum ML 2026: Beyond the Hype](https://www.opodab.com/2026/09/quantum-machine-learning-technical-guide-ml-engineers-2026.html) |

**(F) Detail teknis Loihi 2** (substrat paling konkret buat eksperimen): 128 neurocore async + 6 core x86, total 1 juta neuron/123 juta synapse per chip. Tiap neurocore 8,192 neuron, microcode-programmable. Timestep minimum 200ns. Support graded spike (bawa payload integer, bukan cuma on/off) — penting buat JEPA karena butuh representasi kontinu.

**(F) Detail mekanisme photonic MZI**: matrix multiplication terjadi FISIK lewat interferensi cahaya di mesh phase-shifter. Tantangan belum solved: hubungan phase-shift↔hasil matrix itu non-linear, susah di-program presisi.

## 3. Kandidat Arsitektur AI "Generasi Baru" (bukan transformer biasa)

**(F) JEPA (Joint Embedding Predictive Architecture)** — Yann LeCun. Argumen: LLM (prediksi token) fundamentally limited buat world-understanding karena teks itu lossy compression dunia. JEPA prediksi REPRESENTASI ABSTRAK dari bagian tersembunyi, bukan detail mentah. Implementasi nyata: V-JEPA (Meta, 2024).

**(F) Neurosymbolic AI** — gabungan neural network (pattern recognition) + symbolic logic (reasoning formal). Target: genuine reasoning, bukan cuma interpolasi statistik.

**(F) Active Inference / Free Energy Principle** (Karl Friston) — framework dari neuroscience: otak minimalin "prediction error"/surprise. Kerangka matematika beda total dari gradient descent biasa.

**(F) Definisi kasar ASI** (Nick Bostrom, *Superintelligence*, 2014): "intelek yang jauh ngelampaui performa kognitif manusia di HAMPIR SEMUA domain." Kriteria kasar: (1) generalisasi lintas-domain, (2) bentuk abstraksi baru di luar training data, (3) recursive self-improvement, (4) sample efficiency jauh di atas manusia.

## 4. Tooling Gratis buat Eksperimen (Student Budget = Rp0)

- **snnTorch** — PyTorch extension buat spiking neural network, GPU-accelerated, gradient-based training. [docs](https://snntorch.readthedocs.io/en/stable/readme.html)
- **Lava (Intel)** — framework open-source resmi buat SNN, BSD-3/LGPL-2.1, jalan di CPU/GPU tanpa chip Loihi fisik, model yang sama bisa di-port ke Loihi 2 asli nanti.
- **INRC (Intel Neuromorphic Research Community)** — akses GRATIS ke Loihi 2 asli via cloud buat Research Member (perlu afiliasi institusi riset + project proposal). [Join steps](https://intel-ncl.atlassian.net/wiki/spaces/INRC/pages/1784807425/Join+the+INRC)

## 5. Log Eksperimen Sendiri — Status & Masalah Terbuka

**(E) Eksperimen 1** (`train.py`, task "easy", 1 seed): Dense JEPA loss 0.00014 vs Spiking JEPA loss 0.00000, spike_rate 0.043.

**(E) Eksperimen 2** (`multi_seed_experiment.py`, task "hard", 10 seed, parameter count disamakan 42,496 vs 42,496): Dense loss mean 0.00011±0.00001, Spiking loss mean 0.00000±0.00000, spike_rate mean 0.027±0.003. Gap konsisten di semua 10 seed.

**(H) MASALAH TERBUKA — confound belum terselesaikan**: neuron LIF (spiking) secara matematis adalah integrator temporal (akumulasi membrane potential antar-timestep) yang berfungsi seperti low-pass filter. Data sintetis kita (sinusoid+coupling) itu sinyal mulus/periodik — cocok secara kebetulan sama karakteristik low-pass filter. **Kemungkinan besar spiking "menang" karena inductive bias LIF yang cocok sama data, BUKAN karena sparsity/efficiency-nya.** Belum ada bukti sebab-akibat yang bersih.

**Next step buat isolasi variabel** (belum dijalankan): tambah kontrol ketiga — GRU/RNN dense (punya temporal integration SAMA seperti LIF, tapi TANPA sparsity). Kalau GRU dense juga ngalahin MLP dense dengan margin serupa → yang menang itu temporal integration, bukan sparsity. Spiking cuma bisa diklaim unggul kalau dia ngalahin GRU dense JUGA, sambil tetep pakai spike_rate rendah.

**Kelemahan desain eksperimen lain yang belum diperbaiki**:
- Data sintetis masih terlalu "bersih"/periodik — real-world data (video, sensor fisik) punya struktur non-periodik yang lebih sulit
- Belum ada perbandingan terhadap literatur eksternal (belum tau apakah hasil ini sejalan/kontradiksi sama paper neuromorphic-JEPA yang mungkin udah ada — belum dicari secara spesifik)
- `repr_dim=64` dan `hidden_dim=128` dipilih sembarangan, belum ada justifikasi/sensitivity analysis

## 6. Kelemahan & Kegagalan Nyata — Kenapa Belum Ada yang Gantiin Komputer Sekarang

Ini bagian yang sebelumnya kurang: semua paradigma di §2-3 punya masalah NYATA yang belum solved, bukan cuma "belum sempat di-scale up".

**(F) Neuromorphic — kenapa von Neumann masih dominan:**
- **Gak ada standar benchmark.** Beda sama ANN yang punya ImageNet/GLUE dkk sebagai ukuran baku, neuromorphic computing belum punya cara standar buat ngukur performa — sulit bandingin klaim satu riset ke riset lain.
- **Gak ada software/hardware interface standar.** Kebanyakan riset neuromorphic masih pakai bahasa & algoritma yang didesain buat von Neumann — ekosistemnya belum matang buat "native" neuromorphic programming.
- **Overhead ADC (Analog-to-Digital Converter)** — bottleneck tersembunyi di in-memory computing: computation-nya analog, tapi harus dikonversi balik ke digital buat nyambung ke sistem lain, dan konversi ini makan energi/area chip yang signifikan.
- **Alasan paling praktis**: ekosistem von Neumann udah MATANG (compiler, library, tools, talent pool) buat general-purpose computing. Neuromorphic cuma menang di domain SANGAT spesifik (event-driven sensing, low-power edge inference) — bukan pengganti general-purpose.

**(F) Memristor — masalah reliability yang serius:**
- **Device variability brutal**: resistansi antar switching cycle bisa fluktuasi **>20% dari nilai nominal** (device TiO₂) — ini bukan noise kecil, ini bikin hasil komputasi gak presisi.
- **Precision rendah**: cuma **4-6 bit effective precision** karena nonlinearity & stochasticity fisik — jauh di bawah presisi floating-point yang biasa dipakai deep learning.
- **Endurance terbatas**: cuma tahan **10⁶-10⁹ write cycle** sebelum degradasi performa — buat training AI yang butuh update weight berulang-ulang jutaan kali, ini bisa jadi masalah umur pakai nyata.
- **Drift & sensitif suhu** — nilai memristor bisa "melayang" dari waktu ke waktu dan berubah sama suhu, bikin kalibrasi ulang terus-terusan.

**(F) Kasus nyata: prosesor "biasa" pun bisa cacat permanen — Intel 13th/14th Gen (2024)**
Ini contoh penting buat kalibrasi ekspektasi: bahkan silicon CONVENTIONAL yang udah dikembangin puluhan tahun dengan dana raksasa masih bisa gagal fatal:
- **Root cause**: bug microcode di algoritma "Enhanced Thermal Velocity Boost" (eTVB) ngasih permintaan voltase yang salah ke processor
- **Mekanisme kerusakan**: "Vmin Shift Instability" — di voltase & suhu tinggi, sirkuit clock di dalam core core **terdegradasi secara fisik dari waktu ke waktu**
- **Permanen & gak bisa di-undo**: chip yang udah terdegradasi TETAP rusak walau microcode-nya di-patch — patch cuma nyegah kerusakan BARU, gak bisa benerin yang udah kena
- **Pelajaran buat riset ini**: kalau silicon conventional (teknologi paling matang, paling banyak diinvestasi di dunia) masih bisa punya cacat desain yang gak kedeteksi sampai jutaan unit terjual, **substrat eksperimental (memristor, neuromorphic, photonic) yang jauh lebih baru punya risiko reliability yang jauh lebih tinggi** — ini alasan kuat kenapa industri hati-hati banget sebelum adopsi masal.

**(F) Photonic — 5 masalah scalability utama**: electro-optic efficiency, computing parallelism, spatial integration, reconfigurability, robustness. Detail konkret: (1) all-optical gate butuh power level mW buat efek nonlinear yang dibutuhkan switching — boros; (2) buat integrasi >10⁴ aktuator photonic, loss per komponen harus ditekan **di bawah 0.15dB** — target fabrikasi yang sangat ketat; (3) reconfigurable photonic circuit kompleksitasnya **O(n·2ⁿ)** untuk n-operand — naik eksponensial, bikin circuit besar gak praktis dikontrol.

**(F) Spiking Neural Network — masalah training fundamental (langsung relevan ke eksperimen kita sendiri):**
- **Fungsi spike itu non-differentiable** — gak bisa dipakai gradient descent/backpropagation standar secara langsung, karena spike itu diskrit (ada/gak ada), bukan kontinu.
- **Solusinya (yang kita PAKAI tanpa sadar implikasinya)**: `surrogate.fast_sigmoid()` di kode `spiking_jepa.py` kita itu **surrogate gradient** — pendekatan buatan yang "nge-pura-pura" fungsi spike itu differentiable pas training, padahal aslinya enggak. Ini trik yang umum dipakai, tapi artinya training SNN itu **gak 100% prinsip asli neuromorphic (event-driven native training)** — kita masih "nebeng" gradient descent gaya ANN dengan modifikasi, bukan cara belajar yang genuinely baru.
- **Gap akurasi**: SNN secara umum **masih di bawah ANN** di benchmark visual standar, walau gap-nya makin nyempit (dan di task tertentu malah bisa nyamain/ngelampaui, seperti yang keliatan di eksperimen kita — tapi ingat confound yang udah dicatat di §5).

**Implikasi buat eksperimen kita**: hasil "spiking menang" di §5 jadi makin perlu dicurigai — bukan cuma soal confound LIF-sebagai-filter, tapi karena **cara kita training spiking model-nya sendiri pakai surrogate gradient (bukan native event-driven learning)**, jadi perbandingannya mungkin gak fair dari sisi METODE training-nya juga, gak cuma arsitekturnya.

## 7. Logika/Teori buat Ketahanan (Durability) — Menjawab Masalah Endurance di §6

Masalah di §6 (memristor cuma tahan 10⁶-10⁹ cycle, variability >20%, Intel CPU bisa degradasi permanen) bukan masalah tanpa solusi teoretis — ada beberapa cabang riset yang eksplisit nyerang ini:

**(F) Von Neumann's own fault-tolerance theorem (1956, "Probabilistic Logics")** — ironis: orang yang arsitekturnya mau kita lampaui JUGA yang ngasih fondasi matematis buat bikin sistem reliable dari komponen yang gak reliable. Hasil intinya: **selama probabilitas error tiap komponen individual < 1/6, redundansi terstruktur (replikasi + majority-logic voting) bisa bikin keseluruhan sistem seandal yang kita mau** — berapa pun kecilnya target error rate yang diinginkan, selalu ada desain sirkuit (lebih dalam/redundan secara konstan) yang mencapainya. **Ini teorema yang paling relevan buat substrat eksperimental** (memristor/neuromorphic) yang masing-masing device-nya gak reliable, tapi sistem gabungannya BISA dibuat reliable kalau redundansi-nya didesain benar.
Sumber: [Von Neumann's Contributions to Automata Theory (Shannon)](https://www.ams.org/bull/1958-64-03/S0002-9904-1958-10214-1/S0002-9904-1958-10214-1.pdf)

**(F) Wear-leveling + Error Correction Code (ECC) gabungan** — pendekatan nyata yang udah dipakai industri (awalnya buat SSD/EEPROM, sekarang diadaptasi ke memristor): (1) **wear-leveling** — distribusi write operation secara EVEN ke seluruh array, supaya gak ada sel yang kepake berlebihan sementara yang lain nganggur; (2) **ECC** — deteksi & koreksi error dari sel yang udah mulai degradasi; (3) **row-sparing** — kalau ECC udah terlalu sering ke-trigger di baris tertentu (tanda sel itu "lemah"), baris itu "dipensiunkan" dan diganti baris cadangan. Pendekatan **RETROFIT** gabungin ketiganya dan terbukti bisa extend lifespan jauh di atas spek endurance standar.
Sumber: [Realizing Extreme Endurance Through Fault-aware Wear Leveling](https://ieeexplore.ieee.org/document/10071093/), [WoLFRaM](https://arxiv.org/html/2010.02825v1)

**(F) Analog Error-Correcting Codes — didesain khusus buat 2 jenis noise analog:**
- **Limited-Magnitude Errors (LME)** — noise kecil tapi tersebar luas (dari cell-programming noise, gangguan read/write biasa) — DNN secara umum TOLERAN ke ini
- **Unlimited-Magnitude Errors (UME)** — jarang terjadi tapi SANGAT merusak (sel yang "macet"/stuck, short circuit) — ini yang butuh ECC serius, karena DNN rentan ke error besar yang terisolasi walau toleran ke noise kecil tersebar
- Implikasi desain: strategi koreksi error-nya HARUS beda buat dua jenis noise ini, bukan satu solusi buat semua
Sumber: [Analog Error-Correcting Codes for Dependable In-Memory Computing](https://link.springer.com/chapter/10.1007/978-3-032-17948-7_7), [A New Class of Geometric Analog ECC](https://arxiv.org/html/2603.03723)

**(F) Self-healing & robustness intrinsik neuromorphic — ini temuan paling menjanjikan:**
- **Circuit self-repair terinspirasi astrocyte** (sel pendukung neuron di otak biologis, bukan neuron itu sendiri) — tiap astrocyte "momong" beberapa neuron, bisa kompensasi synapse yang rusak dengan modifikasi voltase synapse sehat di sekitarnya. Hasil terukur: **fault tolerance rate 81.10%, resilience improvement 18.90%**.
- **Material self-healable buat memtransistor** — kerusakan fisik di material-nya sendiri bisa "pulih sendiri", bukan cuma dikompensasi di level sirkuit.
- **Robustness intrinsik yang udah terukur**: algoritma neuromorphic **BISA TAHAN kehilangan 32% neuron dan 90% spike tanpa degradasi akurasi signifikan** — ini angka yang jauh lebih toleran dibanding ANN konvensional yang biasanya langsung collapse kalau sebagian besar unit dimatikan.
Sumber: [Astromorphic Self-Repair of Neuromorphic Hardware](https://arxiv.org/pdf/2209.07428), [Self healable neuromorphic memtransistor (Nature)](https://www.nature.com/articles/s41467-020-17870-6)

**Sintesis buat riset kita**: kombinasi yang paling masuk akal buat "komputer gen baru yang tahan lama" bukan nunggu material yang sempurna (gak akan pernah ada), tapi **menerima bahwa komponen individual gak reliable, dan desain redundansi di level sistem** (persis teorema Von Neumann) — neuromorphic punya KEUNTUNGAN ALAMI di sini karena arsitekturnya yang terdistribusi (banyak neuron kecil) secara natural lebih toleran kehilangan sebagian unit dibanding chip monolitik kayak CPU konvensional (makanya Intel 13th/14th Gen gagal total, bukan "degradasi graceful" — gak ada redundansi di desainnya).

## 8. Analisis Von Neumann Lebih Dalam (reasoning sendiri, dicek ulang)

**(H→sebagian F setelah dicek)** Akar masalah von Neumann bukan "memory lambat" — itu gejala. Akar masalahnya: **von Neumann machine didesain buat fleksibilitas maksimal (Turing-complete, alamat memory generik), tapi dipakai buat workload yang justru STATIS** (neural network yang udah dilatih punya computation graph fixed). Overhead (branch prediction, cache hierarchy, generic addressing) itu biaya buat fleksibilitas yang gak kepake di inference.

**Koreksi diri (penting)**: kesimpulan awal "solusinya hybrid, partisi statis→struktur fisik, dinamis→controller fleksibel" itu **BUKAN ide baru** — itu cuma restate pola CPU+GPU/TPU yang udah jadi standar industri sejak lama. **Confidence sebagai "kontribusi baru": ~0%.** Analogi ke System 1/System 2 reasoning juga masih spekulatif (confidence ~40-60%, belum diverifikasi teknis, cuma analogi kedengeran pas).

## 9. Local Learning Rules buat Predictive Objective — Cek Gap yang Diajukan Sebelumnya

**Hipotesis awal (di §7 lama)**: belum ada yang gabungin "world-model terus belajar" (JEPA) dengan "substrat yang cuma bisa di-training pakai local learning rule" (photonic/memristor in-situ). **Setelah dicek literatur: hipotesis ini SEBAGIAN SALAH** — gap-nya lebih sempit dari yang dikira.

**(F) Yang udah ada (closing sebagian besar gap):**
- **EchoSpike Predictive Plasticity (ESPP)** — online local learning rule buat SNN, pakai predictive+contrastive coding (semangat sama kayak JEPA: prediksi representasi, bukan reconstruction pixel), didesain buat neuromorphic processor murah.
- **Latent Predictive Learning (LPL)** — gabungan predictive coding + Hebbian learning (local rule) buat self-supervised learning di SNN.
- **Predictive coding ≈ backpropagation** (Whittington & Bogacz) — local Hebbian plasticity rule bisa approximate backprop secara matematis. **Tapi diakui sendiri: "struggle beyond simple problems and baseline architectures."**
- **Equilibrium Propagation (EqProp/EqSpike)** — local-in-space-AND-time, klaim estimasi 1000x hemat energi inference / 100x training di neuromorphic silicon (**estimasi, belum diukur di chip fisik**).
Sumber: [EchoSpike Predictive Plasticity](https://arxiv.org/pdf/2405.13976), [PC-SNN: Predictive Coding-based Local Hebbian Plasticity](https://arxiv.org/pdf/2211.15386), [On predictive coding vs backpropagation](https://journals.plos.org/plosone/article?id=10.1371%2Fjournal.pone.0266102), [Equilibrium Propagation with Continual Weight Updates](https://arxiv.org/pdf/2005.04168), [EqSpike](https://www.sciencedirect.com/science/article/pii/S2589004221001905)

**Gap yang BENERAN masih terbuka (lebih sempit dari hipotesis awal):**
1. **Scaling** — ESPP/LPL/predictive-coding semua diakui literaturnya sendiri "struggle beyond simple problems" — belum ada yang jalan di skala JEPA asli (world-model video/sequence kompleks).
2. **Validasi hardware fisik** — semua masih di simulasi SNN (GPU), belum dikonfirmasi jalan di chip photonic/memristor asli. Klaim efisiensi EqSpike itu estimasi, bukan pengukuran chip nyata.
3. **Trik EMA-target-encoder JEPA spesifik** belum ketemu paper yang eksplisit gabungin itu dengan predictive coding/EqProp — ada secara prinsip, implementasi eksplisitnya belum ketemu.

**Pelajaran metodologis**: hipotesis "gap riset" HARUS dicek ke literatur sebelum diklaim sebagai arah baru — percobaan pertama (§7 lama) hampir salah lapor "belum ada yang kerjain" padahal udah ada, cuma belum lengkap/scaled.

## 10. Metodologi Ablation SNN-vs-RNN — Riset Dulu Sebelum Desain Eksperimen

Sebelum eksekusi ablation GRU (masih PENDING approval, lihat aturan di memori `feedback_research_no_premature_execution`), cek dulu apa metodologi yang udah established di literatur — jangan desain sendiri dari nol.

**(F) Precedent yang ada**: cara standar bandingin SNN vs RNN itu **"quantized encoding" pada RNN baseline + cuma masukin timestep pertama ke ANN non-temporal** (karena ANN biasa gak punya dimensi waktu) — ini presisi/detail yang desain `ablation_gru.py` kita BELUM terapkan (kita kasih GRU full sequence, bukan quantized encoding yang sama persis sama yang spiking terima).
Sumber: [SNN-based Equalization comparison](https://arxiv.org/html/2609.36103)

**(F) Temuan penting**: **"SNN bisa lakuin temporal computation TANPA butuh recurrent connection"** — ini beda fundamental dari RNN/GRU (yang temporal processing-nya eksplisit lewat recurrent weight). Artinya SNN vs GRU itu BUKAN cuma "dua cara beda buat hal yang sama" — mekanismenya genuinely beda, jadi perbandingan kita VALID secara konsep (bukan membandingkan apel-ke-apel yang identik, tapi membandingkan dua strategi temporal processing yang berbeda — itu memang tujuannya).
Sumber: [Frontiers - SRNN temporal computation](https://www.frontiersin.org/journals/neuroscience/articles/10.3389/fnins.2022.760298/full)

**(F) Pitfall yang diakui lapangan sendiri**: "time-to-first-spike/few-spikes conversion melindungi sparsity SNN, TAPI domain waktu dipakai buat encoding sehingga RNN gak bisa di-convert dengan cara yang sama" — **gak ada metode konversi yang bikin perbandingan SNN-vs-RNN itu 100% apple-to-apple**. Ini bukan kegagalan desain kita, ini keterbatasan yang diakui bidang ini sendiri.

**(F) Metodologi ablation yang benar (dari "Controlled Ablation Study" + paper SNN pruning)**: pendekatan yang direkomendasikan adalah **implementasi tensor-native SATU kodebasae yang bisa toggle tiap confound secara independen** (sparsity on/off, temporal-recurrence on/off sebagai flag di model yang SAMA) — BUKAN 3 kelas model terpisah (`DenseJEPA`/`GRUJepa`/`SpikingJEPA` di kode kita sekarang) yang masing-masing punya implementasi, inisialisasi, dan detail teknis berbeda-beda. **Desain `ablation_gru.py` kita PUNYA risiko confound implementasi** (bukan cuma confound konseptual) karena 3 model beda kelas, bukan 1 model dengan flag yang di-toggle.

**(F) Pitfall umum yang diakui lapangan**: "misalignment antara desain ablation dan sistem original", "batas ablation yang ambigu", "spesifikasi prosedural yang gak lengkap" — checklist buat self-audit desain kita sendiri sebelum dijalankan.

**(F) Kandidat kontrol ketiga yang lebih kuat dari GRU**: **Liquid Time-Constant Networks (LTC)** — continuous-time, ODE-based, **terbukti ngalahin GRU/LSTM 2-2.5x akurasi** di berbagai benchmark (gesture, traffic forecasting). LTC beda dari GRU (gate diskrit/fixed) DAN beda dari SNN (spike diskrit) — LTC punya "time constant yang fluktuasi ikut data", representasi continuous-time yang lebih murni buat isolasi "temporal integration" dibanding GRU yang gate-nya fixed. **Kemungkinan GRU bukan kontrol yang paling kuat/representatif** buat hipotesis "temporal integration doang" — LTC lebih murni secara konsep.
Sumber: [Liquid Time-constant Networks](https://arxiv.org/pdf/2006.04439)

**Implikasi buat rencana ablation (BELUM di-approve, cuma dicatat buat revisi plan nanti)**:
1. Desain ulang jadi SATU kodebase ber-flag (sparsity: on/off, recurrence-type: none/GRU/LIF), bukan 3 kelas model terpisah
2. Pertimbangkan LTC sebagai kontrol temporal-integration yang lebih murni daripada GRU
3. Akui secara eksplisit di laporan: gak ada perbandingan SNN-vs-RNN yang 100% bebas confound di literatur manapun — ini keterbatasan bidang, bukan kegagalan kita

## 11. Deep-Dive: Liquid Time-Constant Networks (LTC)

**(F) Formulasi matematis** — LTC itu continuous-time RNN di mana hidden state dikontrol ODE orde-1: `τ(x,t) dh(t)/dt = -h(t) + f(x(t), h(t), W, b)`. Bagian "liquid"-nya: **time constant τ itu sendiri FUNGSI dari input saat itu** — network otomatis "melambat" atau "mempercepat" dinamikanya tergantung densitas informasi sinyal yang masuk. Ini beda fundamental dari GRU (gate-nya FIXED, gak adaptif ke konten input) dan dari SNN (state-nya diskrit/spike, bukan kontinu).
Sumber: [Liquid Time-constant Networks (arXiv)](https://arxiv.org/pdf/2006.04439)

**(F) Masalah training LTC asli**: ODE-nya **stiff**, butuh numerical solver (variable-step Runge-Kutta atau custom fused Euler) — mahal secara komputasi buat training.

**(F) Solusi praktis — CfC (Closed-form Continuous-time)**: derivasi closed-form approksimasi dari solusi LTC, **TANPA butuh ODE solver sama sekali**. Hasil: **160x lebih cepat training dibanding ODE-RNN, 220x lebih cepat dibanding continuous latent model lain — TANPA kehilangan akurasi**. Ini varian yang harus dipakai buat eksperimen praktis, bukan LTC asli.
Sumber: [Closed-form continuous-time neural networks (Nature)](https://www.nature.com/articles/s42256-022-00556-7)

**(F) TEMUAN PALING PENTING — LTC udah divalidasi di hardware neuromorphic ASLI, bukan simulasi:** Liquid Neural Network diimplementasi di **Intel Loihi-2** (chip fisik, bukan simulasi GPU) — hasil **91.3% akurasi, cuma 213 microjoule per frame**. Ini beda signifikan dari status SNN kita di §9 (ESPP/LPL/EqProp semua masih simulasi GPU, belum ada yang diukur di chip fisik). **LTC punya validasi hardware yang lebih kuat dari pendekatan SNN murni yang kita eksplorasi sejauh ini.**
Sumber: [Exploring Liquid Neural Networks on Loihi-2](https://www.alphaxiv.org/abs/2407.20590)

**(H→lebih kuat setelah verifikasi berulang) Kombinasi LTC + JEPA**: dicek dengan **4 query berbeda framing** (self-supervised contrastive + liquid network, neural ODE + contrastive predictive coding, LTC + world model/planning/RL, dan pencarian langsung "LTC"/"liquid neural network" + "joint embedding predictive architecture"). **Konsisten gak ketemu kombinasi eksplisit di semua 4.** Hasil pencarian terakhir eksplisit bilang: *"JEPA focuses on masked prediction in embedding space... while LTC networks are specialized for time-series prediction... these are two distinct architectures"* — gak pernah digabung di yang terindex.

**Kalibrasi kehati-hatian (tetap, belajar dari §9)**: ini LEBIH KUAT dari klaim gap yang salah sebelumnya (yang cuma berdasar 1 pencarian) karena udah dicoba 4 framing berbeda dengan hasil konsisten — tapi TETAP bukan bukti 100% pasti "belum ada", cuma "gak ketemu di pencarian yang terindex search engine". Bisa jadi ada di paper yang pakai terminologi berbeda, preprint yang belum terindex, atau conference paper yang gak muncul di web search biasa.

**Temuan terkait yang relevan**: **Contrastive Predictive Coding (CPC, van den Oord 2018)** — ini predecessor historis langsung ke JEPA (prediksi representasi masa depan di latent space pakai contrastive loss), dan ada riset umum soal "fusing predictive coding with contrastive learning" buat latent space yang lebih smooth terhadap noise frekuensi tinggi — tapi gak satu pun yang pakai LTC/continuous-time sebagai encoder-nya.
Sumber: [Representation Learning with Contrastive Predictive Coding](https://arxiv.org/pdf/1807.03748)

**LTC juga udah dipakai di real-world time-series applications** (robotics, autonomous driving, multi-agent control, blockage prediction di mmWave link) — jadi bukan cuma akademis murni, udah ada track record aplikasi nyata yang relevan buat kredibilitas substrat ini.
Sumber: [GitHub - raminmh/liquid_time_constant_networks](https://github.com/raminmh/liquid_time_constant_networks), [Blockage Prediction mmWave LTC](https://arxiv.org/pdf/2306.04997)

**Revisi kandidat kontrol buat ablation (masih BELUM di-approve, cuma update catatan)**: LTC/CfC kemungkinan kontrol yang lebih kuat & lebih relevan dari GRU buat isolasi "temporal integration" — selain lebih murni secara matematis (continuous-time asli vs GRU yang gate diskrit), LTC JUGA punya jalur validasi hardware neuromorphic yang GRU gak punya. Kalau mau nguji hipotesis "temporal integration vs sparsity", LTC/CfC > GRU sebagai kontrol.

## 12. Kemampuan Von Neumann Secara Teoretis — Kenapa Dia "Menang" (Computability Theory)

Pertanyaan yang lebih dalam dari sebelumnya: bukan "apa kelemahan von Neumann", tapi **apa yang von Neumann BISA lakuin secara teoretis yang bikin dia 'menang' sejak awal** — ini masuk computability theory, bukan cuma engineering.

**(F) Church-Turing Thesis**: sistem yang **Turing-complete** bisa mensimulasikan APAPUN yang bisa dihitung sistem Turing-complete lain — cuma beda EFISIENSI (cepat/lambat), BUKAN beda KEMAMPUAN. "Instruction set yang lebih kaya itu beli efisiensi dan ekspresivitas, BUKAN tambahan computational power." Ini kunci: von Neumann menang bukan karena dia arsitektur paling efisien, tapi karena dia Turing-complete DAN bisa dibangun secara fisik praktis.
Sumber: [Von Neumann Thought Turing's Universal Machine was 'Simple and Neat'](https://cacm.acm.org/opinion/von-neumann-thought-turings-universal-machine-was-simple-and-neat/)

**(F) Sejarah krusial — kenapa von Neumann yang menang, bukan yang lain**: **Turing** nemuin BAHWA komputasi universal itu mungkin (secara matematis). **Von Neumann** nemuin CARA BANGUN mesin yang menjalankannya secara fisik praktis (control unit, program counter, instruction set, addressable memory, stored-program dengan ALU/IO). **"Von Neumann architecture membuat hasil universalitas itu menjadi FISIK."** Dia gak menang karena idenya paling pintar — dia menang karena dia yang pertama bikin universalitas Turing itu BISA DIBANGUN dan DIJUAL secara massal.

**(F) Neuromorphic computing juga Turing-complete** (dibuktikan formal, 2021/2022) — artinya **secara KEMAMPUAN komputasi, neuromorphic SAMA PERSIS dengan von Neumann, gak lebih gak kurang.** Semua substrat yang kita bahas (neuromorphic, photonic, memristor — asalkan digital/diskrit) itu Turing-complete juga, cuma beda efisiensi buat workload tertentu. **Mereka gak "mengalahkan" von Neumann secara kemampuan — cuma lebih efisien di domain sempit.**
Sumber: [Neuromorphic Computing is Turing-Complete](https://arxiv.org/pdf/2104.13983)

**(F) TEMUAN PALING DALAM — satu-satunya celah teoretis buat ngelampaui von Neumann**: **analog neural network dengan bobot real-valued (presisi tak terbatas) secara TEORI bisa capai "super-Turing" compute power** — kemampuan komputasi yang BENERAN ngelampaui apa yang bisa dihitung Turing machine manapun (termasuk von Neumann digital). Ini riset nyata (teori analog computation, Hava Siegelmann dkk), bukan fantasi.

**TAPI — ini yang bikin nyambung balik ke §6**: **"keunggulan analog computation itu bergantung ke PRESISI TAK TERBATAS, yang secara praktis gak mungkin."** Dan inget temuan kita sendiri di §6: **memristor cuma punya 4-6 bit effective precision** karena nonlinearity & noise fisik — **JAUH dari presisi tak terbatas yang dibutuhkan buat beneran capai super-Turing.** Jadi celah teoretis buat ngelampaui von Neumann itu ADA DI ATAS KERTAS, tapi **tertutup secara fisik** oleh masalah precision/noise yang sama yang udah kita identifikasi sebagai kelemahan reliability substrat eksperimental.
Sumber: [Analog Neural Networks: Computational Power](https://groups.cs.umass.edu/binds/analog-neural-networks-analog-computational-power/)

**(F) Hypercomputation secara umum dianggap gak mungkin secara fisik**: butuh komputasi tak terbatas (explicit/implicit), dan **bahkan kalau komputasi tak terbatas bisa direalisasi, Halting Problem tetap gak terpecahkan** — ada batas yang lebih dalam dari sekadar "kurang presisi/kurang cepat".
Sumber: [Hypercomputation and the Physical Church-Turing Thesis](https://www.journals.uchicago.edu/doi/10.1093/bjps/54.2.181)

**Sintesis final**: von Neumann "menang" bukan karena superior secara mentah — dia menang karena (1) Turing-complete (kemampuan maksimal yang bisa dibuktikan ADA), (2) yang PERTAMA bikin itu jadi fisik/praktis. Semua alternatif yang kita bahas (neuromorphic, photonic, memristor digital) **gak mengalahkan von Neumann secara kemampuan** — mereka Turing-complete juga, cuma menang efisiensi di domain sempit. **Satu-satunya jalan teoretis buat BENERAN ngelampaui (bukan cuma lebih efisien dari) von Neumann adalah analog computation presisi-tak-terbatas — dan itu justru diblokir oleh masalah precision/noise fisik yang persis sama dengan kelemahan reliability yang udah kita temukan di §6.** Dua temuan yang kelihatannya beda topik (computability theory vs device engineering) ternyata nyambung di titik yang sama.

## 13. Deep-Dive: Siegelmann Super-Turing — Detail Matematis & Kritiknya

**(F) Model ARNN (Siegelmann & Sontag, 1993/1994)**: Analog Recurrent Neural Network — rangkaian finite neuron kontinu, recurrent (ada loop/cycle), menghasilkan dynamical system dengan long-term memory.

**(F) Hasil matematis PRESISI (ini intinya, bukan cuma "analog lebih kuat" secara umum)**: Siegelmann & Sontag buktikan **polynomial-time pada linear saturated RNN SAMA DENGAN**:
- **Turing machine biasa** — KALAU bobot network-nya **rational** (pecahan biasa)
- **Nonuniform circuit (kelas kompleksitas P/poly)** — KALAU bobot network-nya **real** (termasuk irrational, presisi tak terbatas)

**P/poly itu KELAS KOMPLEKSITAS YANG PROVEN lebih kuat dari apa pun yang bisa dihitung Turing machine dalam waktu polynomial** — bisa "encode" informasi tak terbatas lewat nilai bobot presisi-tak-terbatas, equivalent ke Turing machine yang dapat "advice" (bocoran informasi tambahan) tak terbatas. Jadi: **garis pembedanya BUKAN "analog vs digital" secara umum — garis pembedanya persis di RATIONAL vs REAL weight.**
Sumber: [Analog computation via neural networks (Siegelmann & Sontag)](http://www.sontaglab.org/FTPDIR/siegelmann_sontag_tcs1994.pdf)

**(F) "Thesis of Analog Computation"** — Siegelmann & Sontag bikin thesis paralel ke Church-Turing TAPI buat realm analog: "gak ada analog device yang reasonable bisa lebih kuat dari ARNN order pertama." Ini sendiri masih THESIS (belum proven, kayak Church-Turing juga cuma thesis bukan teorema), bukan fakta final.

**(F) Nuansa penting — gak unlimited**: "analog network gak infinitely powerful... resource constraint tetap bikin reduksi computational power yang nontrivial" — super-Turing power-nya BERSYARAT dan terbatas, bukan kekuatan tanpa batas.

**(F) Istilah "super-Turing" (Siegelmann) secara eksplisit BEDA dari "hypercomputation" generik**: super-Turing secara eksplisit DIKLAIM bisa direalisasi secara fisik (bukan cuma thought experiment) — ini bikin klaimnya LEBIH BERANI dan LEBIH BISA DIUJI dibanding hypercomputation murni yang sering cuma eksperimen pikiran (infinite time, dll).

**(F) KRITIK paling tajam — dan ini yang nutup celahnya**: ada 2 kategori argumen formal lawan hypercomputation/super-Turing:
1. **"Argument by Infinity"** — model yang butuh infinity tertanam (termasuk presisi real-number tak terbatas) gak bisa direalisasi fisik
2. **"Argument by Physical Vagueness"** — klaim "analog = akses ke real number penuh" itu sendiri gak jelas secara fisik apa artinya

**Poin paling dalam**: sistem fisik yang KONTINU (bukan diskrit) punya variabel yang nilainya real number ARBITRARY — tapi Turing machine cuma bisa akses **computable real number** (himpunan yang JAUH lebih kecil dari semua real number — ada real number yang secara matematis TERBUKTI gak bisa didekripsi/dihitung oleh program apa pun). Jadi pertanyaannya bukan "apakah alam punya nilai real presisi tak terbatas" (mungkin iya, mungkin tidak) — pertanyaannya **"apakah ADA PROSES FISIK yang bisa BENERAN MENGAKSES/MEMANFAATKAN presisi tak terbatas itu buat komputasi"** — dan ini balik lagi ke noise/thermal fluctuation/quantum uncertainty yang SECARA FUNDAMENTAL (bukan cuma "teknologi belum maju") mungkin mencegah pengukuran/kontrol presisi tak terbatas di sistem fisik mana pun.
Sumber: [Hypercomputation and the Physical Church-Turing Thesis](https://www.journals.uchicago.edu/doi/10.1093/bjps/54.2.181), [Practical intractability: a critique of hypercomputation](https://arxiv.org/pdf/1210.3304)

**Revisi sintesis (lebih presisi dari §12)**: celah super-Turing itu BUKAN "memristor belum cukup bagus hari ini" (masalah engineering sementara) — kemungkinan itu **masalah fisika fundamental** (noise termal, prinsip ketidakpastian) yang mencegah SISTEM FISIK APA PUN (bukan cuma memristor, bukan cuma hari ini) mengakses presisi real-number tak terbatas yang dibutuhkan buat super-Turing jadi nyata. Kalau ini benar, celah teoretis Siegelmann itu **secara permanen gak bisa dicapai secara fisik** — bukan tinggal nunggu material lebih baik.

## 14. AUDIT KRITIS OPUS (2026-10-02) — Ini Menggantikan Kesimpulan Lama di Atas

Spawned Opus (model reasoning terpisah) khusus buat baca penuh §1-13 dan audit jujur. **Ini bukan riset tambahan biasa — ini koreksi besar ke keseluruhan arah**, dan user udah approve rekomendasi arahnya (lihat §15). Detail lengkap di bawah, dipisah biar jelas mana yang FAKTA KOREKSI (harus dipegang) vs REKOMENDASI (sudah diputuskan).

### 14a. Temuan Paling Kritis — Eksperimen §5 Kemungkinan Invalid dari Akarnya

**(F, koreksi keras ke §5)** Loss Spiking JEPA = 0.00000 persis di SEMUA 10 seed, variansi NOL. **Ini signature representation collapse** (encoder keluarin output hampir konstan, predictor jadi trivial bisa "prediksi", loss→0 TANPA model belajar apa pun) — bukan signature kemenangan. Belum ada satu pun diagnostic collapse yang dijalanin (effective rank/singular value spectrum, variance per dimensi, trivial-constant baseline, VICReg-style check).

**(F, lebih fundamental)** Loss JEPA mentah **BUKAN metrik performa yang valid buat dibandingin antar-arsitektur** — diukur di ruang representasi yang beda skala/norma per model (spiking output rate-coded/bounded/sparse → norma lebih kecil → MSE mekanis lebih kecil, terlepas kualitas). **Selisih 0.00011 vs 0.00000 bisa 100% artefak skala**, bukan bukti apa pun. Dependent variable yang valid: downstream linear-probe (freeze encoder, latih readout di task terdefinisi, lapor akurasi/R² di held-out set) + normalized-loss kalau tetap mau lapor loss.

**Implikasi tegas**: revisi metodologi di §10-11 (satu kodebase ber-flag, kontrol LTC vs GRU) itu **membenerkan independent variable sementara dependent variable-nya masih rusak**. Harus dibenerin DULU sebelum ablation apa pun berarti.

**ATURAN KERAS: angka di §5 TIDAK BOLEH dilaporkan ke siapa pun (dosen/writeup/"hasil preliminary") sampai collapse diagnostic + downstream probe dijalanin.**

### 14b. 3 Confound Tambahan yang Belum Tercatat

1. **Compute/depth mismatch** — model spiking di-unroll T timestep, dapet T× computation & effective depth vs MLP dense yang 1 forward pass. Param-matching (42,496=42,496) itu kontrol yang SALAH; yang perlu di-match: total MAC/FLOP + effective depth.
2. **Kemungkinan data leakage** — kalau sinusoid sintetis deterministik & fase bisa diinfer persis dari window konteks, prediksi jadi trivially solvable.
3. **Task saturated, nggak ada headroom** — Dense JUGA dapet ~0.00011 (praktis nol). Task "hard" kita gak hard. Ranking antar-arsitektur ditentukan floor numerik, bukan kapabilitas.

### 14c. Kesalahan Logika Antar-Section (Dikoreksi)

- **§1→§2 Landauer salah sasaran**: energi chip modern **didominasi data movement** (CV²f di wire/DRAM), BUKAN bit erasure. Angka canonical (Horowitz, ISSCC 2014): 32-bit add ≈0.1pJ vs 32-bit DRAM access ≈640pJ — beda ~4 orde besaran. Gap 10⁹× ke Landauer itu benar secara angka tapi **gak relevan secara kausal** — nutup gap itu BUKAN yang nahan kita. Reversible computing jadi terlalu diprioritaskan di §2 akibatnya; plus ada trade-off energi-vs-kecepatan yang hilang (dissipation skala ~1/τ, reversible cuma mendekati nol kalau dijalanin LAMBAT — fatal buat AI yang throughput-bound).
- **§6↔§7 kontradiksi nggak kesadar**: §6 bilang overhead ADC "bottleneck tersembunyi", §7 rekomendasiin ECC+wear-leveling+redundansi — tapi **biaya energi/area obat itu belum pernah dihitung** apakah masih lebih hemat dari von Neumann setelahnya. Pertanyaan yang bisa dijawab di atas kertas (energy accounting), belum dijawab.
- **§7 failure model von Neumann 1956 nggak cocok**: teorema itu asumsiin error INDEPENDEN & TRANSIEN. Failure mode memristor yang kita dokumentasiin sendiri (drift/suhu = terkorelasi across array; endurance/wear = progresif permanen; stuck cells = permanen) **BEDA KELAS** — majority voting gak nolong buat error terkorelasi atau degradasi permanen. Kasus Intel 13th/14th Gen yang kita pakai contoh itu justru **systematic correlated degradation** — persis kelas yang teorema itu TIDAK cover. Pelajaran "Intel gagal karena nggak ada redundansi" kurang tepat.
- **Threshold "<1/6" butuh kualifikasi**: von Neumann 1956 sendiri dapet ≈0.0107 buat NAND multiplexing-nya; 1/6 itu hasil belakangan (Hajek & Weller 1991) yang **tergantung fan-in gate** (3-input). Nulis "<1/6" tanpa kualifikasi = overclaim halus.
- **§9/§11→§5 misalignment terbesar**: gap nyata yang kita identifikasi sendiri adalah scaling+validasi hardware fisik, tapi program kerja kita adalah eksperimen toy disimulasi GPU — ujung spektrum BERLAWANAN. Simulasi GPU **secara prinsip gak bisa nguji klaim hardware** (sparsity gak ngasih speedup di GPU, spike_rate bukan proxy energi tanpa model energi). Jembatan yang hilang dan murah ditambahin: hitung SynOps × energi-per-SOP publikasi Loihi 2.
- **§12-13 cabang terminal**: computability class (Turing-complete, P/poly) itu **frame yang salah** buat pertanyaan "substrat buat AI lebih maju" — yang nentuin AI itu konstanta energi/waktu per komputasi + learnability/inductive bias, BUKAN apa yang computable. Turing-equivalence literally gak ngomong apa-apa soal itu.

### 14d. Klaim yang Dikoreksi/Terlalu Kuat

- **"Neuromorphic Turing-complete = sama persis dengan von Neumann"** — category error: Turing-completeness asumsiin memory TAK TERBATAS. Chip fisik (Loihi 2 maupun CPU) itu finite-state machine, bukan Turing machine beneran. Kesimpulannya gak salah tapi jauh lebih lemah/kurang bermakna dari yang dibaca — ini statement tentang model IDEAL, bukan chip nyata. Juga: hasil ini sendiri nyaris formalitas (apa pun dengan state cukup+nonlinearitas+unbounded memory itu Turing-complete), bukan temuan dalam.
- **"4-6 bit precision membunuh memristor"** — overclaim buat INFERENCE (4-bit itu sekarang NORMAL buat inference model besar, lihat literatur quantization/QAT). Buat TRAINING objeksinya lebih kuat (butuh presisi update tinggi). Catatan lama nyampur dua kasus. Solusi lapangan yang hilang: hardware-aware training (latih dengan noise device disimulasikan di loop) — lihat PCM 64-core IBM (Le Gallo et al., Nature Electronics 2023).
- **"Neuromorphic gak punya standar benchmark"** — STALE. **NeuroBench** (Yik et al., Nature Communications 2025) udah ada, persis usaha komunitas buat ini.
- **Framing P/poly di §13 lemah, dan koreksinya bikin argumen LEBIH kuat**: "power" P/poly seluruhnya ada di advice string yang DIBERIKAN, bukan dihitung — kalau network dikasih real number uncomputable sebagai bobot, dia bisa "baca" info uncomputable dari situ, tapi komputasinya sendiri gak ngelakuin keajaiban, keajaibannya diselundupkan lewat initial condition (question-begging: butuh udah PUNYA info uncomputable buat set bobotnya, gak ada prosedur fisik yang "nemu" bobot itu tanpa jadi hypercomputational sendiri).
- **Argumen fisika §13 udah ada TEOREMANYA, bukan pertanyaan terbuka**: **Maass & Orponen (1998)** — analog NN dengan analog noise cuma bisa kenalin SUBCLASS dari regular language (definite languages) — **COLLAPSE ke bawah Turing**, bukan cuma kehilangan keunggulan. **Maass & Sontag (1999)** — analog neural net dengan noise Gaussian/distribusi umum lain TIDAK BISA kenalin regular language arbitrary. **Ini dijawab formal 27 tahun lalu**, jawabannya lebih keras dari yang dicatat sebelumnya ("mungkin diblokir fisika" → harusnya "sudah dibuktikan collapse ke bawah Turing").

### 14e. Blind Spot — Konsep/Paper yang Sama Sekali Absen (diurut dampak)

1. **State Space Models (S4/S5/Mamba/Liquid-S4)** — lubang TERBESAR. Liquid-S4 (penulis LTC/CfC yang sama) udah eksplisit nyambungin LTC ke S4. Premis §11 ("LTC kontrol paling murni") ketinggalan satu generasi — kandidat kontrol yang benar: S4/S5/Mamba atau Liquid-S4, bukan GRU/CfC polos.
2. **Representation collapse diagnostics**: VICReg, Barlow Twins, dimensional collapse (Jing et al. 2022), "A Cookbook of Self-Supervised Learning" — kita ngerjain joint-embedding SSL TANPA satu pun alat diagnostik dari literatur joint-embedding SSL sendiri.
3. **Hinton: Forward-Forward (2022) + "Mortal Computation"** — Mortal Computation (gagasan: buat efisiensi energi ekstrem, pengetahuan terikat ke keanehan analog fisik chip, gak bisa dicopy antar instance) itu **PERSIS tesis riset kita**, dirumuskan lebih tajam, gak pernah muncul.
4. **Physical neural networks / Physics-Aware Training**: Wright et al. (Nature 2022), Momeni et al. (Nature 2024/2025) — jembatan hilang antara §6 (device gak presisi) dan §9 (butuh local rule): jawabannya bisa hybrid in-situ forward/digital backward, bukan cuma local rule.
5. **Energy-based models, Deep Equilibrium Models, modern Hopfield** — JEPA itu SATU instansiasi dari program Energy-Based Model LeCun; EBM adalah framing yang nyambung ke hardware (relaksasi ke fixed-point, bukan forward-pass layer-by-layer), JEPA cuma satu pilihan arsitektur di dalamnya. "Ambil anaknya, tinggalin bapaknya."
6. **Dreamer v3 / TD-MPC2** — latent world model yang BENERAN nyelesaiin task lintas-domain. Baseline empiris yang relevan buat JEPA, absen total.
7. **"The Hardware Lottery" (Sara Hooker, 2020/CACM 2021)** — tesis "von Neumann menang karena pertama bikin universalitas jadi fisik/bisa dijual massal" yang kita "temuin sendiri" di §12 itu **literally argumen paper ini** — kita reinvent tanpa tau papernya ada.
8. **Photonics sebagai INTERCONNECT (bukan compute)** — di 2026, silicon photonics menang sebagai co-packaged optics/optical link antar-chip, nyerang BENERAN bottleneck data-movement dari §1. Kita salah naruh kolom (cuma di "compute" §2), lewatin tesis yang jauh lebih well-supported.
9. **Thermodynamic/probabilistic computing (p-bits)** — Camsari/Datta, Coles et al. "Thermodynamic AI" (2023), Extropic/Normal Computing. **Dual langsung dari §13**: daripada noise MEMBLOKIR keunggulan analog, PAKAI noise termal SEBAGAI resource komputasi (cocok buat EBM/diffusion/Bayesian/MCMC). **"Bet substrat terbaik di file ini yang nggak ada di file ini."**
10. **Active Inference/FEP** — disebut sekali di §3, nggak pernah dipakai lagi. Thread menggantung, perlu diputuskan dipakai atau dibuang eksplisit.

### 14f. Dua Cacat Fundamental (Flagged Jujur oleh Opus)

**Cacat 1 (struktural, fixable)**: komponen empiris kita **secara prinsip gak bisa nguji hipotesis yang jadi alasan dia dibangun** — simulasi GPU gak bisa ngomong soal energi/sparsity-benefit/hardware-fit, karena sparsity gak ngasih speedup di GPU dan spike_rate bukan satuan energi tanpa model energi. Perbaikan murah: SynOps energy model + downstream probe + data event nyata (DVS128 Gesture/N-MNIST/SHD).

**Cacat 2 (strategis, lebih penting)**: file ini sebenarnya **dua proyek yang disatuin aspirasi, bukan mekanisme** — thread hardware (§1-2,6-7,12-13) dan thread AI-arsitektur (§3,5,9-11) gak pernah ketemu di SATU klaim yang bisa diuji. Satu-satunya jembatan ("spiking sparsity hemat energi di neuromorphic") belum pernah diukur. **Asumsi yang gak pernah diperiksa 14 section**: bahwa "komputer gen baru buat AI lebih maju" = SUBSTRAT baru. Hipotesis alternatif (constraint-nya objective/data/algoritma, atau integrasi system-level/interconnect — BUKAN fisika device) gak pernah dipertimbangkan, padahal 2 koreksi terkuat audit ini (energi = data movement bukan bit-erasure; photonics menang sebagai interconnect bukan compute) **dua-duanya nunjuk ke hipotesis alternatif itu**.

## 15. ARAH RESMI (DISETUJUI USER, 2026-10-02) — Menjadi Auditor Metodologi, Bukan Lanjut Arsitektur

**Keputusan**: user approve rekomendasi Opus sebagai arah resmi, mengganti rencana ablation GRU/LTC lama.

**Objek riset**: masalah PENGUKURAN, bukan arsitektur. Bangun protokol evaluasi hardware-grounded + collapse-audited buat self-supervised predictive learning di substrat event-driven, lalu pakai protokol itu buat re-evaluasi klaim yang udah ada di literatur (ESPP/EchoSpike, LPL, PC-SNN) di bawah satu protokol seragam.

**3 Deliverable yang direncanakan (BELUM dieksekusi, nunggu plan detail+approval per aturan kerja)**:
1. **Protokol**: downstream linear-probe sebagai dependent variable utama + collapse diagnostics wajib (effective rank, per-dim variance, trivial-constant baseline) + compute/depth accounting (MAC+effective depth) + model energi berbasis SynOps (pakai angka energi-per-SOP Loihi 2 publikasi).
2. **Instrumen**: satu kodebase ber-flag (sparsity on/off, recurrence-type none/gated-discrete/LIF/continuous-time, learning-rule surrogate-BPTT/local) — kontrol temporal-integration pakai **S4/S5 atau Liquid-S4** (bukan GRU/CfC polos). Dijalanin di data event NYATA gratis: DVS128 Gesture, N-MNIST, SHD/SSC (Heidelberg) — bukan data sintetis yang saturated.
3. **Audit**: jalanin klaim ESPP/LPL/PC-SNN lewat protokol itu. Kalau mereka juga lapor loss (bukan probe) atau gak lapor collapse diagnostic — itu TEMUAN. Taruh hasil di bawah **NeuroBench** (komunitas, bukan ruang kosong).

**Kenapa arah ini** (vs alternatif): lanjut ablation GRU/LTC = benerin variable yang salah (lihat §14a). Ngejar gap LTC+JEPA = "novelty-by-combination tanpa hipotesis mekanisme" — resep paper yang gak disitasi siapa-siapa. Arah ini **menyerap** kerja ablation sebagai satu komponen dengan metrik yang bener, dan cocok sama comparative advantage yang KEBUKTI sepanjang sesi ini: adversarial self-auditing (nemu confound sendiri di §5, nurunin confidence sendiri ke 0% di §8, batalin gap claim sendiri di §9) — bukan arsitektur baru. Precedent genre ini: Dacrema et al. "Are We Really Making Much Progress?" (RecSys 2019), Henderson et al. "Deep RL That Matters", Musgrave et al. "A Metric Learning Reality Check" — semua highly-cited.

**Yang di-drop/ditutup**: §12-13 (computability theory) ditutup sebagai esai selesai, bukan dilanjutin — frame-nya salah buat pertanyaan AI. Kolom quantum di §2 dibuang total. Data sintetis sinusoid ditinggalkan (saturated), pindah ke data event nyata. Reversible computing diturunin dari "kandidat substrat" jadi "catatan fisika" (fondasi Landauer-nya salah sasaran). Thermodynamic/p-bit computing jadi thread BACA/position-paper, bukan thread eksekusi (lebih hardware-gated dari neuromorphic).

## 16. FASE A DIEKSEKUSI (2026-10-02) — Klaim §5 RESMI DITARIK

User approve plan Fase A (lihat §15). Dieksekusi: `real_data.py` (load N-MNIST via `tonic`, 10,000 sampel test split, ToFrame 20 time-bin), `collapse_diagnostics.py` (effective rank/SVD + per-dim variance), `downstream_probe.py` (linear probe + trivial-constant baseline), `run_phase_a.py` (orkestrasi). Hipotesis dinyatakan SEBELUM run (lihat §15/15-plan di chat) — gak diubah setelah liat angka.

**(E) Hasil (N-MNIST test split, 7000 train/3000 eval, 300 step self-supervised, probe 300 step):**

| Model | Downstream test acc | Effective rank | Top-1 SV variance frac | % dim near-zero variance | Verdict collapse diagnostic |
|---|---|---|---|---|---|
| **Dense MLP JEPA** | **67.80%** | 21.87/64 | 0.522 | 0.00% | Gak ada sinyal collapse |
| **Spiking (LIF) JEPA** | **14.33%** | 4.16/64 | 0.416 | **92.19%** | **COLLAPSE** |
| Trivial-constant (kontrol wajib) | 12.40% | 1.00/64 | 1.000 | 100.00% | Collapse (by definition) |

**(F→E, kesimpulan terkonfirmasi langsung, bukan dugaan lagi)**: SpikingJEPA (14.33%) **TIDAK ngungguli trivial-constant baseline (12.40%) secara meaningful** — selisihnya di dalam margin yang udah ditentuin sebelum run (5%). Collapse diagnostic independen (92.19% dimensi punya variance mendekati nol) **konsisten dan mengkonfirmasi** verdict dari downstream probe. **Dugaan Opus TERBUKTI BENAR: klaim "Spiking JEPA menang" di §5 (data sintetis) itu representation collapse, bukan temuan nyata. Resmi ditarik — JANGAN dikutip lagi sebagai temuan dalam bentuk apa pun.**

**Yang justru REAL**: Dense MLP JEPA dapet 67.80% test accuracy di task klasifikasi digit 10-kelas (random chance 10%), effective rank sehat (21.87/64), nol dimensi collapse. **Self-supervised JEPA biasa (dense) beneran belajar representasi yang berguna di data event nyata** — ini hasil yang solid, encoder dense-nya bukan collapse.

**Catatan jujur soal KENAPA spiking collapse (belum diselidiki, bukan kesimpulan final)**: avg spike_rate selama training cuma 0.0556 (5.56% neuron aktif) — ini mungkin tanda neuron "mati" (dead neuron problem, umum di SNN kalau threshold/beta gak di-tuning), bukan properti fundamental spiking neural network secara umum. Belum dicoba tuning hyperparameter (beta, threshold, learning rate) buat spiking model — jadi hasil ini bilang "implementasi SpikingJEPA KITA collapse di setup ini", BUKAN "spiking neural network secara umum gak bisa buat JEPA". Bedanya penting buat gak overclaim ke arah yang salah juga.

**Status Fase A (versi awal)**: selesai, hipotesis terkonfirmasi. User minta perluasan — 1 setting hyperparameter dianggap belum "lebar dan adil" buat vonis final (fair concern: mungkin itu cuma bug tuning, bukan limitasi arsitektur).

## 17. Grid Sweep (2026-10-02) — Perluasan Fase A, Konfirmasi Lebih Kuat

**(F, verifikasi sebelum eksekusi)** GPU/CPU dicek dulu beneran (bukan nebak): 1 run 300 step = 13.47 detik, GPU 49°C→53°C, power 16W→19W — nol risiko thermal buat 45 run (~12-15 menit total). Juga ketangkep blind spot sendiri sebelum run: **cherry-picking risk** (lapor kombinasi terbaik doang = nipu diri sendiri) — solusinya lapor SEMUA 45 hasil + tuntut 3/3 seed konsisten, bukan "pernah berhasil sekali".

**(E) Desain**: `grid_sweep.py` — Spiking: 4 beta [0.5,0.7,0.9,0.95] × 3 LR [1e-3,3e-3,1e-2] × 3 seed = 36 run. Dense: 3 LR yang sama × 3 seed = 9 run (simetris, dikasih kesempatan sama walau udah kerja).

**(E) HASIL — 0/12 kombinasi spiking RELIABLE, di SEMUA 36 run:**

| Kombinasi (beta, lr) | Acc (mean±std, 3 seed) | Collapsed | Beats trivial |
|---|---|---|---|
| Semua 8 kombinasi beta≤0.9, lr apapun kecuali satu | 0.1240±0.0000 (persis = trivial) | 3/3 | 0/3 |
| beta=0.9, lr=0.001 | 0.1314±0.0129 | 3/3 | 0/3 |
| beta=0.95, lr=0.003 (terbaik dari 12) | 0.1372±0.0193 | 3/3 | 0/3 |
| **Semua 12 kombinasi** | **rata-rata ~0.126** | **36/36 run** | **0/36 run** |

**Dense (kontrol simetris)**: lr=0.001 → 0.6907±0.0108 (RELIABLE, 0/3 collapsed); lr=0.003 → 0.6299±0.0337 (RELIABLE); lr=0.01 → 0.4283±0.0887 (1/3 collapsed, LR kegedean). **2/3 kombinasi dense reliable** — dense gak butuh tuning aneh buat kerja, langsung jalan di 2 dari 3 LR wajar.

**(F→E, kesimpulan terkonfirmasi lebih kuat dari §16)**: **SpikingJEPA (implementasi kita, arsitektur LIF 3-layer ini) GAGAL KONSISTEN di SEMUA 12 kombinasi hyperparameter yang genuinely masuk akal dicoba (rentang beta dari "cepat lupa" sampai "sangat leaky", rentang LR 10x).** Ini BUKAN lagi "1 setting kebetulan buruk" — ini kegagalan yang robust terhadap variasi hyperparameter. §16 (dugaan collapse) dikonfirmasi lebih kuat, bukan direvisi. **Klaim "spiking menang" TETAP ditarik, dan sekarang dengan keyakinan jauh lebih tinggi** (45 run, bukan 1 run; 3 seed per kombinasi, bukan 1; grid hyperparameter penuh, bukan 1 titik).

**Catatan jujur yang tetap berlaku**: ini kesimpulan soal **implementasi SpikingJEPA KITA** (snnTorch LIF 3-layer, arsitektur spesifik ini) di **task spesifik ini** (N-MNIST digit classification via JEPA predictive objective). BUKAN klaim umum "spiking neural network gak bisa dipakai buat predictive self-supervised learning" — ESPP/EchoSpike (§9) buktiin itu BISA jalan dengan desain yang beda (local learning rule, bukan surrogate-BPTT; arsitektur beda). Confound training-method (surrogate gradient vs native event-driven learning, dicatat di §6) masih belum terpisah — mungkin itu akar masalahnya, bukan LIF/beta-nya.

## 18. Next Step (BELUM diputuskan, nunggu user)

Pilihan yang masih terbuka: (a) lanjut ke Fase B dengan kontrol S4/Mamba + dataset lain, anggap urusan spiking-collapse-di-setup-ini udah terjawab cukup kuat; (b) coba desain ulang training method (bukan cuma hyperparameter) buat spiking — misal coba local learning rule ala ESPP sebagai pembanding, biar confound training-method juga keabisan; (c) fokus total ke Dense JEPA yang kebukti solid, lanjut ke Deliverable #3 (audit klaim ESPP/LPL/PC-SNN pakai protokol yang sama persis). Tetap nunggu approval eksplisit sebelum eksekusi apa pun, sesuai `feedback_research_no_premature_execution`.

## 19. Fact-Check Saran Eksternal (Gemini, 2026-10-02) — SVD/Probe Dianggap Bottleneck

User dapet saran dari Gemini (AI lain) bilang grid sweep 45-kombinasi kita "menyiksa GPU" dan "membuang waktu berminggu-minggu" karena SVD (buat effective rank) + 300-step probe di tiap kombinasi, dengan 3 rekomendasi optimasi: (1) two-stage sweep (filter murah dulu pakai proxy 20-30 step, baru full-run di top-3/5), (2) sub-sampling 1024-2048 titik buat SVD, (3) caching embedding offline biar forward pass sekali doang.

**(E) Verifikasi langsung, bukan diterima mentah**: diukur SVD asli di skala kita (matrix 7000×64, 45x berturutan) = **0.16 detik TOTAL**, 3.568ms per call. Dibandingin total runtime grid sweep (~10-15 menit, didominasi training loop 300-step × 45 run), SVD itu **<0.03% dari total waktu**. Klaim "berminggu-minggu" **gak match realita di skala kita sama sekali**.

**Analisis tiap poin:**
1. **Argumen kompleksitas Big-O (poin 1 Gemini)** — benar secara matematika murni, tapi menyesatkan secara praktik: notasi O(min(mn²,m²n)) kedengaran serius tapi `n` (dimensi embedding) kita cuma 64, jadi secara ABSOLUT itu kecil banget. Pola kesalahan ini **persis** sama kayak yang Opus tangkep soal Landauer limit di §14c ("benar secara angka tapi gak relevan secara kausal") — notasi/angka yang benar tapi disalahartikan relevansinya ke konteks skala nyata.
2. **Two-stage sweep (proxy murah + filter)** — metodologi VALID secara umum (successive halving itu teknik nyata), **tapi riskan spesifik buat pertanyaan riset kita**: collapse sering muncul BERTAHAP (EMA target drift, predictor "nyerah" makin lama training) — proxy 20-30 step bisa gak collapse PADAHAL collapse di step 300, nyembunyiin fenomena yang justru mau dideteksi. Ditolak buat riset ini, bukan karena metodenya salah secara umum, tapi karena gak cocok sama pertanyaan spesifik kita.
3. **Sub-sampling SVD** — valid secara teknik, tapi nyelesain masalah yang gak ada (liat poin verifikasi di atas).
4. **Caching embedding offline** — satu-satunya poin yang genuinely actionable secara umum, tapi **kode kita (`grid_sweep.py`) UDAH cuma panggil `extract_embeddings()` sekali per kombinasi** — saran ini nembak masalah yang udah gak ada.

**Kesimpulan**: saran Gemini **tidak diterapkan** — diagnosa akar masalahnya salah (SVD bukan bottleneck di skala kita), dan satu rekomendasinya (two-stage filtering) berisiko ngerusak validitas hasil yang udah kita punya kalau diterapkan retroaktif. Grid sweep 45-kombinasi yang udah selesai dijalanin (§17) tetap valid apa adanya.

**Pelajaran metodologis buat dicatat**: saran dari AI/sumber eksternal lain harus diverifikasi terhadap angka nyata sebelum diterapkan — sama persis prinsip yang udah dipegang sepanjang riset ini (cross-check klaim "gap" lewat literatur di §9/§11, verifikasi angka Opus lewat eksekusi nyata di §16-17). "Kedengaran masuk akal secara teori" bukan alasan cukup buat ngubah metodologi yang udah jalan dan ketahanannya udah terbukti via eksekusi nyata.

**(E) Addendum — "KRITIK GEMINI BATCH 2" ternyata identik byte-per-byte sama kritik batch 1** (file `1. Cacat Metodologi dan Desain E...` vs `KRITIK GEMINI BATCH 2` di Downloads, 2026-10-02) — minta Gemini audit ulang file yang SAMA tanpa konten baru cuma ngasih output yang SAMA, bukan analisis independen baru tiap kali diminta. **Pelajaran tambahan**: kalau mau kritik eksternal yang genuinely baru, input-nya harus beda (pertanyaan lebih spesifik, atau dokumen yang udah di-update) — minta "audit ulang" ke hal yang sama persis itu gak produktif.

## 20. LAMPIRAN: Semua Data Hasil Test (Kompilasi Bersih, Tanpa Narasi)

Semua angka empiris (E) yang udah dihasilkan sepanjang project ini, satu tempat, buat referensi cepat — detail/diskusi lengkap ada di section masing-masing di atas.

### 20a. Eksperimen Sintetis (data sinusoid buatan, ditinggalkan per §15/§16 — SUDAH TIDAK DIPAKAI, cuma arsip)

| Script | Task | Model | Loss | Spike rate | Params |
|---|---|---|---|---|---|
| `train.py` | easy, 1 seed | Dense | 0.00014 | — | — |
| `train.py` | easy, 1 seed | Spiking | 0.00000 | 0.043 | — |
| `multi_seed_experiment.py` | hard, 10 seed | Dense | 0.00011 ± 0.00001 | — | 42,496 |
| `multi_seed_experiment.py` | hard, 10 seed | Spiking | 0.00000 ± 0.00000 | 0.027 ± 0.003 | 42,496 |

**Status: raw loss di tabel ini TERBUKTI metrik gak valid buat dibandingin antar-arsitektur (audit Opus §14a).** Angka ini jangan dikutip sebagai bukti apa pun.

### 20b. N-MNIST — Fase A, Single Run (`run_phase_a.py`)

Data: N-MNIST test split, 7000 train / 3000 eval, 300 step self-supervised + 300 step linear probe.

| Model | Downstream test acc | Effective rank (/64 max) | Top-1 SV variance frac | % dim near-zero variance | Verdict |
|---|---|---|---|---|---|
| Dense MLP | **67.80%** | 21.87 | 0.522 | 0.00% | Bersih |
| Spiking LIF | 14.33% | 4.16 | 0.416 | 92.19% | COLLAPSE |
| Trivial-constant | 12.40% | 1.00 | 1.000 | 100.00% | Collapse (by definition) |

### 20c. N-MNIST — Grid Sweep, 45 Run (`grid_sweep.py`)

Data sama (N-MNIST, split sama), 3 seed per kombinasi.

**Spiking (4 beta × 3 LR × 3 seed = 36 run) — SEMUA kombinasi dilaporkan:**

| beta | lr=0.001 | lr=0.003 | lr=0.01 |
|---|---|---|---|
| 0.5 | 0.1240±0.0000 (3/3 collapsed) | 0.1240±0.0000 (3/3 collapsed) | 0.1240±0.0000 (3/3 collapsed) |
| 0.7 | 0.1240±0.0000 (3/3 collapsed) | 0.1240±0.0000 (3/3 collapsed) | 0.1240±0.0000 (3/3 collapsed) |
| 0.9 | 0.1314±0.0129 (3/3 collapsed) | 0.1243±0.0006 (3/3 collapsed) | 0.1240±0.0000 (3/3 collapsed) |
| 0.95 | 0.1272±0.0047 (3/3 collapsed) | 0.1372±0.0193 (3/3 collapsed) | 0.1240±0.0000 (3/3 collapsed) |

Trivial-constant baseline buat pembanding: **0.1240**. Tidak ada satu pun dari 36 run spiking yang beda meaningful dari ini.

**Dense (3 LR × 3 seed = 9 run), simetris:**

| lr | Acc (mean±std) | Collapsed | Reliable? |
|---|---|---|---|
| 0.001 | **0.6907 ± 0.0108** | 0/3 | YA |
| 0.003 | **0.6299 ± 0.0337** | 0/3 | YA |
| 0.01 | 0.4283 ± 0.0887 | 1/3 | Tidak (LR kegedean) |

### 20d. Ringkasan Angka Paling Penting (TL;DR)

- **Dense JEPA di data real (N-MNIST): 62-69% akurasi, konsisten, gak collapse.** Ini hasil yang VALID dan bisa dipegang.
- **Spiking JEPA (snnTorch LIF, surrogate gradient) di data real: ~12-14% di SEMUA 37 kombinasi/run yang dicoba (1 Fase A + 36 grid) — gak pernah beda meaningful dari baseline trivial (12.40%).** Collapse, bukan kemenangan.
- **Eksperimen sintetis lama (20a): metriknya sendiri invalid, jangan dipakai buat klaim apa pun.**

## 21. Deliverable #3 — Audit Klaim ESPP/LPL/PC-SNN (2026-10-02)

User pilih fokus ke Dense (udah solid) + lanjut Deliverable #3: audit klaim paper local-learning-rule SNN pakai protokol collapse kita.

### 21a. LPL (Halvagal & Zenke, Nature Neuroscience 2023) — Clone & Inspeksi Kode

**(F)** Repo publik ada: `github.com/fmi-basel/latent-predictive-learning`. **Dua implementasi terpisah:**
- `spiking_simulations/` — **C++, butuh Auryn simulator** (compile dari commit spesifik `36b3c197`), beda total stack dari PyTorch kita. Adaptasi mahal (bukan swap-in, butuh belajar framework simulasi baru + bridge format output).
- Top-level (`lpl_main.py`, `models/`, `metrics/`) — **PyTorch Lightning biasa**, compatible langsung sama environment kita, tapi ini implementasi NON-spiking (VGG-11 di CIFAR-10 dkk).

**(F) Temuan kunci — LPL udah punya diagnostic collapse sendiri, LEBIH lengkap dari kita**: `metrics/metrics.py` punya `participation_ratio()` — `pr = (sum eigenvalue)² / sum(eigenvalue²)` — ini SETIPE sama effective-rank SVD kita. Plus: class correlation matrix, manifold radius/dimension, uniformity/push-pull metrics (Wang & Isola 2020), sliced Wasserstein distance. **Kesimpulan: LPL BUKAN kandidat audit yang bagus** — mereka udah rigorous, kemungkinan lebih rigorous dari kita.

### 21b. ESPP (Graf, Su, Indiveri, arXiv 2405.13976) — Baca Full-Text

**(F)** Pakai frozen encoder + classifier terpisah (mirip downstream probe) — metodologi dasarnya OK.
**(F)** **TIDAK ADA** diagnostic collapse (effective rank/variance/participation ratio) dan **TIDAK ADA** baseline trivial/random yang dilaporkan di paper.
**(F)** Akurasi tinggi: **N-MNIST 95.28%** (vs ETLP 94.30%), **SHD 84.32%** (vs ETLP 59.19%, OSTTP 77.33%, BPTT 83.2%).
**Kesimpulan jujur**: gap metodologi ADA (gak explicit cek collapse), tapi **dampaknya kecil** — akurasi 95%+ itu secara matematis gak mungkin dicapai encoder yang collapse (bandingkan: trivial-constant KITA di N-MNIST cuma 12.40%). Gap metodologi nyata, tapi gak nunjuk ke kesimpulan yang salah.

### 21c. PC-SNN (Lan, Xiong, arXiv 2211.15386) — Baca Full-Text

**(F)** **Bukan self-supervised** — predictive coding dipakai sebagai local learning rule alternatif backprop buat **supervised classification langsung ke label**, bukan buat belajar representasi tanpa label.
**(F)** Gak ada linear probe/frozen encoder, gak ada diagnostic collapse, gak ada baseline trivial.
**(F)** Akurasi: Caltech 99.3%, MNIST/N-MNIST 98.1%/98.5%.
**Kesimpulan jujur**: risiko collapse yang kita selidiki (JEPA-style, EMA target encoder, trivial shortcut) **gak relevan buat setup PC-SNN** — supervised training langsung ke label gak punya failure mode yang sama. Bukan "mereka lolos audit", tapi "audit ini gak applicable ke mereka".

### 21d. Kesimpulan Deliverable #3 Round 1 (Negative Result Jujur)

**3 dari 3 paper yang dicek (LPL, ESPP, PC-SNN) TIDAK rentan ke collapse-blindness yang sama kayak eksperimen kita sendiri — tapi karena 3 ALASAN BEDA**, bukan karena mereka semua "lolos tes":
1. LPL: udah punya diagnostic representation sendiri yang lebih lengkap dari kita
2. ESPP: gap metodologi ada (gak explicit check), tapi akurasi absolut terlalu tinggi buat collapse jadi penjelasan yang masuk akal
3. PC-SNN: gak masuk kategori risiko ini sama sekali (supervised, bukan self-supervised)

**Pelajaran metodologis**: audit "apakah paper X punya masalah yang sama kayak kita" itu gak otomatis nemu masalah — kadang hasilnya genuinely negative, dan itu HARUS dilaporkan sebagai negative result yang jujur (bukan disembunyiin atau dipaksa nemu masalah biar ada "temuan"). Ini konsisten sama disiplin yang udah dipegang sepanjang riset ini (§8, §9, §16 — gak overclaim).

**Next**: cari kandidat paper lain yang lebih mungkin match kriteria audit (self-supervised/predictive + local learning rule + SNN, idealnya ada kode publik).

### 21e. Kandidat Baru Ditemukan — PredNext (arXiv 2509.24844) Jauh Lebih Match

Search lanjutan nemu beberapa kandidat (SpikeCLR, NeuroMoCo, "Learning Scalable Temporal Representations... Without Labels", NeuroTrain benchmark framework). **PredNext paling match** — eksplisit bikin varian SNN dari **SimSiam & BYOL** (self-distillation, negative-sample-free) SELAIN SimCLR/MoCo/BarlowTwins. SimSiam/BYOL itu **keluarga arsitektur YANG SAMA PERSIS dengan JEPA kita** (EMA/momentum target, no negative samples) — paling rawan collapse di seluruh literatur SSL, beda dari ESPP/PC-SNN yang gak di kategori ini.

**(F) Baca full-text, temuan kunci:**
- **TIDAK ADA diagnostic collapse** (no effective rank/variance/participation ratio) di hasil utama mereka — cuma downstream accuracy
- **TAPI mereka SENDIRI nunjukin collapse kejadian** di ablation terpisah: "same-view-only prediction" (Table 7) bikin akurasi jatuh ke **5.03%** di UCF101 — mereka SADAR collapse itu nyata risikonya di keluarga arsitektur ini, tapi gak apply diagnostic yang sama ke hasil UTAMA mereka
- Kutipan eksplisit: **"SimSiam lacks negative samples... leading to relatively unstable training"** — pengakuan instabilitas, tapi gak diikutin pengecekan rigorous
- **Kode belum publik** ("plan to release", belum ada link per saat ini) — audit cuma bisa dari paper, gak bisa re-run
- **Angka akurasi (penting buat interpretasi)**: SimSiam baseline 50.81% top-1 (UCF101, tanpa ImageNet pretrain, 101 kelas), PredNext+SimSiam 54.93%. Dengan ImageNet pretrain: PredNext+SimSiam 72.24% vs baseline 70.32%.

**(H) Kenapa ini kasus yang lebih AMBIGU dan lebih menarik dari ESPP/PC-SNN**: akurasi 50-54% di task 101-kelas itu JAUH di atas chance (~1%) — jadi BUKAN collapse total. Tapi beda dari ESPP (95%+, hampir mustahil collapse signifikan), margin 50-54% di sini **ninggalin ruang nyata buat PARTIAL collapse** (encoder cuma makai sebagian kecil kapasitas representasinya, akurasi "cukup tinggi buat lolos" tapi belum optimal) — ini PERSIS jenis kasus yang baru ketahuan kalau effective-rank/variance dicek, gak kelihatan dari angka akurasi doang.

**Ini kandidat audit terbaik sejauh ini**: mereka SENDIRI udah buktiin collapse itu risiko nyata di arsitektur mereka (Table 7), tapi gak apply diagnostic yang sama ke hasil utama yang dilaporkan sebagai temuan positif. Gak bisa re-run (kode belum ada), tapi kritiknya bisa dirumuskan dengan presisi dari apa yang mereka tulis sendiri.

### 21f. Pencarian Kandidat BERKODE — SpikeCLR Ditemukan

Cek 3 kandidat tambahan dari search sebelumnya (NeuroMoCo, SpikeCLR, "Learning Scalable Temporal Representations... Without Labels") khusus buat ketersediaan kode:

| Paper | Kode publik? | Linear probe? | Diagnostic collapse? |
|---|---|---|---|
| **SpikeCLR** (Vaillant dkk) | **(F) YA** — `github.com/maxime-vaillant/SpikeCLR` | **(F) YA**, eksplisit (Linear Probing + Fine-Tuning) | **(F) TIDAK ADA** (no effective rank/variance/participation ratio) |
| NeuroMoCo (arXiv 2406.06305) | (F) Tidak ada | Gak jelas | (F) Gak ada sama sekali |
| "Scalable Temporal Reps..." (arXiv 2511.18542) | (F) Tidak ada (cuma sebut framework PyTorch/Lava-DL) | (F) Ya (linear eval) | (F) Parsial — ada t-SNE, KL divergence, cosine similarity, spike rate, TAPI bukan effective rank/participation ratio |

**(F) SpikeCLR jadi kandidat audit terbaik buat RE-EXECUTION** (bukan cuma kritik dari paper): kode asli ada, metodologi dasarnya udah pakai linear probing (match protokol kita), tapi nol diagnostic collapse. **Catatan penting soal jenis risiko**: SpikeCLR itu contrastive learning (bukan self-distillation/EMA-target kayak JEPA/BYOL/SimSiam) — negative sample di loss contrastive secara desain nyegah collapse TOTAL (representasi konstan bikin loss buruk). Tapi **dimensional/partial collapse** (Jing et al. 2022, udah di-flag Opus di §14e) tetep bisa kejadian di contrastive learning juga — anisotropic collapse ke subspace berdimensi rendah, bukan ke titik konstan. Ini risiko yang beda dari yang kita temuin sendiri di SpikingJEPA (§16-17, collapse total), tapi tetap valid buat dicek.

**Next**: clone `github.com/maxime-vaillant/SpikeCLR`, inspeksi struktur kode (read-only, belum eksekusi apa pun) buat nentuin effort adaptasi ke protokol collapse kita.

### 21g. Clone & Inspeksi Kode — Temuan yang Mengubah Arah Audit

**(F) Koreksi nama**: repo-nya sendiri dinamain **EventCLR** (bukan "SpikeCLR" — itu nama yang dipakai di teks paper, repo/project-nya EventCLR). Dicatat biar gak salah kutip ke depannya.

**(F) Struktur**: PyTorch Lightning + SpikingJelly (bukan snnTorch kayak kita), MLflow buat tracking, support **N-MNIST** — dataset yang SAMA PERSIS kita pakai sendiri di §16-17. Requirement: Python≥3.12, PyTorch≥2.9, Lightning≥2.6, **cupy** (GPU backend SpikingJelly) — beda versi dari environment kita (torch 2.5.1), kemungkinan butuh venv terpisah.

**(F) TEMUAN KUNCI — kontradiksi langsung sama audit-dari-paper sebelumnya**: `utils/metrics.py` fungsi `encoder_representation_metrics()` punya:
- `stable_rank` dan **`effective_rank`** — formula **PERSIS SAMA** kayak `collapse_diagnostics.py` kita (SVD → p=S²/sum(S²) → exp(entropy))
- **`dead_neuron_frac`** dan `saturated_neuron_frac` — persis hipotesis "dead neuron" yang kita duga (belum terverifikasi) jadi sebab collapse SpikingJEPA kita sendiri di §16!
- Plus cosine similarity positive/negative pair (alignment/uniformity check gaya contrastive learning)

**Artinya**: audit dari paper (WebFetch arxiv, §21f) bilang "gak ada diagnostic collapse" — tapi KODE ASLI-nya PUNYA diagnostic lengkap. **Gap-nya bukan di kode, tapi di apa yang dipublish/dilaporin di teks paper** — mereka ngukur ini secara internal (kemungkinan di-log ke MLflow), tapi gak nongol di hasil yang dibaca pembaca paper. Ini **TEMUAN YANG BEDA DAN LEBIH SPESIFIK** dari yang dikira: bukan "mereka gak pernah ngecek", tapi **"mereka ngecek tapi gak transparent soal hasilnya ke pembaca"**.

**Revisi framing audit**: dari "re-run buat nangkep collapse yang mereka lewatin" jadi **replication check** — jalanin kode mereka, pakai diagnostic MEREKA SENDIRI (`effective_rank`, `dead_neuron_frac`) plus punya kita, dan lihat: (a) apakah angka yang muncul konsisten sama akurasi yang dipublish di paper, (b) apakah ada sinyal collapse/dead-neuron yang KALAU dipublish bakal ngubah interpretasi hasil mereka. Bonus: ini jadi cross-validation buat formula `effective_rank` kita sendiri (dua implementasi independen, formula sama — kalau hasilnya konsisten, itu nambah keyakinan ke tooling kita sendiri).

### 21h. Log Teknis Setup & Debugging (dicatat detail — bakal diaudit Opus lagi nanti)

**Rencana eksekusi yang di-approve**: smoke test `main.py --target-dataset nmnist --backbone resnet18 --pretrain-epochs 30 --eval-epochs 20 --eval-types lp --samples-per-class 1.0 --n-subset-runs 1`, dengan patch `representation_log` 20→5 di `modules/self_supervised.py` (biar dapet 6 checkpoint diagnostic: epoch 0,5,10,15,20,25, bukan cuma 1).

**(E) Setup venv**: Python 3.12 terpisah (`spikeclr_repo/.venv`), `uv pip install -r requirements.txt` (torch 2.9.1+cu128, SpikingJelly dari git commit pinned, Lightning 2.6.0, tonic 1.6.0, mlflow 3.10.0) — semua sukses.

**(E) Masalah #1 — dependency ketinggalan**: `lmdb` gak ada di `requirements.txt` mereka tapi di-import di `datasets/nimagenet.py` (dependency transitif lewat `dataset_factory.py` yang import semua dataset class termasuk yang gak kepake). Fix: `uv pip install lmdb`.

**(E) Masalah #2 — cupy hardcoded, bukan opsional**: smoke test awal kita pakai `neuron.LIFNode()` polos (tanpa kwargs) dan itu default ke backend `torch`, jalan normal. Tapi kode ASLI EventCLR pass `backend='cupy'` secara eksplisit lewat `neuron_kwargs` yang di-hardcode di **`main.py` baris 123** — bukan di file backbone yang kita duga awal. Butuh 2 ronde investigasi: percobaan pertama patch `functional.set_backend(..., backend='torch')` di 6 lokasi (`models/pretraining.py`, `models/finetuning.py`, `models/linear_probing.py`, `models/backbones/sew_resnet.py`, `models/backbones/spiking_vgg.py`, `modules/supervised.py`) — ini GAGAL karena error terjadi di construction time (`LIFNode.__init__`) SEBELUM `set_backend` dipanggil (`set_backend` ada di titik yang terlalu akhir, strategi salah). Fix yang beneran kerja: patch `main.py:123` langsung, `'backend': 'cupy'` → `'backend': 'torch'`.
**Pelajaran teknis**: patch di titik "biasanya betul" (API resmi `set_backend`) belum tentu nutup akar masalah kalau konfigurasi aslinya diinject lebih awal di construction — harus trace ke titik literal string/value itu pertama kali muncul (`grep` buat string `'cupy'` langsung, bukan nebak API call mana yang relevan).

**(E) Masalah #3 — CUDA OOM di batch_size default**: `--pretrain-batch-size 256` (default) + SEWResNet18 full + multi-step mode (4 timestep) + N-MNIST di laptop GPU 8GB → **CUDA out of memory** (7.63GB/7.70GB terpakai) di epoch 0, batch pertama. Fix yang di-apply: turunin `--pretrain-batch-size`/`--eval-batch-size` ke 32, plus `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` (disaranin langsung di pesan error PyTorch).
**Catatan buat interpretasi hasil nanti**: batch_size 32 (bukan 256 default mereka) bisa mempengaruhi training dynamics (gradient noise lebih tinggi, butuh LR beda buat konvergensi optimal) — hasil smoke test kita TIDAK akan match persis skala paper mereka, ini emang bukan tujuannya (tujuannya cek collapse signal, bukan reproduce akurasi final mereka tepat).

**Status**: fix OOM belum di-rerun (interupsi user buat dokumentasi ini duluan). Next: rerun dengan parameter yang udah di-adjust.

**(E) Masalah #4 — training epoch 0 SUKSES (1875 batch, ~5 menit, loss turun ke 4.2), tapi diagnostic MEREKA SENDIRI crash dengan OOM baru:**
```
on_train_epoch_end -> encoder_representation_metrics -> sim_matrix = H_i_n @ H_j_n.T
torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 13.41 GiB
```
**Root cause**: `self._h_i_buffer`/`_h_j_buffer` di `modules/self_supervised.py` nampung representasi SEMUA batch dalam 1 epoch (60.000 sampel @ batch_size 32 = 1875 batch terkumpul), terus `encoder_representation_metrics()` bikin `sim_matrix = H_i_n @ H_j_n.T` — matrix **60.000×60.000** = 3.6 miliar float = ~14.4GB. **Ini bukan masalah yang kita introduce lewat patch** — `representation_log` ORIGINAL mereka (20) juga trigger di epoch 0 (`0 % 20 == 0`), jadi config asli mereka JUGA akan OOM di GPU manapun dengan VRAM <~16GB.

**Temuan genuinely baru** (gak ketangkep dari baca paper atau "kritik Gemini" yang cuma restate Opus): **kode diagnostic collapse mereka sendiri punya bug skalabilitas O(N²)** — gak di-chunk/subsample, cuma teruji di hardware dengan VRAM besar (kemungkinan A100/H100 di lab mereka). Ironis: ini persis "sub-sampling buat SVD" yang kita TOLAK pas disaranin Gemini buat kode KITA SENDIRI di §19 (karena emang gak perlu di skala kecil kita, N=7000/D=64) — tapi di sini, di skala N=60.000 buffer penuh + full similarity matrix O(N²), itu genuinely dibutuhkan. **Pelajaran: teknik optimasi yang sama bisa salah di satu konteks (skala kecil, gak perlu) dan bener di konteks lain (skala besar, perlu) — gak ada aturan universal, harus dicek skalanya tiap kasus.**

**Fix yang mau di-apply**: cap `_h_i_buffer`/`_h_j_buffer` ke subset (~2048 sampel) sebelum dihitung diagnostic-nya, bukan nampung semua 60.000. Ini modifikasi KITA ke kode mereka buat kebutuhan audit di hardware terbatas, dicatat jelas sebagai patch bukan kode asli.

### 21i. Hasil Run (setelah fix OOM) — (H) PRELIMINARY, BUKAN Kesimpulan Kuat

**(F) Masalah #5 — gak ada checkpoint/resume mechanism**: dicek `pipelines/pretrain.py`/`modules/self_supervised.py`, gak ada `ModelCheckpoint` callback atau save mid-training. Artinya **interupsi apa pun (termasuk stop manual) = kerja hilang total, harus restart dari epoch 0.** Ini keterbatasan teknis kode mereka yang baru ketahuan lewat eksekusi nyata, bukan dari baca paper.

**(E) Data yang ke-collect sebelum run dihentikan (alasan thermal, lihat di bawah):**

| Epoch | pretrain_loss | effective_rank | stable_rank | dead_neuron_frac | firing_rate_mean |
|---|---|---|---|---|---|
| 0 | 4.20 | 190.40 | 7.52 | 0.0% | 0.134 |
| 5 | 2.58 (run 1) | 33.97 | 7.04 | 1.76% | 0.180 |
| 10 | — | 38.16 | 7.86 | 2.15% | 0.188 |
| 13 | 2.06 | — | — | — | — |

**(H) Interpretasi — DITURUNKAN kekuatannya setelah dikoreksi user**: tren "effective_rank jatuh tajam lalu plateau" (190→34→38) ITU TETAP TERLIHAT di data yang ada, TAPI **cuma 3 titik, bukan kurva penuh sampai konvergensi (target awal 15 epoch, apalagi buat match skala paper mereka 500 epoch)**. **Ini BUKAN bukti kuat "EventCLR gak collapse"** — ini cuma **indikasi awal yang mengarah ke arah itu**, bisa aja rank jatuh lagi di epoch 20+ yang belum ke-observasi. **Koreksi dari overclaim sebelumnya**: kesimpulan "EventCLR tidak collapse" yang ditulis di respons chat sebelum ini terlalu kuat buat data n=3 — revisi jadi "sinyal awal MENDUKUNG non-collapse, belum cukup buat kesimpulan final."

**Kenapa run dihentikan**: CPU Tctl naik ke 91.5°C di tengah run (laptop Victus, riwayat thermal sensitif — lihat `project_hp_victus_fan_control`/`project_victus_rtw89_wifi_fw_crash`). Fan sudah di RPM maksimal (5400/5233), GPU self-throttle aktif (clock 3105→2340MHz). User sempat coba kipas manual eksternal, gak banyak membantu. **Tidak ada checkpoint tersimpan (lihat Masalah #5), jadi stop = kehilangan progress, bukan "pause".**

**Keputusan user (2026-10-02)**: TIDAK restart buat nambah data point — data n=3 yang ada dipakai apa adanya, didokumentasikan dengan hedging yang jujur (bukan kesimpulan kuat), demi menghindari risiko thermal lagi. **Status Deliverable #3 / audit EventCLR: BERHENTI di titik ini, kesimpulan PRELIMINARY.** Kalau mau kesimpulan lebih kuat nanti, butuh: (a) tambah checkpoint callback di kode mereka (modifikasi kita) sebelum run ulang, (b) pantau suhu lebih ketat/nonstop, (c) idealnya run di sesi terpisah yang gak numpuk sama kerjaan lain di laptop yang sama.

### 21j. Ringkasan Kelemahan Audit EventCLR Ini — Eksplisit, Jangan Disembunyikan

User secara eksplisit flag "test kita masih lemah" (2026-10-02) — ini daftar lengkap kenapa, biar siapa pun yang baca dokumen ini (termasuk Opus di audit berikutnya) gak salah baca kekuatan buktinya:

1. **n=3 titik data** (epoch 0, 5, 10) — jauh dari cukup buat karakterisasi tren statistik yang robust, apalagi dibandingin grid sweep kita sendiri yang pakai 3 seed per kombinasi (§17).
2. **1 seed doang** — gak tau apakah pola 190→34→38 ini konsisten across random init, atau kebetulan satu run ini aja.
3. **Batch_size 32, bukan default 256 mereka** — diadaptasi buat GPU 8GB, training dynamics bisa beda (dicatat di §21h Masalah #3).
4. **Pretrain-epochs 15 (target), 500 di paper mereka** — bahkan kalau sampe epoch 15, itu cuma 3% dari skala training asli. Trend "plateau" di epoch 5-10 bisa aja cuma plateau LOKAL yang nanti jatuh lagi, atau naik lagi, di epoch 100-500 yang gak pernah kita observasi.
5. **1 dataset (N-MNIST), 1 backbone (ResNet18)** — EventCLR support 5 dataset + 3 backbone, kita cuma test 1 kombinasi kecil dari matrix penuh mereka.
6. **Cuma linear-probing (LP), skip fine-tuning (FT) & supervised baseline (SUP)** — gak ada perbandingan ke 2 dari 3 protokol evaluasi yang mereka desain.
7. **Representation_log di-patch (20→5)** — ini intervensi KITA ke kode mereka, bukan konfigurasi asli yang mereka validasi.
8. **Run dihentikan paksa karena thermal, bukan karena mencapai kesimpulan natural** — data berhenti di titik yang ditentukan kondisi hardware, bukan titik yang secara ilmiah optimal buat berhenti.

**Kesimpulan yang BOLEH diambil dari ini**: "ada sinyal awal yang worth diselidiki lebih jauh soal dimensional collapse di EventCLR, berbeda karakter dari collapse total yang kita temuin di eksperimen sendiri." **Kesimpulan yang TIDAK BOLEH diambil**: "EventCLR terbukti collapse" atau "EventCLR terbukti tidak collapse" — dua-duanya overclaim buat bukti yang ada.

## 22. Deep-Dive: State Space Models (S4/Mamba/Liquid-S4) — Nutup Blind Spot Terbesar dari Opus

Ini blind spot yang Opus ranking **#1 paling merusak** (§14e) — belum pernah kita gali sampai sekarang. Riset literatur murni, gak ada eksekusi/risiko thermal.

**(F) S4 (Structured State Space Sequence Model, Gu et al. 2021)**: continuous-time SSM `ẋ(t) = Ãx(t) + B̃u(t)`, `y(t) = C̃x(t) + D̃u(t)`. Inovasi kunci: matrix `A` di-inisialisasi pakai **framework HiPPO** ("Highly Predictive Polynomial Operators") — proyeksi input streaming ke basis polynomial orthonormal (Legendre weighted) yang **secara teoretis optimal buat long-term memory retention**. Reparameterisasi **DPLR** (diagonal plus low-rank) bikin komputasinya reduce ke Cauchy kernel yang efisien — ini yang bikin S4 bisa linear/sub-quadratic di panjang sequence, beda dari attention yang quadratic.
Sumber: [Efficiently Modeling Long Sequences with Structured State Spaces](https://ar5iv.labs.arxiv.org/html/2111.00396)

**(F) Mamba (Gu & Dao, 2023)**: improvement dari S4 — nambah **mekanisme selective**, matrix A/B/C jadi **input-dependent** (dinamika berubah tergantung input saat itu), beda dari S4 yang time-invariant/fixed. Ini modul disebut **S6**. Hasil: **5x throughput inference lebih tinggi dari Transformer**, linear scaling di panjang sequence, SOTA di bahasa/audio/genomics.
Sumber: [Mamba: Linear-Time Sequence Modeling with Selective State Spaces](https://arxiv.org/pdf/2312.00752)

**(F) TEMUAN KUNCI — Liquid-S4 (2022) udah gabungin LTC + S4, dan ini MENGKONFIRMASI kritik Opus secara eksplisit**: Liquid-S4 nambahin **state transition scheme yang input-sensitive** ala LTC ke S4 — persis "nonlinear function yang reaktif ke input" yang jadi ciri khas LTC, dipasang di atas framework SSM yang efisien. **Hasil terukur: 87.32% rata-rata di Long-Range Arena benchmark (SOTA), 96.78% di Speech Command dengan parameter 30% LEBIH SEDIKIT dari S4 biasa.**
Sumber: [Liquid Structural State-Space Models](https://arxiv.org/abs/2209.12951)

**Implikasi langsung ke §10/§11 (desain ablation kita yang lama)**: Opus benar — kontrol "temporal integration" yang paling murni DAN tervalidasi itu **S4/Mamba/Liquid-S4**, bukan GRU. Liquid-S4 secara spesifik udah **membuktikan** LTC (continuous-time, input-dependent) + SSM (efisien, linear-scaling) itu kombinasi yang menang telak (SOTA + parameter lebih sedikit) — bukan hipotesis lagi, ini udah published & benchmark 2022. Kalau mau lanjut ablation GRU-vs-LIF yang pernah kita rencanain (sekarang di-drop per §15), seharusnya kontrolnya **Liquid-S4**, bukan GRU biasa — tapi ini TETAP di bawah arah resmi auditor-metodologi (§15), bukan alasan buat balik ke arsitektur baru.

**Catatan konteks 2026**: ada juga riset lanjutan ("Advancing Intelligent Sequence Modeling: Evolution... from S4 to Mamba", arXiv 2503.18970) yang survey evolusi S4→Mamba lebih jauh — worth dibaca kalau butuh gambaran lengkap state-of-the-art SSM terkini, belum dibaca detail di sesi ini.

### 22a. Lebih Dalam — Matematika Presisi Selectivity Mamba

**(F) Yang DIBUAT input-dependent, dan yang TETAP fixed**: matrix `A` (state evolution) **TETAP FIXED** (buat stabilitas & alasan komputasi) — yang jadi input-dependent itu **Δ (discretization step size), B, dan C**, dihitung dari input saat itu lewat linear projection ringan:
```
Δ̃ₖ = softplus(W_Δ · uₖ)
Bₖ = W_B · uₖ
Cₖ = W_C · uₖ
```
Discretization continuous→discrete: `Ā = e^(AΔ)`, `B̄ = (AΔ)⁻¹(e^(AΔ) - I)·ΔB`.

**(F) Intuisi Δ (ini kuncinya)**: **Δ besar → prioritasin input saat ini, LUPAIN state tersimpan. Δ kecil → ABAIIN input saat ini, PERTAHANKAN state.** Ini mekanisme "selective forget/attend" yang dibangun LANGSUNG ke dalam proses diskritisasi itu sendiri — bukan gate tambahan terpisah kayak LSTM/GRU (forget gate, input gate). Elegan secara matematis: satu parameter (Δ) yang udah ada secara natural di SSM, dibuat input-dependent, otomatis jadi mekanisme seleksi.

**(F) Kenapa ini kerja**: recurrence jadi gak time-invariant lagi (tiap token modulasi dinamikanya sendiri), TAPI tetap compatible sama **parallel scan** (algoritma yang bikin training-nya tetep efisien/paralel, gak kehilangan keunggulan komputasi SSM walau udah jadi "selective").
Sumber: [Here Comes Mamba](https://towardsdatascience.com/here-comes-mamba-the-selective-state-space-model-435e5d17a451/)

### 22b. Limitasi Nyata SSM/Mamba — Biar Gak Kebawa Hype Sepihak

**(F) Kelemahan fundamental — copying/retrieval**: SSM compress seluruh input ke **fixed-size latent state** — ini bottleneck informasi genuine. **Transformer dengan parameter 10x LEBIH SEDIKIT ngalahin Mamba di retrieval, dan Mamba butuh data training 100x LEBIH BANYAK buat belajar copying sederhana.** Ini bukan gap kecil.

**(F) Vanilla SSM (sebelum Mamba) malah gak bisa selective copying SAMA SEKALI**: karena `A,B,C` fixed sama buat semua token, SSM gak bisa bedain token satu sama lain secara konten.

**(F) Tapi Mamba (dengan selectivity) justru MENANG TELAK di induction heads**: "solve perfectly" dan **generalize ke sequence 4000x lebih panjang dari yang dilihat saat training** — metode lain maksimal cuma 2x. Trade-off yang jelas: SSM/Mamba kuat di **long-range extrapolation + compute linear**, lemah di **exact retrieval/copying** (yang justru jadi kekuatan utama attention Transformer).

**Kalibrasi jujur**: SSM/Mamba BUKAN "pengganti Transformer yang lebih baik di segala hal" — ini trade-off nyata, menang di satu sumbu (panjang sequence, efisiensi), kalah di sumbu lain (retrieval presisi). Relevan buat riset kita: task JEPA (prediksi representasi future dari context) itu lebih dekat ke "long-range extrapolation" daripada "exact copying" — jadi secara teori SSM/Mamba COCOK buat use-case ini, tapi itu masih hipotesis belum teruji langsung.

### 22c. Koneksi LANGSUNG ke Domain Kita — SNN + SSM Udah Jadi Bidang Aktif

**(F) SiLIF (2025)** — structured SSM dynamics yang di-parameterisasi khusus buat neuron spiking.

**(F) TEMUAN PALING RELEVAN — diagonal structured SSM UDAH diimplementasi di Loihi 2 asli (2025)** buat streaming sequence processing. **Ini persis validasi hardware yang Opus bilang "belum ada" buat kombinasi LTC+JEPA** — ternyata buat SSM secara umum, validasi hardware neuromorphic itu UDAH ADA, cuma belum spesifik ke LTC+JEPA kita.

**(F) Spiking SpeechMamba** — binary spike, **>70% activation sparsity**, performa kompetitif dengan **30% parameter lebih sedikit**. ini angka efisiensi konkret yang align sama klaim-klaim "neuromorphic hemat energi" yang kita audit sepanjang dokumen ini.

**(F) SpikeMamba (GitHub, The-Swarm-Corporation)** — ada **kode publik** yang gabungin SNN + Mamba buat language modeling. **Kandidat baru buat Deliverable #3** kalau mau dilanjutin (belum dicek collapse-diagnostic-nya, belum dicek apakah applicable ke audit kita) — dicatat dulu, belum dieksekusi.

**Sintesis §22**: blind spot SSM yang Opus flag itu beneran besar — bukan cuma soal "kontrol ablation yang lebih baik" (§22 awal), tapi ternyata **seluruh sub-bidang SNN+SSM udah aktif berkembang** dengan validasi hardware neuromorphic nyata (Loihi 2), appoach yang lebih matang dari yang kita eksplorasi sejauh ini (ESPP/LPL/EventCLR semua masih LIF-based, belum SSM-based). Kalau Deliverable #3 lanjut lagi, SpikeMamba atau paper SNN+SSM lain mungkin kandidat audit yang lebih kuat dari yang udah dicoba.
Sumber: [SiLIF](https://arxiv.org/pdf/2506.06374), [Spiking SpeechMamba context](https://arxiv.org/html/2606.01135v1), [SpikeMamba GitHub](https://github.com/The-Swarm-Corporation/SpikeMamba)

### 22d. NAIK SATU LEVEL — Teori Komputasi SSM, Nyambung Langsung ke §12-13

Ini riset "lebih tinggi" yang diminta user: hubungin arsitektur modern (§22) ke computability theory paling dalam yang udah kita bangun di §12-13.

**(F) Mamba-2 / Structured State Space Duality (SSD, Dao & Gu, Mei 2024)**: **Transformer dan SSM itu DUA SISI MATA UANG MATEMATIKA YANG SAMA.** Kalau matrix `A` Mamba disederhanain jadi diagonal (scalar × identity), transformasi SSM-nya bisa diekspresikan sebagai **causal linear attention**. Koneksinya lewat **semiseparable matrices** — kelas matrix terstruktur yang kompleksitasnya sub-quadratic. Attention (quadratic) dan recurrence (linear) itu BUKAN dua paradigma fundamental beda — itu **dua cara DEKOMPOSISI matrix semiseparable yang SAMA**, beda cuma di cara ngitungnya (satu kasih bentuk attention-like quadratic, satu kasih bentuk recurrent linear).
Sumber: [Transformers are SSMs](https://arxiv.org/pdf/2405.21060), [State Space Duality Part I](https://tridao.me/blog/2024/mamba2-part1-model/)

**(F) Formal language/Chomsky hierarchy**: SSM bisa ekspresikan star-free language, sejumlah counter language, bounded hierarchical structure. **Temuan mengejutkan**: **"Transformer gak well-aligned sama Chomsky hierarchy karena GAK BISA kenalin beberapa regular language"** — bahkan level PALING BAWAH/sederhana di hierarchy Chomsky (regular language, yang dikenalin finite automaton biasa) gak sepenuhnya di-cover Transformer. Ini ngejutin karena regular language itu level paling "mudah" secara teori.
Sumber: [The Expressive Capacity of SSM: A Formal Language Perspective](https://arxiv.org/pdf/2405.17394), [Neural networks and the Chomsky hierarchy](https://arxiv.org/pdf/2207.02098)

**(F) TEMUAN PALING DALAM — Circuit Complexity Bound, ini jawaban definitif**: **SSM (termasuk Mamba) BISA disimulasikan oleh TC0 circuit** (threshold circuit kedalaman konstan) **dan GAK BISA ekspresikan komputasi DI LUAR TC0.** Konsekuensi konkret: SSM **GAK BISA** nyelesaiin permutation composition, **GAK BISA** akurat nge-track langkah catur, **GAK BISA** evaluasi kode, **GAK BISA** nge-track entitas di narasi panjang. **Dan yang paling penting: "Mamba punya kemampuan komputasi yang SAMA PERSIS dengan Transformer secara teoretis"** — ini **MELAWAN asumsi umum** bahwa Mamba lebih expressive/mampu secara komputasi dari Transformer.
Sumber: [The Computational Limits of State-Space Models and Mamba via Circuit Complexity](https://pith.science/paper/2412.06148), [The Illusion of State in State-Space Models](https://arxiv.org/pdf/2404.08819)

**SINTESIS BESAR — ini yang nyambungin §12-13 ke §22**: pola yang PERSIS SAMA kejadian dua kali di dua domain berbeda di dokumen ini:
- **§12**: von Neumann (digital, unbounded memory) = Turing-complete. Neuromorphic (digital) = **SAMA PERSIS** Turing-complete — "gak lebih gak kurang", cuma beda efisiensi.
- **§22d**: Transformer = bounded TC0. Mamba/SSM (arsitektur radikal beda) = **SAMA PERSIS** bounded TC0 — "gak lebih gak kurang", cuma beda efisiensi/profil komputasi (long-range vs retrieval).

**Pola umum yang kebentuk**: **ganti arsitektur/substrat TANPA ganti kelas kompleksitas fundamental-nya itu CUMA mindah-mindah trade-off efisiensi, BUKAN nambah kemampuan komputasi mentah.** Von Neumann vs neuromorphic: sama-sama Turing-complete. Transformer vs Mamba: sama-sama TC0. Ini bukan kebetulan — ini nunjukin ada **hierarki kelas kompleksitas yang tegas** (regular language < TC0 < Turing-complete < P/poly) dan **inovasi arsitektur AI terkini (Transformer, Mamba, SSM) semuanya masih terjebak di SATU kelas yang sama (TC0)**, jauh di bawah Turing-complete, apalagi P/poly. **Implikasi buat pertanyaan awal seluruh riset ini ("komputer generasi baru buat AI lebih maju")**: lompatan generasi yang BENERAN (naik kelas kompleksitas, bukan cuma pindah arsitektur dalam kelas yang sama) itu JAUH lebih jarang dan lebih fundamental dari yang kelihatan — Mamba vs Transformer itu BUKAN lompatan kelas, itu cuma pergeseran di DALAM kelas TC0 yang sama.

### 22e. Chain-of-Thought = Jembatan TC0→Turing (Verifikasi Hipotesis)

**(F) Terverifikasi**: Transformer dengan Chain-of-Thought **TERBUKTI Turing-complete** — treat autoregressive generation sebagai loop, satu forward-pass yang TC0-bounded jadi SISTEM yang Turing-complete begitu diloop tanpa batas (CoT bisa nyelesain P-complete problem kalau step-nya cukup).
Sumber: [On the Turing Completeness of Transformers and Agents](https://arxiv.org/pdf/2609.20335), [Towards Revealing the Mystery behind Chain of Thought](https://arxiv.org/pdf/2305.15408)

**(F) Tapi — pola yang sama kayak Siegelmann §13 muncul LAGI**: bukti ini butuh asumsi ideal (arbitrary precision, positional encoding, hard-max attention). Di dunia nyata: **"transformer di input unbounded kena precision error & sensitivity decay"** — gak bisa beneran simulasi Turing machine yang non-tame dengan confidence positif. Plus ada **"serial-depth bottleneck"**: CoT gak membantu tanpa batas, ada titik di mana nambah step CoT berhenti membantu atau malah ngerusak.
Sumber: [Serial-Depth Bottleneck in LLM Reasoning](https://arxiv.org/pdf/2608.09942), [Position: Turing-Completeness of Autoregressive Transformers Relies...](https://arxiv.org/pdf/2605.19514)

## 23. SINTESIS AKHIR — Satu Benang Merah dari Seluruh Sesi Riset Ini

Ini bukan sitasi satu paper — ini KONEKSI yang kita tarik sendiri dari 5 hasil terpisah yang masing-masing published, dirangkai jadi satu cerita yang belum pernah kelihatan digabung di satu tempat. Ditulis eksplisit sebagai **sintesis kita**, bukan temuan orang lain, biar jelas levelnya.

**Empat "level" kekuatan komputasi yang kita temuin berulang kali, di domain yang beda-beda:**

| Level | Contoh konkret dari dokumen ini | Sifat |
|---|---|---|
| **−1 (di bawah regular)** | Analog + noise fisik (Maass-Orponen/Sontag, §13) | Collapse PAKSA ke bawah — noise itu DESTRUKTIF ke kelas, bukan netral |
| **0 (TC0, bounded-depth)** | Transformer forward-pass, Mamba forward-pass, photonic MZI 1x pass, memristor crossbar 1x pass (§22d) | "Intuisi cepat" — paralel, cepat, TAPI gak bisa reasoning sekuensial unbounded |
| **1 (Turing-complete)** | Von Neumann digital, neuromorphic digital (§12) — DAN level-0 yang DILOOP eksternal tanpa batas (CoT/agent-loop, §22e) | "Deliberasi lambat" — unbounded tapi butuh WAKTU (langkah sekuensial) |
| **+1 (P/poly, super-Turing)** | Analog presisi-tak-terbatas (Siegelmann, §13) | Cuma ADA di atas kertas — butuh info uncomputable yang gak ada prosedur fisik buat "nemuin"-nya (question-begging) |

**POLA YANG KEBENTUK, DAN INI INTINYA**: di SELURUH dokumen ini, **satu-satunya cara yang KETAHUAN dan TERBUKTI buat naik kelas kompleksitas (0→1) itu nambah WAKTU/LOOP SEKUENSIAL EKSTERNAL** (chain-of-thought, agent-loop, program von Neumann biasa) — **BUKAN ganti substrat atau arsitektur statisnya**. Setiap kali kita cari jalan naik kelas LEWAT substrat/arsitektur baru (neuromorphic, photonic, Mamba, SSM), hasilnya SELALU: **sama persis dengan yang lama, cuma beda efisiensi** (§12, §22d). Dan setiap kali kita cari jalan naik LEWAT presisi/fisika analog (bukan lewat waktu/loop), hasilnya **diblokir noise** (collapse ke BAWAH, §13) atau **cuma valid di model ideal yang gak bisa dibangun** (question-begging, §13 & §22e).

**Implikasi paling tajam, dan ini jawaban paling jujur buat pertanyaan yang memulai SELURUH sesi ini** ("komputer generasi baru buat AI yang lebih maju"): **bukan komputernya (substrat) yang perlu diganti — itu udah dicoba lewat 6+ kandidat substrat di dokumen ini dan SEMUANYA mentok di kelas yang sama. Yang beneran kebukti bisa naik kelas adalah DIMENSI WAKTU — loop eksternal yang ngasih "tape gak terbatas" ke module yang bounded.** Ini juga ngejelasin (secara post-hoc, bukan prediksi baru) kenapa progress AI paling impactful belakangan ini bukan arsitektur baru (Mamba dkk, yang kebukti mentok di kelas yang sama kayak Transformer), tapi **reasoning model & agentic tool-use** — itu bukan tren kebetulan, itu **satu-satunya axis yang di dokumen ini TERBUKTI nembus batas kelas kompleksitas**.

**Batasan kejujuran yang tetap berlaku**: ini sintesis KITA (connecting-the-dots dari 5+ paper terpisah), bukan satu hasil yang udah ada orang bilang persis gini. Tiap komponen individual-nya published & robust (circuit complexity SSM, CoT Turing-completeness, Siegelmann, dll) — yang baru dari kita itu NARASI yang nyatuin semuanya jadi satu pola, dan itu HARUS dicek ulang sebelum diklaim sebagai "penemuan" ke siapa pun di luar catatan internal ini (persis disiplin yang kita pegang dari awal — §8, §9, cross-check dulu sebelum percaya).

## 24. AUDIT OPUS RONDE 2 (2026-10-02) — §23 DITARIK, §21g DIKOREKSI, Bug Tooling Ditemukan & Diperbaiki

Spawn Opus lagi khusus buat audit §0+§16-23 (termasuk §23 yang baru ditulis). Opus baca kode langsung (bukan cuma dokumen), spot-check `collapse_diagnostics.py`, `spikeclr_repo/utils/metrics.py`, `spikeclr_repo/modules/self_supervised.py`, dan `git diff` di `spikeclr_repo`. Temuannya lebih keras dari ronde 1.

### 24a. VONIS UTAMA: §23 Punya Lubang Logika Struktural — DITARIK, Bukan Dikasih Hedging

**(F, verifikasi langsung ke klaim paling konkret — dicek kode berdampingan, Opus benar 100%):**
- Formula `effective_rank` kita (`collapse_diagnostics.py`): `p = s / s.sum()` — pakai **S**
- Formula EventCLR (`utils/metrics.py`): `p = (S**2) / (S**2).sum()` — pakai **S²**, PLUS row-normalize (`F.normalize(dim=1)`) yang kita gak lakuin

**→ KLAIM DI §21g "formula PERSIS SAMA, jadi cross-validation" RESMI DITARIK.** Dua formula beda (pangkat beda, preprocessing beda). Angka 190/34/38 (EventCLR) **TIDAK sebanding** dengan 21.87/4.16 (eksperimen kita sendiri di §16).

**Lubang logika §23 (ringkasan, detail lengkap di laporan Opus — lihat chat log 2026-10-02):**
1. **Von Neumann=Turing-complete vs Mamba=TC0 itu BUKAN struktur paralel** — pasangan pertama diukur di resource TAK TERBATAS (dan von Neumann Turing-complete JUSTRU KARENA dia udah pakai loop sekuensial unbounded — loop udah ada di premisnya), pasangan kedua diukur SATU forward-pass (loop dibuang secara sengaja dari definisinya). "Pola yang sama muncul dua kali" itu artefak membandingkan constrained-vs-constrained, bukan temuan empiris.
2. **Headline "lever-nya WAKTU" itu restatement Turing 1936/von Neumann 1945**, bukan sintesis baru — dan **§12 di dokumen kita SENDIRI udah nulis kalimat yang sama** 11 section sebelumnya, kita gak nyadar pas nulis §23.
3. **§13 (Siegelmann) di dokumen kita SENDIRI udah membantah klaim "cuma ganti arsitektur gak bisa nyebrang kelas"** — linear saturated RNN dengan bobot rational itu Turing-complete, beda dari feedforward. Mekanisme kedua buat nyebrang kelas, ada di file kita sendiri, kelewat.
4. **Frame "kelas kompleksitas" ini udah divonis SALAH di §14c sendiri** ("yang nentuin AI itu konstanta energi/waktu + learnability, BUKAN apa yang computable") dan ditutup resmi di §15 — §22-23 buka lagi frame itu tanpa nyadar udah pernah nutup.
5. **Selection effect nyata**: ada mekanisme lain buat nyebrang kelas yang gak dicari (SPACE/memori eksternal — CoT itu sebenernya "mekanisme SPACE berpakaian waktu"; oracle/tool-use trivial Turing-complete; thermodynamic computing yang udah di-flag di §14e #9 tapi gak pernah dicek di lensa complexity class).
6. **Konsekuensi TC0 (gak bisa permutation composition dkk) itu CONDITIONAL ke TC0⊊NC1, yang masih CONJECTURE** — ditulis di §22d seolah teorema pasti.
7. **"Mamba kemampuan komputasi SAMA PERSIS Transformer" itu overclaim** — containment di upper bound yang sama ≠ kesetaraan expressive power. **Pola overclaim yang SAMA PERSIS kejadian lagi** (udah dikoreksi sekali di §14d soal "neuromorphic=von Neumann").

**Pola berulang yang Opus tandai (penting buat dipegang ke depan)**: 4 kali di dokumen ini, overclaim muncul dalam bentuk "dua hal dikasih label/nama yang sama → disimpulkan SAMA PERSIS" padahal yang kebukti cuma "sama-sama termuat di batas yang sama". **Kalibrasi konkret: tiap mau nulis "sama persis"/"identik", WAJIB tunjukin dua objeknya berdampingan (dua formula/definisi/batas resource), bukan cuma dua label yang kedengeran mirip.**

**STATUS §22-23 SETELAH INI**: §22 (matematika Mamba/SSM, limitasi retrieval, koneksi SNN) **tetap valid**, cuma butuh qualifier precision/uniformity yang hilang + 2 klaim tanpa sitasi (angka "10x/100x" di §22b itu dari Jelassi et al. "Repeat After Me" 2024, belum disitasi; klaim Loihi-2+SSM di §22c butuh sitasi langsung, bukan cuma disambungin ke SiLIF/SpeechMamba). **§23 DITARIK sebagai klaim, DITURUNIN jadi catatan proses**: bukan "ini jawaban final", tapi "ini contoh nyata gimana kita sendiri kebawa excitement dan buka kembali frame yang udah kita tutup sendiri 8 section sebelumnya — termasuk Opus di ronde 1 gak nyadar ini pas itu". Itu sendiri pelajaran yang lebih berharga dari klaimnya.

### 24b. Verifikasi Eksekusi — Bug Tooling Diperbaiki, Baseline Random-Init Diukur

**(E) Fix #1 — threshold scale-dependent di `collapse_diagnostics.py`**: `var < 1e-6` (absolut) diganti jadi relatif (`var < relative_eps * mean_variance`, default `relative_eps=1e-3`). Diverifikasi jalan normal di smoke test data random.

**(E) Fix #2 — dua definisi `effective_rank` sekarang eksplisit terpisah**: `effective_rank_S` (punya kita, Roy & Vetterli) dan `effective_rank_S2` (gaya EventCLR). Dua-duanya dilaporkan, gak ada lagi yang diasumsikan "sama".

**(E) BUG BARU ketemu LIVE pas verifikasi (bukan dari Opus, ketemu sendiri pas testing)**: `clamp(min=1e-12)` di `effective_rank()` bikin kasus **collapse TOTAL** (embedding literally semua nol) malah dilaporkan "full rank, no collapse" — karena semua singular value abis di-clamp jadi SERAGAM (sama-sama 1e-12), entropy-nya jadi MAKSIMAL. **Ini collapse-detector kita sendiri salah paling parah justru di kasus collapse paling parah.** Fix: guard eksplisit `if s.sum() < 1e-10: return rank=0, degenerate_all_zero=True` sebelum clamp. Diverifikasi: random-init EventCLR sekarang bener dilaporkan `effective_rank=0.00, COLLAPSE SUSPECTED` (sebelumnya: `256.00, no collapse signature` — salah total).

**(E) Baseline random-init EventCLR (resnet18, N-MNIST, 256 sampel, 0 training step)**: dijalanin read-only, thermal dicek dulu (38°C→68°C setelah, aman). Hasil:
- Kode MEREKA sendiri (`encoder_representation_metrics`): `effective_rank=NaN, stable_rank=NaN, dead_neuron_frac=1.0, firing_rate_mean=0.0` — **NaN karena divide-by-zero di matrix all-zero**, at least ini keliatan rusak (gak silent-wrong kayak punya kita sebelum fix)
- Kode kita (setelah fix): `effective_rank=0.00, COLLAPSE SUSPECTED` — benar

**Anchor baru buat trajectory §21i**: **0 (random init, baru diukur) → 190 (epoch 0 di log lama, SETELAH 1875 step, bukan beneran random init) → 34 (epoch 5) → 38 (epoch 10)**. Ini ngasih gambaran lebih lengkap: backbone EventCLR mulai dari **mati total** (semua neuron silent), "epoch 0" di log sebelumnya itu udah mulai "bangun", lalu stabil di subspace ~35-40. Trajectory ini masih n=4 titik (nambah 1 dari random-init), **masih belum cukup buat kesimpulan kuat** (Opus benar: butuh lebih banyak titik + linear probe + trivial-baseline yang belum pernah dijalanin di protokol §15 buat EventCLR — pelanggaran protokol sendiri yang Opus tangkep).

### 24c. Next Step (Belum Diputuskan)

Rekomendasi prioritas dari Opus: (1) audit PredNext berbasis paper (nol GPU/risiko, paling match — mereka sendiri buktiin collapse di Table 7 tapi gak apply diagnostic yang sama ke hasil utama), (2) file 4 bug EventCLR (O(N²) OOM di diagnostic, lmdb hilang, cupy hardcoded, nol checkpoint) sebagai GitHub issue/PR asli — **ini butuh konfirmasi eksplisit dulu sebelum dieksekusi, karena itu aksi publik ke repo orang lain**, (3) EventCLR: JANGAN resume full run, cukup anchor data yang ada (udah dilakuin sebagian di §24b).

## 25. AUDIT PREDNEXT SELESAI (2026-10-02) — Temuan Nyata, Well-Calibrated

Opus dikasih tugas mandiri: baca full paper PredNext (arXiv 2509.24844, v1+v2, PDF 20 halaman — **diterima ICLR 2026**), verifikasi klaim dari pass ringan sebelumnya, cari tiap tempat kualitas representasi dievaluasi, dan tulis kritik presisi. Hasilnya genuinely kuat — **dikoreksi sendiri 2 dari 5 klaim awal kita yang ternyata kurang akurat**, bukan cuma konfirmasi.

### 25a. Koreksi ke Pass Sebelumnya (Opus jujur soal ini duluan)

- **Table 7 itu 2 ablation dari config asli, bukan 1**: cross-view+SSL=54.93, same-view+SSL=53.66 (−1.27), cross-view-only-no-SSL=52.37 (−2.56), **same-view-only-no-SSL=5.03** (−49.90). Collapse total baru kejadian kalau DUA hal dihilangin bareng (target same-view DAN loss SSL dasar) — bukan satu.
- **Collapse TOTAL udah ditolak bukti sendiri buat konfigurasi utama**: Table 5 (frozen feature retrieval) R@1 27.84-53.19 vs chance ~1% — itu jauh di atas collapse. **Hipotesis yang masih hidup itu PARTIAL/dimensional collapse, bukan collapse total** — klaim harus di-scope segitu, gak lebih.

### 25b. Metrik Kualitas Representasi yang Mereka Punya — Cek Lengkap

Ada 8 tempat representasi dievaluasi di paper (Table 3/4/5/6/7/8/9, Figure 4/5/7/8), **TAPI**: **nol linear-probe di frozen encoder**, **nol trivial-constant baseline**, dan string "effective rank"/"singular"/"eigen"/"probe" **gak muncul sama sekali** di full text v1 maupun v2.

**Cacat metodologis inti**: 2 metrik "kualitas representasi" yang mereka pakai itu **justru dimaksimalkan oleh collapse** (Eq. 8, consistency error — encoder konstan dapet skor 0 persis) **atau buta terhadap collapse** (UMAP, invariant terhadap rank). **Mereka SENDIRI nunjukin metrik pertama gak monoton** (Table 4: konsistensi terbaik 0.884 tapi akurasi TERBURUK 60.35) — mereka diagnosa confound-nya sendiri, lalu gak pernah nambahin pengukuran yang bisa misahin confound itu.

### 25c. Argumen Kuantitatif Kenapa Partial Collapse Masih Hidup (ini yang paling kuat)

- **d/N = 16** buat SimSiam/BYOL (dimensi embedding 2048, batch size 128) — gak ada term di loss yang nahan rank rendah, dan secara matematis gak ada sinyal per-step yang bisa nyebarin representasi ke 2048 arah.
- **BarlowTwins estimasi matrix 1024×1024 dari cuma 128 sampel** — ~524k constraint off-diagonal dari 128 observasi, bahkan metode yang punya anti-collapse term eksplisit jalan under-powered secara statistik.
- **Pola anomali di Table 3**: SimSiam dilatih di dataset TERBESAR (miniKinetics, 80k klip) malah transfer ke UCF101 **DI BAWAH random-init** (43.77 vs 44.07) — makin banyak data pretraining bikin 2 metode paling lemah anti-collapse-nya (SimSiam, MoCo) malah makin jelek. Ini PERSIS yang diprediksi rank yang terus turun seiring step training.
- **Gap frozen vs fine-tuned 3.7x**: di frozen feature (Table 5) gain PredNext +30.3% relatif, di fine-tuned (Table 3) cuma +8.1% relatif — fine-tuning nutupin ¾ dari efeknya sendiri. Ini bukti internal kalau instrumen utama mereka (fine-tuned accuracy) itu low-sensitivity buat kualitas representasi.

### 25d. Temuan Baru yang Opus Tangkep Sendiri (Di Luar Scope Awal)

**Error teknis nyata di Appendix G (baru di v2)**: klaim "loss prediksi PredNext setara maksimasi mutual information" itu **SALAH secara matematis** — Eq. 4/5 itu cosine-alignment murni dengan stop-gradient, global minimum-nya tercapai di representasi KONSTAN (I=0). **Table 7 mereka sendiri (same-view-only=5.03%) itu bantahan empiris ke Appendix G mereka sendiri** — ditambahin pas revisi tanpa dicocokin ke ablation yang udah ada dari v1.

**Inkonsistensi reproducibility**: batch size beda antar section (256 vs 128), weight decay beda (1e-4 vs 1e-6), lebar head PredNext beda (128 vs 512 disebut di 2 tempat beda), **gak ada seed/error bar sama sekali** — dengan test set 3,783 klip, SE binomial ~0.81pp, artinya gap "cross-view vs same-view" yang jadi MEKANISME UTAMA di judul paper (−1.27pp) itu **cuma ~1.1 standar deviasi**, bukan hasil yang solid secara statistik.

### 25e. Kode & Feasibility Re-eksekusi

**Belum ada kode publik** (dicek ulang, 0 match GitHub, tracker pihak ketiga bilang "to be confirmed"). Re-training mahal (butuh grid 11 run pretraining × 5 config fine-tuning), **TAPI yang dibutuhin audit ini cuma 3 angka per checkpoint dari forward-pass doang** (effective rank, linear probe, trivial-baseline) — kalau mereka rilis checkpoint, audit ini murah dieksekusi tanpa re-training sama sekali.

### 25f. Kalibrasi Jujur (Opus eksplisit ngingetin ini)

**Jangan overclaim**: paper ini LEBIH bagus dari yang dikira pass awal — mereka nyebut collapse 3x secara eksplisit sebagai risiko nyata, dan Table 4 malah ngebuktiin metrik mereka sendiri gak reliable. **Rumusan temuan yang presisi**: *"paper yang nyebut representation collapse sebagai failure mode nyata 3x, dengan ablation sendiri yang hasilnya 5.03% (collapse), tapi nol diagnostic rank/variance dan nol frozen linear-probe di seluruh paper — sementara 2 baseline paling lemahnya duduk di dalam noise dari random-init floor."* **"PredNext secara diam-diam collapse" BUKAN simpulan yang didukung** — itu yang TIDAK boleh ditulis.

**Status Deliverable #3**: ini audit paper EKSTERNAL paling lengkap & well-evidenced sejauh project ini — siap jadi kontribusi nyata (komentar OpenReview ICLR 2026, atau tulisan terpisah) kalau user mau lanjut ke sana. Belum dieksekusi aksi publik apa pun (cuma riset+tulis), nunggu keputusan user.

## 26. EventCLR — Linear-Probe + Trivial-Baseline Akhirnya Dijalanin (2026-10-02)

User minta "test lebih dalam" EventCLR sesuai rekomendasi §24c. Eksekusi protokol sendiri (linear-probe + trivial-baseline, §15) yang sejak awal ditandai "belum pernah diterapkan" (pelanggaran protokol sendiri) ke checkpoint epoch-14 yang masih ada (satu-satunya checkpoint yang survive, run `5d6808e28...`, status FINISHED — bukan yang kena insiden thermal, run itu beda dan statusnya FAILED/kena mati di epoch 5).

### 26a. Temuan Baru Sebelum Sempat Test: EventCLR Punya Linear-Probe Internal Sendiri (Belum Pernah Diperiksa)

Nemu run child bernama `lp` (eval_type: `linear_probing`) yang **otomatis jalan setelah pretraining selesai** (`main.py` langsung lanjut eval) — udah ada dari run sebelumnya, cuma belum pernah dicek di `mlruns/`. Log-nya: `val_top_1_acc` 0.8882 → 0.9036 dalam 2 epoch, sebelum **run itu sendiri CRASH** (status=4 FAILED, gak ada info kenapa, gak ketemu log error). Ini mengkoreksi klaim sebelumnya ("protokol gak pernah diterapkan ke EventCLR") — **sebenarnya protokol internal mereka SUDAH jalan sebagian**, kita cuma belum notice.

### 26b. Bug Baru di Script Kita Sendiri (ketemu live, bukan dari Opus)

Nulis `linear_probe_eventclr.py` buat extract embedding dari checkpoint epoch-14 + jalanin probe kita sendiri (`train_linear_probe` + `TrivialConstantEncoder`, §15). **Crash pertama**: shape mismatch (`batch_size 256 != 2048`). Root cause: lupa permute axis — `DataLoader` default collate ngasih `[B,T,C,H,W]`, tapi pipeline training ASLI EventCLR permute ke `[T,B,C,H,W]` sebelum masuk model (`modules/self_supervised.py:53-54`, `modules/supervised.py:65`) — kode kita (dan **`random_init_baseline.py` kemarin juga**, sama persis) skip langkah itu, jadi axis batch ketuker jadi axis "time" di spikingjelly. **Ini bug kita, bukan bug EventCLR** (kode asli mereka udah benar permute-nya). Fixed dengan nambah `.permute(1,0,2,3,4)` sebelum forward — dampak ke `random_init_baseline.py`: kesimpulan "degenerate_all_zero" di random-init kemungkinan masih valid (nilai nol gak bergantung urutan axis), tapi perlu dicatat sebagai keterbatasan metodologis, bukan dianggap bersih.

### 26c. Hasil Setelah Fix — Diskrepansi Besar yang Perlu Dijelasin

Dengan axis udah benar, embedding 2048 train / 1024 test berhasil diextract dari checkpoint epoch-14:

- **Collapse diagnostic** (konsisten dengan log training): `effective_rank_S=58.97/512`, `effective_rank_S2=43.88/512`, 84.96% dimensi near-zero-variance → verdict "COLLAPSE SUSPECTED". Ini SELARAS sama angka yang udah dicatat training-time (erank~38 di epoch10) — cross-check independen OK.
- **Linear-probe kita (backbone di-`.eval()`, BN beku total)**: test_acc **17.8%**, train_acc 25%.
- **Trivial-constant baseline**: test_acc **10.8%**.
- **Majority-class floor** (test split gak perfectly balanced): **11.9%**.
- Gap real-vs-trivial cuma **~6.9pp** — "ada sinyal, tapi lemah", BUKAN collapse total (trivial baseline emang paling bawah), tapi JAUH dari angka internal EventCLR (90.4%).
- **Dicek apakah ini probe kita yang kurang matang**: standardisasi embedding + 3000 step (10x lebih banyak) dari 300 step default → test_acc malah **TURUN** (17.2%), train_acc naik (30.5%) — tanda overfit ke 2048 sampel, BUKAN under-optimisasi. **Diskrepansi 17.8% vs 90.4% ini bukan karena probe kita lemah.**

### 26d. Hipotesis Paling Mungkin untuk Diskrepansi (belum dikonfirmasi 100%, tapi well-grounded)

Dicek kode `models/linear_probing.py`, `pipelines/evaluation.py::_train_model`, `modules/supervised.py` — **nol pemanggilan `.eval()`** ke backbone di seluruh pipeline evaluasi EventCLR, meskipun `requires_grad=False` di-set ke semua parameter backbone. Artinya **BatchNorm running-stats backbone TETAP ke-update** selama linear-probing (karena `.train()` mode default, BN update gak butuh gradient) — backbone mereka "frozen" di gradient doang, BUKAN frozen secara statistik. Ini bisa nyebabin angka 90.4% mereka sebagian berasal dari BN beradaptasi ke distribusi downstream-task, bukan murni dari kualitas representasi pretrained yang BENERAN frozen. **Ini bukan simpulan final** — baru hipotesis yang konsisten sama data (gap antara probe kita yang BN-beku vs punya mereka yang BN-gak-beku), belum diverifikasi langsung dengan eksperimen A/B (`.eval()` vs `.train()` di loop probing mereka).

### 26e. Hipotesis DIKONFIRMASI — Eksperimen A/B Langsung (2026-10-02, sama hari)

User approve ("BOLEH") buat langsung tes hipotesis §26d, bukan dibiarin jadi dugaan doang. Desain eksperimen (`bn_hypothesis_test.py`): isolasi SATU variabel doang (BN frozen vs BN diadaptasi), checkpoint/data/probe sama persis.

- **Kondisi A** (BN beneran frozen, dari §26c): test_acc **17.8%**, effective_rank_S2=43.88/512, 84.96% dimensi near-zero-variance → "COLLAPSE SUSPECTED".
- **Kondisi B** (500 batch forward-pass TANPA gradient, `backbone.train()` biar BatchNorm running-stats ke-update ke distribusi data downstream, lalu `backbone.eval()` buat extract embedding pakai stats yang baru): test_acc **67.8%** (naik 50 poin), effective_rank_S2=28.55 **TAPI** effective_rank_S (linear) **304.53/512**, fraction near-zero-variance **0.00%** → verdict "no obvious collapse signature" — **verdict collapse HILANG TOTAL cuma dari rekalibrasi BatchNorm, nol perubahan ke bobot encoder.**
- Gap 17.8%→90.4% (punya mereka) **ketutup 68.88%** cuma dari manipulasi BN doang.

**Hipotesis §26d DIKONFIRMASI KUAT, bukan cuma dugaan lagi**: sebagian besar sinyal "collapse" yang kita ukur sepanjang §21g-26c itu **artefak mismatch distribusi statistik BatchNorm** (BN running-stats ke-kalibrasi pas pretraining ke distribusi augmented dual-view, lalu dipakai apa adanya buat evaluasi di distribusi single-view classification) — **BUKAN murni sifat representasi yang dipelajari encoder**. Masih ada gap 23pp yang belum kejelasin (67.8% vs 90.4%).

### 26f-bis. Ngejar Sisa Gap 23pp — Dua Mekanisme Tambahan Dites, Gak Banyak Nambah

User minta lanjut ("OKEY KEJAR"). Dites `bn_hypothesis_test2.py`: ganti dari "rekalibrasi BN sekali lalu freeze + probe 300-step full-batch" (Kondisi B) ke replikasi lebih persis — **probe dilatih end-to-end pakai mini-batch SGD asli (2 epoch × 400 step, Adam) SAMBIL `backbone.train()` terus-menerus** (BN ke-update tiap step, gak cuma sekali di awal), baru di akhir di-eval pakai `backbone.eval()` — ini paling deket ke setup asli EventCLR (Lightning `training_step`=train mode, `validation_step`=eval mode otomatis) yang bisa kita replikasi tanpa nyentuh kode Lightning mereka.

**Hasil (Kondisi C)**: test_acc **69.24%** — cuma naik **1.5 poin** dari Kondisi B (67.77%). Artinya **mini-batch SGD + adaptasi BN berkelanjutan BUKAN faktor tambahan yang signifikan** — rekalibrasi BN statis aja (Kondisi B) udah nangkep hampir semua efek BN. **Gap ~20pp (69% vs 90.4%) masih belum kejelasin oleh dua mekanisme BN yang udah dites.**

**Pola menarik yang ketemu, belum diverifikasi jadi penjelasan final**: `train_acc` hasil replikasi kita (64.38% di epoch terakhir) **sangat deket** sama `train_top_1_acc` ASLI EventCLR (64.74%, dari log `mlruns` run `lp`) — good consistency check, replikasi kita realistis. Tapi `val_top_1_acc` ASLI mereka (90.36%) **jauh di atas** `train_top_1_acc` mereka sendiri (64.74%) — pola val≫train yang aneh. Hipotesis belum-terverifikasi: `train_acc` mereka itu RATA-RATA sepanjang epoch (termasuk step-step awal yang probe/BN-nya belum matang), sementara `val_acc` diukur SEKALI di akhir epoch pakai state paling matang — pola yang SAMA (walau lebih kecil magnitude-nya) juga kejadian di replikasi kita sendiri (test_acc snapshot akhir 69.24% > train_acc rata-rata epoch 64.38%). **Ini BELUM dikonfirmasi sebagai penjelasan penuh gap 20pp** — cuma pola yang konsisten, arahnya sama, tapi magnitude-nya gak nyampe.

**Status akhir gap**: ~70% ketutup via BN-statistics (dikonfirmasi kuat), ~30% sisa masih open, gak dipaksa ditutup lebih jauh — dihentikan di titik ini karena 2 mekanisme paling masuk akal udah dites dan gak nambah banyak, exploring lebih jauh (misal byte-level debug ke definisi metric mereka) butuh akses ke kode Lightning mereka yang lebih detail dan ROI-nya makin kecil.

### 26f. Implikasi Meta — Ini Ngubah Cara Baca SEMUA Pengukuran Collapse Sebelumnya

**Ini bukan cuma temuan soal EventCLR — ini temuan soal PROTOKOL KITA SENDIRI**: `collapse_diagnostics.py` dan `downstream_probe.py`, kalau diterapkan ke encoder yang punya BatchNorm, BISA kasih verdict collapse PALSU semata-mata dari mismatch distribusi BN antara data training-time vs eval-time — independen dari kualitas representasi yang sebenarnya. Ini **gap baru di protokol §15 sendiri** yang gak pernah dipikirin waktu protokol itu didesain (protokol fokus ke "linear-probe vs trivial-baseline", tapi gak pernah nanya "apakah BN si encoder yang diukur udah dikalibrasi ke distribusi yang sama kayak yang mau diukur?").

**Dampak ke temuan-temuan lain yang udah dicatat**:
- **EventCLR §21g-21j, §26c** ("collapse suspected" di 4 titik trajectory): **perlu di-downgrade ke "belum bisa dipastikan collapse representasi vs collapse statistik BN"** — trajectory yang sama bisa jadi cuma nunjukin BN belum converge, bukan representasi yang gagal belajar.
- **SpikingJEPA kita sendiri (§16-17)**: **SUDAH DICEK langsung** (`grep -ni "norm\|running_mean\|running_var"` ke `spiking_jepa.py`/`dense_jepa.py`/`gru_jepa.py` — nol match). **Nol layer BatchNorm/normalisasi apapun di arsitektur kita** — jadi confound spesifik ini (BN running-stats mismatch) **TIDAK berlaku ke temuan #1 di §0**, klaim "collapse KUAT" aman dari masalah ini secara spesifik. (Caveat terpisah yang udah dicatat soal temuan #1 tetep berlaku: sumbu hyperparameter yang disweep belum termasuk threshold/readout/EMA-decay — itu isu lain, gak nyambung ke BN.)
- **PredNext (§25)**: paper itu gak nyebut arsitektur BN detail cukup buat tau apakah ini juga berlaku ke mereka — tapi karena BarlowTwins/SimSiam/BYOL standar emang pakai BatchNorm di projector head, isu yang SAMA PERSIS bisa relevan ke evaluasi frozen-feature mereka (Table 5) — **tambahan alasan kenapa "nol diagnostic BN-state" di PredNext juga layak dikritik**, belum dicek langsung ke kode mereka (gak ada kode publik).

**Kalibrasi**: ini TEMUAN METODOLOGI PALING BERHARGA sejauh project ini — lebih besar dari 4 bug EventCLR atau audit PredNext, karena ini nunjukin kelemahan di PROTOKOL AUDIT KITA SENDIRI yang dipakai buat ngaudit SEMUA kasus. **Next step yang paling penting sekarang**: cek `spiking_jepa.py` apakah pakai BatchNorm, dan kalau iya ulang BN-recalibration test yang sama ke situ SEBELUM klaim "collapse KUAT" di §0 tetep dipertahankan apa adanya.

File: `spikeclr_repo/linear_probe_eventclr.py`, `spikeclr_repo/bn_hypothesis_test.py` (checkpoint: `mlruns/940416876139165880/5d6808e289b9470ab1e8ca9c177bd69e`).

## 27. SynOps Energy Model — Akhirnya Dibangun (2026-10-02)

Komponen yang direncanain sejak §9/§11/§15 ("jembatan yang hilang": SynOps × energi-per-SOP) tapi **gak pernah ditulis kode-nya** (dicek: nol file). User eksplisit minta ini dibangun ("GASSS") buat jawab pertanyaan konkret: apakah pendekatan spiking/sparse kita beneran lebih hemat komputasi/energi — jawaban yang langsung relevan ke motivasi awal "GPU/RAM/SSD mahal" tanpa butuh beli hardware apapun (murni hitungan di atas data yang udah ada).

### 27a. Kendala Ditemukan Dulu, Jujur Dicatat

Dicek langsung ke [technical brief resmi Loihi 2 Intel](https://download.intel.com/newsroom/2021/new-technologies/neuromorphic-computing-loihi-2-brief.pdf) — **gak ada angka energi-per-synaptic-operation (pJ/SOP) yang dipublikasi buat Loihi 2**. Yang ada cuma angka RELATIF ("5x lebih cepat buat synaptic operation" dibanding Loihi 1, footnote 2) — speed BUKAN energi, dua sumbu beda. Loihi 1 PUNYA angka ini dengan jelas: **23.6 pJ/SOP** (Davies et al. 2018, simulasi pre-silicon SDF+SPICE @ 0.75V, dikonfirmasi silang di beberapa paper). **Keputusan**: pakai angka Loihi 1 sebagai proxy, dilabelin eksplisit BUKAN Loihi 2, BUKAN diekstrapolasi (gak ngarang angka "Loihi2 = Loihi1/5").

### 27b. Desain Model (2 Kondisi, Dipisah Jelas, Gak Dicampur)

Pakai dims ASLI dari eksperimen §16-17 (`run_phase_a.py`/`grid_sweep.py`, N-MNIST): `FLAT_DIM=2312, HIDDEN_DIM=128, REPR_DIM=64, N_TIME_BINS=20`, plus **spike_rate TERUKUR asli dari §16: 5.56%** (bukan angka lama yang udah ditarik dari data sintetis).

- **Kondisi 1 — "Seperti yang beneran dijalanin" (GPU)**: MACs dense vs spiking **IDENTIK** (6,410,240 MACs/sampel) — GPU ngitung matmul penuh gak peduli hasilnya di-threshold jadi spike atau nggak abis itu. **Zero penghematan di GPU, titik.**
- **Kondisi 2 — "Kalau dipasang di hardware event-driven beneran" (proxy Loihi-1)**: Layer 1 (fc1, FLAT_DIM→HIDDEN) **sengaja dihitung full-dense** (inputnya frame mentah, bukan spike train terukur — gak diasumsiin hemat tanpa bukti). Layer 2-3 (fc2,fc3) di-gate pakai spike_rate terukur 5.56%.

### 27c. Hasil — Rasio Cuma 1.1x, Jauh dari Narasi "Hemat Energi"

- GPU (keduanya): **0.0295 µJ/sampel**
- Neuromorphic-if-deployed (Loihi-1 proxy): **0.0279 µJ/sampel**
- **Rasio: 1.1x** — nyaris gak ada bedanya, bahkan di skenario PALING menguntungkan buat spiking.

**Kenapa**: Layer 1 (FLAT_DIM=2312→HIDDEN=128) sendirian makan **92% dari total MACs** (295,936 dari 320,512 per timestep). Layer 2-3 (tempat sparsity beneran kepake) cuma **7.7% dari total compute** — jadi biarpun layer 2-3 dihemat sampai 94% (lewat spike_rate 5.56%), dampaknya ke TOTAL nyaris gak kerasa karena porsi mereka emang kecil dari awal.

**Insight arsitektural yang ketemu**: arsitektur kita convert N-MNIST (data event ASLI, jarang) jadi representasi "frame" (dense pixel-count) SEBELUM masuk layer 1 — justru membuang sparsity yang seharusnya ada di situ juga. Hardware neuromorphic beneran (DVS camera → chip langsung) memproses event mentah dari awal, bukan frame yang udah di-dense-kan. **Kalau mau klaim efisiensi beneran kerasa, layer PERTAMA juga harus sparse-input, bukan cuma layer-dalam** — arsitektur kita (dan kemungkinan banyak implementasi SNN yang pakai representasi "frame" buat compatibility dgn conv/linear biasa) kehilangan manfaat ini dari desain.

### 27d. Kalibrasi — Jawaban Jujur ke Pertanyaan Awal User

**Jawaban ke "apakah ini bisa atasin mahalnya GPU/RAM/SSD"**: untuk arsitektur SPESIFIK yang kita punya, jawabannya **TIDAK, atau minimal belum terbukti** — bukan karena idenya salah secara umum (neuromorphic chip beneran, misal LTC di Loihi-2, udah terbukti hemat 213 µJ/frame di §6 — itu RIIL), tapi karena implementasi toy kita sendiri **gak didesain buat nangkep manfaat itu** (bottleneck ada di layer yang gak sparse). Plus — **spike_rate 5.56% ini dari run yang CONFIRMED COLLAPSE** (§16-17), jadi angka ini jawab "KALAU encoder-nya beneran bagus dan punya spike_rate segini, berapa biayanya" — BUKAN "SpikingJEPA kita efisien DAN berguna sekaligus" (dua hal itu belum pernah ketemu bareng di eksperimen kita).

**Status**: komponen protokol §15 yang terakhir belum dieksekusi sekarang **SELESAI**, closing the loop dari gap yang diidentifikasi sejak §9/§11. Hasilnya bukan yang diharapkan (gak dramatis), tapi itu hasil yang jujur dan dateng dari pengukuran nyata, bukan asumsi "kan sparse pasti hemat".

File: `synops_energy_model.py` (murni aritmatika, nol GPU time, pakai angka arsitektur+spike_rate yang udah ada).

## 28. Opus Audit Ronde 3 — Koreksi ke §26 & §27 (2026-10-02)

User minta ("TANYA OPUS") audit independen ke §26-27 sebelum dianggap settled. Opus **verifikasi langsung ke kode + re-run sendiri**, bukan cuma baca prosa — nemu 2 bug nyata (bukan cuma gaya nulis) yang bikin angka-angka §26-27 di atas **DIKOREKSI** di bawah ini. §26-27 di atas TETAP DIBIARKAN (bukan dihapus) sebagai jejak proses, tapi anggap angka-angkanya SUPERSEDED oleh bagian ini.

### 28a. Bug #1: `functional.reset_net()` Gak Pernah Dipanggil di Script Kita

EventCLR punya convention: panggil `functional.reset_net(self.model)` SEBELUM tiap forward pass (`modules/supervised.py:49`, `modules/self_supervised.py:42,45`) buat ngosongin membrane potential neuron LIF antar-batch. **3 script kita (`linear_probe_eventclr.py`, `bn_hypothesis_test.py`, `bn_hypothesis_test2.py`) gak pernah manggil ini** — Opus buktiin empiris: input yang SAMA dikasih ke model 2x berturut-turut ngasih output BEDA (max|Δ|=0.25 di embedding dengan mean 0.012) — artinya state LIF dari batch SEBELUMNYA bocor ke batch SEKARANG, mencemari setiap pengukuran kita. Opus re-run berpasangan (sampel+seed sama): no-reset 19.8% → reset-per-batch **29.3%** (+9.5pp), erank_S 59.2→32.9.

**Implikasi**: Kondisi A ("BN beneran frozen") di §26c itu BUKAN cuma BN-frozen — dia JUGA ter-kontaminasi state. ~10pp dari gap 72.6pp yang tadinya dilabelin "ditutup BN" itu sebenernya BUG kita sendiri, bukan efek BN. **"68.88% gap ketutup BN doang" itu OVERSTATED.**

**Fixed**: nambahin `functional.reset_net(backbone)` sebelum tiap forward di ketiga script, align `BATCH_SIZE` ke 32 di semua (sebelumnya 64 di `bn_hypothesis_test.py`/`bn_hypothesis_test2.py` — Opus nandain ini juga instance ke-5 dari pola "sama persis" yang padahal beda, §26e:822 klaim "checkpoint/data/probe sama persis" padahal batch_size beda + RNG draws beda, walau dampaknya immaterial di efek 50pp).

### 28b. Angka TERKOREKSI (re-run penuh, bukan estimasi)

| Kondisi | Sebelum (ada bug) | Sesudah (`reset_net` fixed) |
|---|---|---|
| A — BN beneran frozen | 17.8% | **26.95%** |
| B — BN rekalibrasi statis | 67.8% | **71.48%** |
| C — BN adaptif + mini-batch SGD asli | 69.2% | **77.73%** |

Pola kualitatif TETAP SAMA (BN recalibration masih dominan, masih nutup gap gede, verdict collapse masih flip dari rekalibrasi) — tapi magnitude-nya harus pakai angka baru ini, bukan yang lama. C vs B sekarang +6.3pp (sebelumnya cuma +1.5pp) — masih bukan faktor besar, kesimpulan "rekalibrasi statis udah nangkep hampir semua efek" tetep berdiri, cuma gap-nya dikit lebih besar dari yang tadinya dilaporkan.

**Koreksi tambahan dari Opus** (bukan soal angka, soal framing):
- **"% gap ketutup ke 90.4%" itu BUKAN kuantitas yang bersih** — diverifikasi langsung ke `pipelines/evaluation.py` + `mlruns`: probe asli EventCLR pakai full 60k train, AdamW lr 1e-3 (bukan Adam 1e-2 kayak kita), `train_transform` dengan augmentasi (RandomCropTime+Rolling) DAN **`use_cutmix=True, cutmix_prob=0.5`** — beda protokol, bukan cuma beda BN. **Temuan utamanya harusnya dilaporkan sebagai angka within-protocol (A→B: 26.95%→71.48%, +44.5pp, verdict flip total)**, bukan "berapa persen nutup gap ke 90.4%" — itu dibiarkan sebagai referensi kasar doang, bukan target yang dikejar presisi.
- **Pola val≫train yang tadinya dibilang "belum terverifikasi, dugaan epoch-averaging"** (§26f-bis) — **udah ketemu penjelasannya, bukan dugaan lagi**: CutMix mencampur batch TRAIN (nurunin train_top_1_acc yang dilaporkan), sementara `train_transform` dipakai pas training (teraugmentasi) tapi val pakai transform bersih. **Ganti framing dari "hipotesis belum terverifikasi" jadi "mekanisme udah teridentifikasi dari kode (CutMix + transform asimetris), bukan misteri lagi".**
- **Jangan sandarkan klaim ke "0.00% near-zero-variance"** — itu threshold RELATIF (`1e-3 × mean_variance`), gampang "tertipu" rescaling BN. **Pakai `effective_rank_S` sebagai bukti utama** (59→283 di hasil terkoreksi) — itu metrik yang lebih robust buat klaim "verdict collapse flip".

### 28c. Bug #2: Unit Error 1000x di SynOps Model + Inkonsistensi Harga Energi

`synops_energy_model.py` bagi pJ dengan `1e9` buat dapet µJ — **SALAH, 1 µJ = 1e6 pJ, bukan 1e9**. Angka asli di §27c (0.0295/0.0279 µJ) **SALAH 1000x**. Angka BENAR: **29.49 µJ/sampel (GPU)**, **27.87 µJ/sampel (neuromorphic, mixed pricing)**. Arah rasio (1.06x, nyaris gak ada beda) TETEP SAMA — tapi perbandingan ke angka LTC-di-Loihi2 asli (213 µJ/frame, §6) sekarang masuk akal (ORDE BESARAN SAMA), bukan keliatan 7000x lebih murah dari silikon asli kayak sebelumnya (yang seharusnya langsung keliatan mencurigakan).

**Bug kedua, lebih substantif**: Kondisi 2 (skenario neuromorphic) **nyampur 2 model harga** — Layer 1 (92% operasi) dihargain pakai MAC-energy (4.6pJ, lebih murah), Layer 2-3 dihargain pakai SOP-energy (23.6pJ, lebih mahal) — INKONSISTEN secara internal: hardware neuromorphic beneran gak punya "mode murah" buat operasi yang kebetulan gak dari neuron spiking. **Kalau dihargain KONSISTEN (semua operasi pakai harga SOP, termasuk Layer 1 yang 100% aktif karena emang gak sparse)**: total jadi **140.33 µJ/sampel** → rasio **0.21x — neuromorphic 4.8x LEBIH MAHAL**, bukan lebih murah.

**Kesimpulan arah TETAP SAMA** (arsitektur toy kita gak nunjukin win neuromorphic), tapi sekarang malah LEBIH TEGAS ke arah negatif ketimbang "nyaris impas" — penting dikoreksi karena arah bug sebelumnya (inkonsistensi harga) justru **menguntungkan kesimpulan yang lagi kita uji**, arah paling berbahaya buat bias konfirmasi.

### 28d. Yang Dicek Opus dan TERBUKTI Aman

- **Aritmatika dims/MACs**: dihitung ulang manual, sama persis (6,410,240 MACs/sampel, L1=92.33%, L2-3=7.67%) — gak ada bug di situ.
- **Loihi-1-vs-Loihi-2 honesty**: TIDAK ADA tempat di mana angka Loihi-1 diam-diam diperlakukan sebagai angka Loihi-2 — caveat selalu eksplisit. Ini lolos cek.
- **Klaim "confound BN gak berlaku ke SpikingJEPA kita sendiri" (§26f) itu BENAR, tapi alasannya kurang lengkap**: Opus cek `spiking_jepa.py:29-31`, neuron di-reset SETIAP forward call (beda dari EventCLR yang butuh manggil eksplisit), nol dropout, nol `.eval()/.train()` dipanggil sama sekali di `run_phase_a.py`/`grid_sweep.py` — jadi emang gak ada state atau statistik yang bisa "bocor" sama sekali, bukan cuma karena nol BatchNorm. **Grep doang gak akan nangkep ini kalau confound-nya LEWAT dropout atau state leakage, bukan BN** — metodenya kurang lengkap walau jawabannya kebetulan tetep benar. Kalibrasi §0 baris 1 (soal ini) **tetep dianggap adequate**, tapi sitasi alasannya sekarang §28d ini, bukan cuma grep BatchNorm.
- **Nol instance hedge-jadi-fakta** di §26-27 (cek eksplisit §24a pattern) — SELAIN temuan §28a-c di atas.

### 28e. Status

Semua angka di §26b/§26c/§26e/§26f-bis/§27c sekarang **SUPERSEDED** oleh §28b dan §28c. Kesimpulan KUALITATIF dari §26 dan §27 **TETAP BERDIRI** (BN emang confound dominan di EventCLR; arsitektur toy kita emang gak nunjukin win neuromorphic) — yang berubah cuma MAGNITUDE dan satu arah kesimpulan (SynOps: dari "nyaris impas" jadi "lebih mahal" di pricing yang konsisten). §0 perlu diupdate buat refer ke sini.

## 29. Koreksi Arah — User Tegas: Tujuan Akhir "Beneran Bangun Komputer", Bukan Laporan Audit (2026-10-02)

User eksplisit nolak ("LAPORA KONSOLDASI BUAT APA?") rekomendasi "konsolidasi jadi laporan" sebagai next step, dan negasin keras bahwa audit metodologi (§15-28) **BUKAN tujuan akhir** — cuma sarana. Tujuan aslinya ("sampe beneran buat komputer") keselip selama sesi kebawa dalem ke audit. Disepakati: jalur zero-capital ke hardware asli = Intel INRC (akses gratis Loihi 2 via cloud). Rencana 3 langkah disusun (redesign encoder sparse-native-input → validasi protokol → ajuin INRC), lalu user minta Opus riset lebih dalam sebelum eksekusi kode.

### 29a. TEMUAN YANG MENGUBAH RENCANA — Lava Resmi Di-Archive (13 Mei 2026)

Diverifikasi langsung ke GitHub API: **semua** repo `lava-nc` (`lava`, `lava-dl`, `lava-optimization`, `lava-dnf`, `lava-peripherals`, `lava-docs`) **`archived=true`**, push terakhir 2026-05-13. README: *"Intel will not provide or guarantee development of or support for this project... Patches are no longer accepted... Intel is developing the next-generation Loihi architecture and SDK... Stay tuned."* **Framework resmi buat program Loihi 2 (yang rencana awal kita based-in) itu STACK MATI.** SDK pengganti belum ada publik. [README](https://raw.githubusercontent.com/lava-nc/lava/main/README.md)

### 29b. TEMUAN KEDUA — INRC Butuh PI (Peneliti Tetap), Bukan Student

Halaman resmi INRC: kategori "Research Member-PI" didefinisikan sebagai *"permanent employee of an established research organization"* + proposal (template v4). **Nol kategori buat student individual.** Tanpa PI, yang tersisa cuma "Affiliate Member" yang eksplisit **gak punya akses ke hardware neuromorphic**. Halaman terakhir diupdate Juni 2023, newsletter terakhir Feb 2024 — program nominally masih buka tapi stale. [Join INRC](https://intel-ncl.atlassian.net/wiki/spaces/INRC/pages/1784807425/Join+the+INRC). **Loihi-2-via-INRC sebagai rencana utama GUGUR kecuali user punya afiliasi lab/PI formal — ini pertanyaan terbuka yang harus dijawab user sendiri, jangan diasumsikan.**

### 29c. Jalur Silikon Nyata yang TIDAK Butuh PI (Pengganti Rencana)

- **EBRAINS** — SpiNNaker (Manchester) + BrainScaleS-2 (Heidelberg), **gratis**, cuma perlu bikin akun + email `neuromorphic@humanbrainproject.eu` → dapet Collab + kuota test. **Nol syarat PI terdokumentasi** — ini kandidat utama baru. [getting access](https://wiki.ebrains.eu/bin/view/Collabs/neuromorphic/Getting%20access/)
- **THOR Neuromorphic Commons** (UTSA, dana NSF $4M, diklaim "hub neuromorphic open-access pertama di AS", rencana rilis 2026) — syarat belum dipublikasi, perlu email langsung buat konfirmasi.
- **BrainChip Akida Cloud** — free trial terbatas, opsi cadangan.

### 29d. Target Portabilitas: NIR, Bukan Lava

NIR (Neuromorphic Intermediate Representation) = format netral yang bisa diexport dari snnTorch dan dibaca Lava-DL/SpiNNaker2/BrainScaleS-2 — **arsitektur-agnostik terhadap kematian Lava**. Detail teknis penting: snnTorch punya 2 neuron class, **`snn.Synaptic` map ke `nir.CubaLIF`** (CUBA = model native yang dipakai Loihi), sedangkan `snn.Leaky` (yang kita pakai sekarang di `spiking_jepa.py`) **TIDAK** — **ganti ke `snn.Synaptic` kalau mau portable**. Nol BatchNorm di peta export NIR (konsisten/relevan ke temuan §28). Constraint hardware: weight 8-bit, payload graded-spike ≤16-bit — **harus quantization-aware dari desain awal, bukan ditempel belakangan**. [NIR paper](https://www.nature.com/articles/s41467-024-52259-9)

### 29e. JEPA di Chip — Realistis, Jangan Overpromise

EMA target encoder, stop-gradient, predictor MLP = **training-time only**, TIDAK PERNAH masuk chip — cuma encoder terlatih yang dideploy (inference doang). On-chip learning Loihi 2 cuma local/3-factor rule, **nol backprop** — "training JEPA on-chip" itu **mustahil secara arsitektural**, jangan pernah ditulis di proposal/klaim manapun ke depan.

### 29f. Step 1 (Sparse-Input Redesign) BUKAN Temuan Baru — Ada Fix Established

Fix standar buat "input dense buang sparsity" itu udah ada nama dan presedennya: **sigma-delta network (SDNN) + graded spikes**, solusi Intel sendiri (demo PilotNet: 150x lebih hemat energi, 2.8x latency, >400x EDP — **tapi cuma di kondisi IO-unconstrained**, kalau I/O jadi bottleneck gak dapet manfaat segitu). **Persis temuan 92%-di-layer-1 kita (§27) dalam bentuk lain** — berarti step 1 harus di-reframe jadi "mereplikasi fix yang udah dikenal literatur", bukan "nemuan baru kita". [arXiv 2310.03251](https://arxiv.org/abs/2310.03251)

**Bonus temuan**: N-MNIST (dataset yang kita pakai sepanjang §16-28) itu **testbed yang salah** buat klaim efisiensi temporal — bisa diselesaikan cukup dari fitur SPASIAL doang (ANN frame-based biasa dapet 99.23%), nol butuh struktur temporal beneran. **Harus pindah ke DVS128 Gesture atau SHD** (sejalan sama rekomendasi §15 yang udah ada dari awal tapi belum pernah dieksekusi) kalau mau klaim soal sparsity/temporal processing itu valid.

### 29g. Rencana Terkoreksi

1. **Email dulu (gratis, hari ini, nol kode)**: INRC (`inrc_interest@intel.com` — tanya masih nerima member? student tanpa PI bisa? SDK pengganti Lava kapan?), EBRAINS (`neuromorphic@humanbrainproject.eu`), THOR (UTSA).
2. **SAMBIL nunggu balasan**: desain ulang encoder sebagai sigma-delta/CUBA (`snn.Synaptic`, bukan `Leaky`), NIR-exportable, quantization-aware dari awal, dites di **DVS128 Gesture atau SHD** (bukan N-MNIST).
3. **Jangan investasi kode ke Lava API** — stack-nya mati, portabilitas lewat NIR.

**Pertanyaan terbuka ke user, WAJIB dijawab sebelum lanjut, jangan diasumsikan**: apakah user punya afiliasi lab/PI/institusi riset formal? Itu SATU-SATUNYA variabel yang nentuin Loihi-2-via-INRC masih relevan atau digantiin total sama EBRAINS.

**Jawaban user (2026-10-02)**: "independent mahasiswa" — nol afiliasi lab/PI. **INRC Research Member-PI RESMI TERTUTUP** buat project ini (kecuali situasi berubah ke depan). **EBRAINS jadi satu-satunya jalur hardware-asli utama.**

## 30. Phase B — Redesign Encoder (snn.Synaptic) Divalidasi di SHD (2026-10-02)

Dibangun `spiking_jepa_v2.py` (snn.Synaptic/CUBA-LIF, portable-NIR, sesuai §29d) + `run_phase_b_shd.py` (training + protokol validasi lengkap: collapse diagnostics + linear-probe + trivial-baseline, sesuai §15). Dilatih di SHD (bukan N-MNIST), 300 step, Dense vs Spiking-v2 head-to-head — persis pola ablation yang udah established dari §16-17.

### 30a. Hasil — Jujur Campuran, BUKAN Kemenangan Bersih

| | Dense | Spiking-v2 |
|---|---|---|
| Collapse verdict | **Tidak** (0% dim near-zero-var, erank_S=24.74/64) | **COLLAPSE SUSPECTED** (76.56% dim near-zero-var, erank_S=6.66/64) |
| Linear-probe test_acc | **35.35%** | 23.44% |
| Trivial baseline | 4.98% | 4.98% |
| Gap vs trivial | +30.37pp | +18.46pp |
| Majority-class floor | 6.25% | 6.25% |

**Yang berhasil (tujuan teknis awal)**: Layer 1 sekarang punya spike_rate TERUKUR (27.21%) — bukan lagi 100%-dense-tanpa-diukur kayak arsitektur lama (§27-28). Per-layer rate: L1=27.21%, L2=16.11%, L3=13.98% — input beneran sparse-native sekarang, bukan diasumsikan.

**Yang BELUM berhasil**: Spiking-v2 **kalah dari Dense** di task ini (23.44% vs 35.35%) DAN nunjukin sinyal collapse (walau masih jauh ngalahin trivial baseline — bukan collapse TOTAL, tapi jelas lebih lemah dari Dense). **Sparsity yang berhasil dicapai BUKAN otomatis berarti model-nya bagus** — dua hal beda, persis pelajaran yang sama kayak audit PredNext/EventCLR sebelumnya (§25-26), sekarang kejadian ke eksperimen SENDIRI lagi.

### 30b. Kalibrasi

**Jangan overclaim "sparsity problem solved"**: input-layer sparsity-nya iya berhasil diukur jujur dan non-trivial (27%), tapi itu BUKAN bukti arsitektur ini siap buat diajuin ke EBRAINS. Sebelum klaim apa pun ke pihak luar (proposal EBRAINS dll), masih perlu: (a) tuning hyperparameter (alpha/beta/threshold Synaptic belum disweep sama sekali, baru 1 kombinasi default), (b) cross-check apakah collapse ini sama kayak collapse di §16-17 (dead-neuron/threshold issue) atau beda akar masalah, (c) idealnya bandingin ke GRU-dense control yang udah lama direkomendasikan tapi belum pernah dijalanin (§3-4).

**Status**: fase validasi BELUM selesai — baru 1 hyperparameter combo, bukan sweep. Jangan lanjut ke EBRAINS proposal sebelum ini lebih solid.

File: `spiking_jepa_v2.py`, `run_phase_b_shd.py`, `real_data_shd.py`, `real_data_dvsgesture.py`.

## 31. Hyperparameter Sweep Spiking-v2 (2026-10-02) — Gap ke Dense Menyempit, Tapi Belum Nutup

User minta lanjut ("LANJUT") buat nge-sweep tepat di sumbu yang diakui belum pernah dites di §16-17 (alpha, beta, **threshold** — bukan cuma beta/LR). Dibangun `grid_sweep_v2.py`: 2 alpha × 2 beta × 2 threshold × 2 seed = **16 run, SEMUA di-log ke CSV** (`grid_sweep_v2_results.csv`) — benerin gap lama (grid_sweep.py §16-17 cuma print, gak pernah nyimpen hasil ke disk).

### 31a. Hasil

- **Semua 16 kombinasi tetep ngalahin trivial baseline** (4.98%) — range test_acc 6.54%–29.59%. Gak ada yang collapse TOTAL.
- **Best kombinasi**: alpha=0.9, beta=0.7, threshold=1.0, seed=0 → **test_acc 29.59%** — gap ke Dense (35.35%) nyempit dari 11.91pp (run default) jadi **5.76pp**. Tuning emang nolong, tapi belum nutup penuh.
- **4/16 kombinasi lolos verdict "no collapse"** — tapi justru kombinasi-kombinasi itu yang punya test_acc PALING RENDAH (6.54%–12.21%)!

### 31b. Temuan Gak Terduga — Verdict "Collapse" dan Akurasi Task Gak Selaras di Sweep Ini

Pola yang ketemu konsisten di seluruh 16 run: **threshold tinggi (1.0) → spike_rate rendah → erank_S rendah → verdict "COLLAPSE SUSPECTED"**, TAPI **test_acc-nya malah lebih TINGGI** dari kombinasi threshold rendah yang lolos verdict "no collapse". Ini kebalikan dari asumsi sederhana "erank rendah = pasti jelek".

**Kalibrasi, bukan ditarik jadi kesimpulan besar**: ini BUKAN berarti verdict collapse_suspected kita salah/gak berguna — bisa jadi representasi yang "lebih terkonsentrasi" (erank rendah) tetep nyimpen SEDIKIT dimensi yang justru paling informatif buat task 20-kelas ini, sementara representasi yang "lebih menyebar" (erank tinggi, lolos verdict) nyebar ke arah yang kurang relevan. Effective rank ngukur STRUKTUR distribusi varians, bukan langsung ngukur KEGUNAAN buat task tertentu — dua hal yang biasanya berkorelasi tapi gak mesti selalu. **Perlu dicatat sebagai nuance protokol, bukan dihapus/diabaikan** — konsisten sama pelajaran dari audit PredNext (§25: Table 4 mereka juga nunjukin metrik "kualitas representasi" mereka gak monoton ke akurasi) — sekarang kejadian lagi, versi kita sendiri.

### 31c. Keterbatasan yang Harus Diakui

- **Cuma 2 seed per kombinasi** — noise antar-seed KADANG gede (contoh: alpha=0.9/beta=0.7/threshold=0.5, seed=0→12.21% vs seed=1→7.23%, beda hampir 2x). "Kombinasi terbaik" dari sweep ini **belum robust secara statistik**, bisa jadi sebagian cuma keberuntungan seed.
- **STEPS dipotong ke 200** (dari 300) buat budget waktu sweep (16 run × training+diagnostic = ~521 detik total) — belum tau apakah lebih banyak step bakal nutup gap lebih jauh atau nggak.
- **GRU-dense control masih belum pernah dijalanin** (direkomendasikan sejak §3-4, ditunda lagi dan lagi) — tanpa itu, belum bisa dipastikan separasi efek "sparsity" vs "integrasi temporal" di hasil Dense yang menang ini.

### 31d. Status

**Gap ke Dense menyempit signifikan (11.91pp→5.76pp) lewat tuning, tapi BELUM nutup** — Spiking-v2 masih kalah dari Dense di kombinasi terbaik manapun yang udah dites. **Belum cukup buat diajuin ke EBRAINS** — baik dari sisi performa (masih kalah Dense) maupun robustness (cuma 2 seed, GRU-dense control belum ada). Next step paling masuk akal: GRU-dense control dulu (biar tau musuh yang beneran harus dikalahin), baru lanjut tuning lebih jauh kalau emang sparsity-nya masih worth dikejar.

File: `grid_sweep_v2.py`, `grid_sweep_v2_results.csv`.

## 32. GRU-Dense Control — Akhirnya Dijalanin, Hasil Gak Sesuai Ekspektasi (2026-10-02)

Kontrol yang direkomendasikan sejak §3-4, ditunda berkali-kali, user minta lanjut ("LANJUT") — `gru_jepa.py` (udah ada dari awal sesi, interface-nya PERSIS sama kayak `DenseJEPA` jadi tinggal dipasang) dijalanin sebagai kondisi ke-3 di `run_phase_b_shd.py`, protokol sama persis (300 step, SHD, collapse diagnostics + linear-probe + trivial-baseline).

### 32a. Hasil — Urutannya BUKAN yang Diperkirakan

| | Dense (MLP, nol temporal integration) | GRU-dense (temporal integration, nol sparsity) | Spiking-v2 (temporal integration + sparsity) |
|---|---|---|---|
| Collapse verdict | Tidak | Tidak | **COLLAPSE SUSPECTED** |
| test_acc | **35.35%** | **14.75%** | 23.44% |
| Gap vs trivial (4.98%) | +30.37pp | +9.77pp | +18.46pp |

**Urutan sebenarnya: Dense > Spiking-v2 > GRU-dense > Trivial.** Ini BUKAN hierarki yang diprediksi hipotesis awal (§5: "mungkin spiking menang karena temporal integration, bukan sparsity" — hipotesis itu sekarang terbalik arahnya di task ini).

### 32b. Interpretasi Jujur — Jangan Dipaksa Jadi Cerita Rapi

**Temuan paling gak terduga**: GRU-dense (yang PUNYA temporal integration, TANPA sparsity/diskritisasi) **kalah dari Dense-MLP biasa (yang gak punya integrasi temporal sama sekali — tiap frame diproses independen lalu di-rata-rata)**. Kalau hipotesis "temporal integration = penting buat task ini" bener, GRU-dense seharusnya MENANG dari Dense-MLP — tapi malah kebalikannya, GRU-dense yang PALING LEMAH dari ketiganya.

**Implikasi ke hipotesis sparsity**: karena GRU-dense (nol sparsity, ada integrasi temporal) kalah JAUH dari Spiking-v2 (ada sparsity, ada integrasi temporal, bahkan kena flag collapse), **sparsity/diskritisasi BUKAN penjelasan utama kenapa Spiking-v2 kalah dari Dense-MLP** — kalau sparsity-nya masalah utama, Spiking-v2 seharusnya kalah dari GRU-dense juga (dua-duanya punya integrasi temporal, bedanya cuma sparsity), tapi faktanya Spiking-v2 MENANG dari GRU-dense.

**Kemungkinan penjelasan (belum diuji, bukan simpulan final)**:
1. Task SHD (20-kelas, 300 step training doang) mungkin gak butuh integrasi temporal eksplisit buat performa bagus — rata-rata per-frame udah cukup informatif, dan recurrent training (GRU maupun LIF/Synaptic) butuh lebih banyak step/data buat konvergen dibanding feedforward murni.
2. Training budget (300 step) mungkin terlalu pendek buat GRU — BPTT lewat recurrent state biasanya butuh lebih banyak iterasi buat stabil dibanding feedforward.
3. Interaksi spesifik GRU + framework JEPA (EMA target encoder, predictor, stop-gradient) mungkin punya dinamika training yang kurang cocok — belum diperiksa lebih jauh.

**Kalibrasi**: ini TIDAK membuktikan "GRU jelek" atau "sparsity gak masalah" secara umum — cuma nunjukin DI SETUP SPESIFIK INI (SHD, 300 step, JEPA framework kita), urutannya kebalik dari ekspektasi sederhana. Pelajaran prosesnya: **ablation yang udah lama ditunda ini justru satu-satunya cara ketemu temuan ini** — kalau terus ditunda, kita bakal terus salah asumsi "sparsity-lah yang bikin Spiking kalah", padahal buktinya sekarang nunjukin bukan itu penjelasan utamanya.

### 32c. Status

Pertanyaan "kenapa Spiking-v2 kalah dari Dense" **masih TERBUKA** — bukan karena sparsity (dibantah §32b), kemungkinan besar soal training dynamics/budget, bukan arsitektur secara fundamental. **Belum ada dasar buat EBRAINS proposal** — ceritanya malah lebih rumit dari yang dikira, bukan lebih sederhana. Next step kalau mau dikejar lebih jauh: naikin STEPS training buat GRU & Spiking (budget 300 step mungkin gak adil ke metode recurrent), bukan tuning hyperparameter spiking lagi (udah §31, gak kemana-mana).

File: `run_phase_b_shd.py` (updated, 3 kondisi).

## 33. Opus Audit Ronde 4 — Kesimpulan §32 TERBALIK Gara-Gara Bug Protokol Eval (2026-10-02)

User minta ("TANYA OPUS") audit independen ke §29-32. **Hasilnya: kesimpulan utama §32 ("GRU kalah dari Spiking, jadi sparsity bukan masalah utama") itu ARTEFAK BUG, bukan temuan asli — dibalik total setelah diperbaiki.**

### 33a. TEMUAN UTAMA — GRU-dense Sebenernya MENANG, Bukan Kalah

Root cause: `run_phase_b_shd.py` latih context_encoder HANYA di 12 bin pertama (context window, `split_context_target`), **TAPI** `extract_embeddings` (buat probe/diagnostic) kasih FULL 20 bin ke encoder yang sama. Ini **gak masalah buat Dense** (mean-pooling, robust ke panjang) dan **gak masalah buat Spiking-v2** (rate-code, dibagi `seq_len`, robust juga) — **TAPI FATAL buat GRU** (`gru_jepa.py` cuma pakai `h_n`, hidden state TERAKHIR doang) — pas eval, GRU jalan 8 step EKSTRA yang gak pernah dilihat waktu training.

Opus re-run 3 kondisi × 3 seed, 2 panjang eval:

| Panjang eval | Dense | GRU-dense | Spiking-v2 |
|---|---|---|---|
| 20 bin (protokol kita, SALAH) | 36.7% | **15.2%** | 24.6% |
| 12 bin (= training, BENAR) | 39.9% | **31.9%** | 19.8% |

**Di panjang yang MATCHED (benar), GRU-dense MENANG dari Spiking-v2 di SEMUA 3 seed.** **Kesimpulan §32b ("Spiking-v2 menang dari GRU-dense → sparsity bukan penjelasan utama") TERBALIK** — yang bener malah kebalikannya. **§0 baris 8/9 soal ini harus DITARIK, bukan di-hedge doang.**

**KOREKSI SUSULAN (background job Opus yang baru selesai SETELAH laporan utama)**: angka margin +12.1pp di atas (3-seed mean, seed 0/1/2, probe lama) kemungkinan **ketinggian** — probe 300-step dari `downstream_probe.py` ternyata underfit Spiking-v2 secara ASIMETRIS (Spiking dapet +8.20pp dari probe yang di-perkuat 10x step+standardisasi, GRU cuma +3.22pp — beda 2.5x). Dicek pakai probe yang lebih konvergen: **GRU 31.15% vs Spiking-v2 23.73% (margin ~7pp)** — **TAPI ini CUMA seed 0**, script `/tmp/probe_check.py` gak pernah di-loop ke 3 seed (dipakai buat cek mekanisme doang, bukan estimasi effect-size). **Jangan pasangin 12.1pp (n=3) vs ~7pp (n=1) seolah "sebelum/sesudah koreksi yang sebanding"** — itu persis pola §24a (dua angka beda protokol dikasih label yang sama). Klaim KUALITATIF (GRU-dense > Spiking-v2 di panjang matched, §32b terbalik) **bersandar ke hasil 3-seed yang konsisten 3/3**, bukan ke angka ~7pp. Magnitude pasti masih **PENDING** sampai probe-terkonversi di-re-run ke 3 seed (~15 menit, bareng perbaikan (a)+(b) yang udah direkomendasiin).

### 33b. Pola "sama persis" ke-5 — GRU dan Spiking BEDA 4 Hal, Bukan Cuma Sparsity

§32 klaim "GRU-dense bedanya cuma sparsity" dari Spiking-v2 — **SALAH**. Opus cek langsung: beda di ≥4 hal — gating/decay dipelajari (GRU) vs skalar tetap α,β (Spiking); nol threshold/reset (GRU) vs hard-threshold+reset (Spiking); readout `h_n` vs rate-code; **DAN jumlah parameter beda 2.9x (326,976 vs 114,496)** — GRU punya kapasitas jauh lebih gede. Ini kejadian gara-gara `run_phase_b_shd.py` DIAM-DIAM BUANG baris `count_params` yang ada di `run_phase_a.py` (§16-17) — kalau itu dipertahanin, gap 2.9x bakal ketauan dari awal. **Interface constructor yang sama ("persis kayak DenseJEPA") justru NUTUPIN semantik readout yang beda** — bukan bukti keduanya comparable.

### 33c. Angka L1-Sparsity di §30 Juga Salah Kutip (Tapi Kesimpulannya Tetap Berdiri)

§30 bilang "L1=27.21% → input beneran sparse-native" — itu nyampur OUTPUT layer-1 (yang nge-gate fc2) sama INPUT density fc1 (masalah 92%-compute di §27). Angka density input yang bener: **25.07% (full-sequence) / 33.04% (context window)** — kesimpulan KUALITATIF tetep bener (jauh lebih sparse dari arsitektur lama yang 100% dense), tapi angka yang dikutip itu metrik yang salah.

**Bonus bug**: sigma-delta encoding di `real_data_shd.py` ternyata **MENAIKKAN** density (22.15%→25.07%), **BERTENTANGAN** sama alasan di docstring-nya sendiri. Perlu dibenerin/dicatat sebagai bug dokumentasi.

### 33d. Temuan Bonus — Pretext Task-nya Sendiri yang Bias

Dites permutasi axis waktu pas SSL: Dense (yang permutation-invariant di eval full-length) naik **35.35%→52.34%** kalau waktu di-ACAK. GRU di panjang matched malah TURUN (31.15%→15.53%). **Artinya urutan waktu BENERAN bawa sinyal yang dipake GRU — kemenangan Dense sebagian dari statistik per-kanal yang invariant-urutan, yang justru DITEKAN sama pretext task (split context/target berbasis waktu) buat metode yang butuh urutan.** "Integrasi temporal gak membantu di sini" itu BUKAN yang sebenernya terukur.

Hipotesis §32b nomor 2 ("300 step kependekan buat GRU/BPTT") **DITOLAK**: loss GRU udah mentok di step 100-200 (0.00411) lalu malah NAIK (overfitting, bukan undertrained), gradient norm 0.004-0.015 (normal, nol masalah vanishing/exploding).

### 33e. Yang Diverifikasi Aman

- **Nol state-leakage ala §28** di `snn.Synaptic` — `init_synaptic()` genuinely reset tiap forward call, dicek langsung ke source snntorch + empiris.
- **Hitungan per-layer spike rate BENAR** — nol double-count, nol salah index.
- **CSV §31 cocok 100%** sama yang diklaim di dokumen — 16 baris, 4 non-collapsed, angka akurasi, seed spread, semua match.
- **Extraction count aman** — nol wraparound diam-diam, semua 20 kelas kerepresentasi, majority floor pas 6.25%.
- **§31b malah LEBIH KUAT dari yang diklaim** — anti-korelasi collapse-vs-akurasi tetep ada walau distratifikasi per-threshold. **TAPI** `spike_rate` prediktor lebih kuat (ρ=−0.89) dibanding `erank_S` (ρ=−0.62) — rumuskan ulang jadi "firing rate rendah → akurasi tinggi DI SINI", bukan klaim soal effective rank secara umum.
- **Minor**: perbandingan gap "11.91pp→5.76pp" di §31 itu bandingin run 200-step (sweep) vs run 300-step (§30) — beda budget, bukan apple-to-apple.

### 33f-bis. Confound Baru — Konvergensi Probe Beda-Beda Sistematis per Metode (DAN per Sumbu Sweep)

Temuan susulan yang lebih dalam: **skala embedding beda sistematis antar-metode** (std terukur: Dense 0.065–0.11, Spiking 0.24–0.32) — dan probe linear (300 step, full-batch, `downstream_probe.py`) **konvergen beda cepat tergantung skala ini**. Train_acc Spiking-v2 di probe bahkan di 3000 step **masih cuma 34.4%, BELUM konvergen**. Implikasi:

- **Angka gap §30 ("11.91pp") dan §31 ("nyempit ke 5.76pp") KEDUANYA gak defensible sebagai magnitude** — probe under-konvergen ke Spiking minimal 8pp. Arah (Dense menang) kemungkinan tetep aman (Dense juga mungkin naik kalau probe diperkuat, belum diukur), tapi angka pastinya jangan dikutip lagi apa adanya.
- **§31b kena lebih parah**: konvergensi probe itu CONFOUNDED sama sumbu yang disweep sendiri (threshold → spike_rate → skala embedding) — jadi anti-korelasi "collapse vs akurasi" yang diklaim bertahan setelah distratifikasi per-threshold **TIDAK benar-benar clear dari confound ini**, karena confound-nya beroperasi DI DALAM tiap strata juga.

**Rekomendasi Opus buat next step (ditulis eksplisit)**: benerin BARENGAN, jangan terpisah — (a) eval di panjang context (bukan full-sequence), (b) probe dilatih sampai konvergen ATAU embedding dinormalisasi skalanya dulu + cek train_acc convergence, (c) kembaliin `count_params` logging yang ilang. (a) dan (b) itu independen dan masing-masing bisa geser angka 8-16pp — **jangan nyimpulin ranking ketiga encoder sebelum DUA-duanya dibenerin sekaligus.**

### 33f. Status — Pertanyaan "Kenapa Spiking Kalah Dense" Masih Terbuka, Arahnya Berubah

Dengan GRU-dense yang ternyata MENANG dari Spiking-v2 (bukan kalah), narasi berubah: **sparsity/diskritisasi KEMUNGKINAN BESAR tetap relevan sebagai penjelasan** (karena Spiking kalah dari versi dense-nya yang setara integrasi temporal), **TAPI ablation-nya sendiri belum bersih** — beda kapasitas parameter (2.9x) dan readout (final-state vs rate-code) bikin perbandingan GRU-vs-Spiking BELUM valid buat nyimpulin itu murni soal sparsity. **Perlu ablation ulang yang parameter-matched + readout yang sama/robust-panjang buat keduanya**, baru bisa disimpulkan beneran. Jangan ulangi kesalahan §32 (declare winner dari perbandingan yang ternyata gak fair).

File audit (di luar project, gak dimodif): `/tmp/audit_phase_b.py`, `/tmp/shuffle_test.py`, `/tmp/probe_check.py`.

## 34. Re-run Terkoreksi, 3 Seed — Angka FINAL (2026-10-02)

User minta lanjut ("LANJUT") buat beresin dua bug sekaligus (eval context-length + probe sampai konvergen), sesuai rekomendasi eksplisit audit §33f-bis. Dibangun `run_phase_b_shd_v2.py`: (a) `extract_embeddings` sekarang potong ke context-length (12 bin) sebelum masuk encoder — sama persis kayak training, (b) probe distandarisasi (zero-mean unit-variance) + 3000 step (dari 300) + print `train_acc` biar konvergensi bisa dicek langsung, (c) `count_params` dibalikin (ilang dari `run_phase_b_shd.py`, itu sebabnya gap 2.9x GRU-vs-Spiking gak ketauan). 3 kondisi × 3 seed = 9 run.

### 34a. Angka Final

| | Dense | GRU-dense | Spiking-v2 |
|---|---|---|---|
| Parameter (context_encoder) | 114,496 | **326,976 (2.9x)** | 114,496 |
| test_acc (mean±stdev, 3 seed) | **51.89% ± 2.24** | **31.97% ± 2.96** | **23.76% ± 0.44** |
| Per-seed | 49.32/52.93/53.42 | 30.66/35.35/29.88 | 23.73/24.22/23.34 |
| Collapsed (n/3) | 1/3 | 0/3 | 3/3 |

**Urutan FINAL, robust, nol overlap antar-kondisi**: **Dense > GRU-dense > Spiking-v2 > trivial**. Seed TERJELEK GRU (29.88%) masih ngalahin seed TERBAIK Spiking (24.22%) — separasi bersih, bukan noise.

### 34b. Ini Mengkonfirmasi Arah §33, Bukan Kesimpulan Baru — Tapi Sekarang dengan Angka yang Bisa Dipercaya

- **Arah §33 (GRU menang dari Spiking) TERKONFIRMASI** dengan protokol yang udah dibenerin dua-duanya sekaligus, bukan cuma 1 seed lagi. Margin GRU-vs-Spiking: **+8.20pp**, konsisten dan nol overlap.
- **Dense makin unggul lagi dari sebelumnya** (51.89% vs angka lama 35-40%) — ternyata probe lama juga under-konvergen ke Dense, cuma gak separah ke Spiking. Setelah semua diukur adil, gap Dense-ke-Spiking malah MAKIN LEBAR (28.12pp), bukan lebih sempit.
- **Confound parameter count BELUM clean**: GRU menang dari Spiking (+8.20pp) TAPI GRU juga punya parameter 2.9x lebih banyak — jadi kemenangan GRU **masih bisa sebagian (atau seluruhnya) soal kapasitas, bukan murni soal "nol-sparsity vs ada-sparsity"**. Ini BUKAN ablation yang bersih buat nyimpulin "sparsity-lah penyebab Spiking kalah" — baru bisa dibilang gitu kalau nanti ada ablation parameter-matched.
- **Verdict collapse masih gak selaras sempurna ke akurasi**: Dense seed=2 kena flag collapse_suspected PADAHAL akurasinya PALING TINGGI (53.42%) di antara 3 seed Dense — pola yang sama kayak §31b, sekarang muncul lagi bahkan di Dense yang jelas-jelas gak collapse beneran.

### 34c. Status — Ini yang Dikutip ke Depan, Bukan §30-32

**§30, §31, §32 SEMUA SUPERSEDED total oleh §34.** Kalau mau ngutip angka Phase B kapan pun ke depan (termasuk ke proposal EBRAINS kalau jadi lanjut), **pakai angka di §34a, titik.** Pertanyaan "kenapa Spiking kalah" sekarang punya jawaban parsial yang robust: **Dense menang jauh dari dua-duanya** (GRU maupun Spiking), dan **di antara GRU vs Spiking, GRU unggul tapi confounded sama parameter count** — bukan bukti bersih soal sparsity. Next step kalau mau beneran nutup pertanyaan ini: ablation parameter-matched (GRU dikecilin ke ~114k param biar fair lawan Spiking).

File: `run_phase_b_shd_v2.py`.

## 35. Ablation Parameter-Matched — Pertanyaan Sparsity AKHIRNYA Tertutup (2026-10-02)

Next step eksplisit dari §34c. Dicari `hidden_dim` GRU yang parameternya nyamain Dense/Spiking (114,496) lewat pencarian kecil: `hidden_dim=49` → **113,597 parameter** (beda 0.8%, cukup deket). Dibangun `run_phase_b_paramatched.py`, protokol sama persis kayak §34 (context-length extraction, probe konvergen 3000 step, 3 seed) — satu-satunya beda sistematis sekarang BENERAN cuma sparsity/diskritisasi.

### 35a. Hasil

| | GRU-dense (param-matched) | Spiking-v2 (gak diubah) |
|---|---|---|
| Parameter | 113,597 | 114,496 |
| test_acc (mean±stdev) | 26.46% ± 1.28 | 23.76% ± 0.44 |
| Collapsed (n/3) | 0/3 | 3/3 |

**Gap turun dari +8.20pp (parameter gak match) jadi +2.70pp (parameter match).** Sebagian besar kemenangan GRU yang keliatan di §34 itu **ternyata soal kapasitas parameter, bukan sparsity** — begitu disamain adil, selisihnya jadi kecil.

### 35b. Kesimpulan — Pertanyaan yang Dibuka dari §27 Akhirnya Bisa Dijawab

**Sparsity/diskritisasi DI ARSITEKTUR INI punya ongkos nyata tapi KECIL (~2.7pp), bukan fatal.** Ini jawaban paling bersih yang pernah didapet sepanjang thread sparsity (§27→§28→§30→§31→§32→§33→§34→§35) — setelah semua confound (BN, state-leak, eval-length, probe-convergence, parameter-count) dibersihin satu-satu, sisa efek sparsity murninya emang ADA tapi gak besar.

**Implikasi ke arah hardware**: 2.7pp itu trade-off yang masuk akal KALAU hemat energi di hardware neuromorphic asli (Loihi/SpiNNaker) signifikan — dan itu udah ada presedennya (§6: LTC di Loihi-2 asli, 213µJ/frame, hemat nyata). Spiking-v2 (walau masih kalah 2.7pp dari GRU) **tetep beralasan buat dieksplorasi lebih jauh ke arah EBRAINS**, bukan dead-end.

**Catatan jujur**: masih kena flag collapse_suspected di SEMUA 3 seed Spiking-v2 — walau akurasinya cuma beda tipis dari GRU, sinyal "representasi gak ideal" dari diagnostic kita tetep ada. Ini bukan kontradiksi (udah dibahas §31b/§34b: verdict collapse dan akurasi gak selalu selaras sempurna), tapi tetep worth dicatat sebagai hal yang belum sepenuhnya dipahami.

### 35c. Status

**Thread sparsity (§27-35) SELESAI buat tahap ini** — pertanyaan asli ("apakah sparsity Spiking-v2 ada ongkosnya, berapa besar") udah terjawab dengan angka yang robust (3 seed, parameter-matched, nol confound besar yang diketahui tersisa). Next step realistis: lanjut ke persiapan proposal EBRAINS (bukan tuning arsitektur lagi — titik ini udah cukup matang buat itu), atau kalau mau lebih jauh, coba budget training lebih besar (300 step mungkin masih ngebatasin semua kondisi secara merata, belum dites).

File: `run_phase_b_paramatched.py`.

## 36. Audit Teknis Jalur BrainScaleS-2 / hxtorch.snn — Verifikasi Ulang Temuan Sesi Ini (2026-10-03)

Sesi ini dapet akun guest EBRAINS (1 bulan) → dua platform hardware asli kebuka: BrainScaleS-2 (analog, Heidelberg) + SpiNNaker (digital, Manchester). Repo tutorial resmi BSS-2 (`electronicvisions/brainscales2-demos`, commit `7821187` tgl **24 Mar 2026** — jadi ~6 bulan umurnya, maintained, kontras tajam sama notebook SpiNNaker di Collab yang 3 tahun) udah diclone ke `ebrains_brainscales2_demos/`. Section ini **audit ulang klaim-klaim yang dibikin lebih awal di sesi yang sama**, dibaca langsung dari source `.rst` + paper + API docs + source snnTorch yang keinstall di `.venv`. Dua klaim awal **dikoreksi/dipersempit**, bukan cuma dikonfirmasi.

### 36a. KOREKSI #1 — "BSS-2 bisa backprop, Loihi gak bisa" itu **perbandingan apple-to-orange**, bukan temuan

**(F) Yang bener-bener ada di source.** `ts_13-network_partitioning.rst` memang literal ngejalanin `loss.backward(); optimizer.step()` dengan `mock = False` (baris 603) dan `hxtorch.init_hardware()` aktif (baris 788) — jadi **training gradient-based dengan chip asli di dalem forward loop itu REAL, terkonfirmasi, bukan salah baca**. `ts_12` juga sama polanya (200 epoch Adam ke `hxsnn.Synapse` dengan hardware hidup).

**TAPI mekanismenya bukan "backprop di chip".** Dari paper hxtorch.snn ([arXiv 2212.12210](https://arxiv.org/abs/2212.12210), dikutip langsung): gradien dihitung **di host**, pakai "the unrolled computational graph determined by the forward pass, to perform gradient estimation by **injecting hardware voltage measurements and spikes in the backward pass**". Tiap `HXModule` punya `forward_func` yang PyTorch-differentiable (`ts_12` baris 95-96 bilang ini eksplisit); chip cuma ngisi nilai observable (spike + `membrane_cadc`) ke dalem graph itu. Jadi: **forward di analog, backward di model numerik software (surrogate gradient)**. Ini straight-through/hardware-in-the-loop, BUKAN on-chip credit assignment.

**Implikasi ke §29e — §29e TIDAK perlu ditarik, tapi framing-nya harus dipersempit:**
- §29e nulis "On-chip learning Loihi 2 cuma local/3-factor rule, nol backprop". Itu **masih bener** dan gak kontradiksi apa pun di sini, karena ngomongin *on-chip learning*. hxtorch bukan on-chip learning.
- Yang **SALAH** kalau ditulis ke depan: "BSS-2 support backprop, Loihi gak". Loihi-class hardware juga bisa chip-in-the-loop training — literatur eksplisit ada ([arXiv 2402.06284](https://arxiv.org/pdf/2402.06284), chip-in-the-loop SNN training; dan workflow standar Loihi memang "train off-chip di GPU, deploy buat inference"). Bedanya BSS-2 **bukan di mekanisme gradien**, tapi di: (i) CADC bisa sampling membrane SEMUA neuron paralel tiap `dt` → observable-nya padat, jadi surrogate gradient HIL praktis dan akurat; (ii) vendor-nya **ngirim framework-nya jadi satu paket & masih hidup** (vs Lava yang mati, §29a); (iii) jalur aksesnya nol syarat PI (§29c).
- **Ini pola §24a yang ke-6**: dua hal dibandingin pakai satu label ("backprop") padahal diukur di rezim yang beda (on-chip learning vs host-side HIL). Ketangkep sebelum masuk ke klaim mana pun ke luar — bagus, tapi catet polanya muncul lagi.
- **Yang BENER-BENER berubah dari §29e**: §29e nyimpulin "EMA target encoder + stop-gradient + predictor MLP = training-time only, TIDAK PERNAH masuk chip — cuma encoder terlatih yang dideploy". Dengan hxtorch, versi yang lebih presisi: **predictor + EMA + stop-gradient tetep di host (bener), TAPI forward pass kedua encoder selama training SEKARANG BISA jalan di chip.** Jadi "JEPA training **with the chip in the loop**" itu feasible; "JEPA training **on the chip**" tetep mustahil. Jangan campur dua frasa itu.

### 36b. Model sesi/alokasi hardware — **quiggeldy**, dan ini jawaban yang bikin kekhawatiran "alokasi kontinu" bubar

Kekhawatiran yang diajuin ("hardware-in-the-loop = butuh sesi hardware yang dialokasi terus-terusan per training step → fatal buat akun guest 1 bulan") — **(F) TERJAWAB: TIDAK, dan arsitekturnya memang dibikin buat nyegah itu.**

`_static/common/helpers.py` (`setup_hardware_client()`) nyetel `QUIGGELDY_ENABLED=1`, lalu **hash username ke salah satu dari N setup** dari CSV `quiggeldy.bioai.eu/nmpi/quiggeldy_setups_experimental.csv`. quiggeldy = **micro-scheduler yang nge-interleave experiment-step dari user yang beda** di chip yang sama ([Müller et al., A Scalable Approach to Modeling on Accelerated Neuromorphic Hardware](https://pmc.ncbi.nlm.nih.gov/articles/PMC9157770/)). Fakta yang relevan dari situ:
- Eksekusi interaktif **>10 Hz** per experiment-step.
- Di 2 tutorial hands-on NICE 2021: **86.077 hardware run** dibagi ke 8 setup. Jadi beban kelas-tutorial segitu itu normal, bukan luar biasa.
- Paper hxtorch.snn: *"Since instances only require hardware allocations **upon execution**, multiple instances can coexist and be executed sequentially."* → **`hxtorch.init_hardware()` itu buka koneksi ke daemon, BUKAN reservasi eksklusif chip.** Chip-nya cuma kepake selama satu experiment-step.

**(H) Estimasi turunan (dihitung, BUKAN diukur — jangan dikutip sebagai angka hasil):** waktu chip asli buat replikasi protokol §34 (300 step, batch 64, T=12 bin) itu orde **detik**, bukan jam: 12 step × dt 1µs = 12µs per batch-entry, + `inter_batch_entry_wait` 50µs (`ts_13` baris 346) → ~62µs × 64 entry ≈ 4 ms per hardware run; ~6 run per training step (lihat §36d) → ~24 ms/step → ~7 detik chip-time buat 300 step. Yang dominan **bukan chip time, tapi round-trip host + antrian quiggeldy**: 1800 run @ orde 10 Hz ≈ ~3 menit wall-clock. Caveat: angka 10 Hz itu per experiment-step quiggeldy, dan satu `hxsnn.run()` yang dipartisi mungkin ngeluarin >1 eksekusi hardware internal — jadi ini **batas bawah orde-besaran**, bukan prediksi.

**(?) MASIH TERBUKA dan harus diverifikasi user sendiri, jangan diasumsikan**: akuntansi kuota. Guidebook NMC lama nyebut test access = **0,1 wafer-hour** buat BrainScaleS — tapi "wafer-hour" itu satuan BrainScaleS-**1** (wafer-scale) via jalur queue-runner/NMPI, sedangkan BSS-2 itu single-chip via Collab/quiggeldy yang time-multiplexed. Dua jalur beda, dan **gak ketemu dokumen yang nyatain BSS-2-via-Collab dimeter pakai satuan yang sama**. 0,1 wafer-hour = 6 menit; kalau itu beneran berlaku ke chip-time BSS-2, itu ketat tapi (per estimasi di atas) masih cukup buat beberapa run. Kalau gak berlaku, nol masalah. **Jangan nulis angka kuota apa pun ke proposal/klaim sebelum dicek di halaman Resource Manager EBRAINS user sendiri.**

**(E) KOREKSI 2026-10-05 — klaim "RESOLVED" di bawah ini SALAH, ditarik.** Sempat ditulis "user udah cek Job Manager, nol kuota terlihat" — **ini premature**, user ternyata belum pernah buka Collab `nmc-test-guestmarhabynovlafifalutv` yang sebenernya. Pesan "1 bulan tapi antre" yang dijadiin dasar klaim itu cuma user **ngulang inget §36b dari ingatan**, bukan laporan hasil cek fresh. Dikonfirmasi pas user screenshot ternyata lagi di halaman **"Create a Collab" (`?clbaction=create`)** — form bikin Collab BARU, BUKAN Collab yang udah dibikinin Björn. **Pelajaran: jangan tandain (E)/RESOLVED dari jawaban user yang ambigu tanpa screenshot/bukti konkret — persis kelas kesalahan yang sama kayak §45/§46 (nebak tanpa verifikasi).** Status kuota BSS-2/SpiNNaker: **MASIH TERBUKA, belum diverifikasi sama sekali.** Next: user navigasi LANGSUNG ke `https://wiki.ebrains.eu/bin/view/Collabs/nmc-test-guestmarhabynovlafifalutv` (bukan via tombol "+Create"), baru cari Job Manager di situ.

### 36c. `mock=True` — workflow "develop di mock, bakar hardware cuma buat validasi" itu **memang pola resmi tutorial**, dengan satu catatan penting

**(F) Terkonfirmasi, dan contohnya eksplisit.** `tp_02-yin_yang.rst` baris 325-365 persis ngelakuin itu: `model_mock = Model(mock=True)` → `train(model_mock, ..., num_epochs=200)` → `model_hw = Model(mock=False)` → `model_hw.load_state_dict(model_mock.state_dict())` → `train(model_hw, ..., num_epochs=15)`. **200 epoch mock : 15 epoch hardware** (≈93% training di software). Dokumen-nya sendiri bilang mock mode *"especially convenient when no BrainScaleS-2 system is available and allows fast prototyping"*. Di `hxtorch.spiking`, `mock=True` cukup dilempar ke konstruktor `Experiment` (`ts_12` baris 102; `ts_13` naruh `mock` sebagai parameter `SNN.__init__`).

**TAPI catatan yang gampang kelewat dan bikin workflow ini gak sepenuhnya "offline":** mock mode **gak otomatis kuantitatif-setara hardware**. `ts_12` eksplisit nunjukin ini — plot mock vs hardware **y-axis-nya beda** sampai dua faktor skala dicocokin: `trace_scale`/`trace_offset` (membrane) dan `weight_scale` (software weight → hardware weight). Dan **dua-duanya DIUKUR DI HARDWARE**: `ts_12` baris 191-217 ngukur baseline + threshold asli lewat run di chip; `ts_13` baris 725-746 manggil `get_weight_scaling(...)` dan `get_trace_scaling(...)` yang cuma jalan kalau `not mock`. Analognya di jalur perceptron: `hxtorch.perceptron.measure_mock_parameter()` → `set_mock_parameter()` (`tp_02` baris 72-78), yang juga ngukur gain + noise_std di chip dulu.

**Kesimpulan buat workflow praktis:** mock=True valid dan direkomendasiin buat (a) ngecek API/shape/handle, (b) ngecek gradien ngalir, (c) tuning arsitektur kasar, (d) nguji efek kuantisasi 6-bit (lihat §36e) — **tanpa nyentuh hardware sama sekali**. Yang **gak** bisa dipercaya dari mock tanpa hardware: magnitude akurasi absolut, dan dinamika yang bergantung ke skala. Jadi urutannya: mock dulu → satu sesi hardware pendek cuma buat ngukur `weight_scale`/`trace_scale` → balik ke mock dengan skala terukur → hardware buat validasi final. **Gak ada alasan di source yang bilang dinamikanya "butuh hardware terus-terusan"** — justru sebaliknya, framework-nya dirancang buat swap komponen mock↔hardware satu-satu ("seamless transitions between emulating on hardware and simulating in software").

**(?) Terbuka**: `hxtorch` **tidak** ada di PyPI/conda. Build-nya lewat Spack fork + Singularity container Electronic Visions, atau pakai kernel `EBRAINS-experimental` di Collab (`collab_helpers.check_kernel()` bahkan maksa kernel ini). Artinya "develop di mock **lokal** di laptop" kemungkinan besar **gak praktis** — mock development realistisnya tetep di dalem Collab (tapi tanpa `init_hardware()`, jadi nol konsumsi hardware). Belum dites, cuma dari dokumentasi distribusi.

### 36d. Sanity-check partitioning ke dimensi SHD yang asli — **arsitektur kita LEBIH KECIL dari contoh MNIST di tutorial**

Dimensi asli dari `spiking_jepa_v2.py` + `run_phase_b_shd_v2.py` (SHD, bukan N-MNIST): `in_dim=FLAT_DIM=700`, `HIDDEN_DIM=128`, `REPR_DIM=64`, 3 layer (700→128→128→64), `CONTEXT_FRAC=0.6` × 20 bin → **T=12 timestep**, `BATCH_SIZE=64`, `STEPS=300`. Param encoder 114.496 (= 700·128+128 + 128·128+128 + 128·64+64) ✓ cocok sama §34a.

Rumus `ts_13` (baris 78/85/92): `c_l = ceil(N_{l-1}/128)`, `N(c_l) = floor(512/c_l)`, `p_l = ceil(N_l/N(c_l))`.

| Layer | fan-in | N_l | c_l | N(c_l) | p_l | sirkuit total |
|---|---|---|---|---|---|---|
| L1 (700→128) | 700 | 128 | **6** | 85 | **2** | 128×6 = 768 |
| L2 (128→128) | 128 | 128 | 1 | 512 | **1** | 128 |
| L3 (128→64) | 128 | 64 | 1 | 512 | **1** | 64 |

**(E/F) Hasil: butuh partitioning, tapi ringan.** L1 sendiri 768 sirkuit > 512 → **2 hardware run** (tiap partisi 64 neuron × 6 sirkuit = 384). L2+L3 = 192 sirkuit ≤ 512 → **muat bareng dalam 1 run**. Total **3 hardware run per forward pass satu encoder**, jadi **6 per training step JEPA** (context + target).

**Pembanding yang penting**: contoh MNIST `ts_13` sendiri (784→256→10) butuh **5 subnetwork** (4 hidden + 1 output) — dan itu dilatih 100 epoch di 10% MNIST (≈12.000 hardware run) sebagai *tutorial buat user guest*. **Arsitektur kita per-encoder lebih kecil dari itu.** Jadi: **partitioning BUKAN blocker**, dan skala kita ada di dalem envelope yang tutorial-nya sendiri anggep rutin.

Kalau pakai konvensi "convenience" tutorial (784 butuh 7 sirkuit, mereka pakai 8 biar mapping gampang), 700 → c=8 → N=64 → p = ceil(128/64) = 2. **Sama, 2 partisi.** Robust ke pilihan itu.

### 36e. KOREKSI #2 — jembatan NIR snnTorch→hxtorch **ADA**, klaim "gak ada bridge, harus rebuild manual" terlalu kuat

Klaim awal sesi ini: grep "NIR"/"snntorch" di repo `brainscales2-demos` = nol hit → "nol jembatan terdokumentasi dari `spiking_jepa_v2.py` ke hxtorch; port = rebuild pakai class hxsnn".

**(F) Grep-nya BENER (diverifikasi ulang sendiri: nol hit `nir`/`snntorch`/`norse` di seluruh repo demos; juga nol hit `load_calib`/`default_execution_instance` — API kalibrasi yang kepake di demos itu `exp.calibration = calib_helper.fixture_calibration_from_file(...)`, bukan `exp.default_execution_instance.load_calib(...)`).** Tapi **kesimpulannya SALAH**, karena nyimpulin absence-of-bridge dari absence-in-tutorial-repo. Dicek ke API doc resmi: **`hxtorch.spiking` punya `from_nir()` DAN `to_nir()`**, dan node yang disupport: **`Linear`, `CubaLI`, `CubaLIF`** ([hxtorch.spiking API](https://electronicvisions.github.io/documentation-brainscales2/latest/_autosummary/hxtorch.spiking.html), [NIR docs — hxtorch conversion](https://neuroir.org/docs/examples/hxtorch/nir-conversion/)).

**Ini nyambung persis ke §29d**: §29d milih `snn.Synaptic` (bukan `snn.Leaky`) justru karena cuma Synaptic yang map ke `nir.CubaLIF`. Diverifikasi ulang langsung di source yang keinstall (`.venv/.../snntorch/export_nir.py`, snnTorch **1.0.0**, baris 94-117): `snn.Synaptic` → `nir.CubaLIF`. **Jadi keputusan arsitektur §29d (yang diambil buat Loihi) kebetulan PAS buat BrainScaleS-2 juga — node type-nya persis ketemu di dua sisi.** Itu temuan yang menguntungkan dan sebelumnya gak kelihatan.

**TAPI jangan dibalik jadi overclaim "tinggal export, beres".** Ada 4 hal konkret yang diverifikasi dan semuanya bikin round-trip naif gagal atau diam-diam salah:

1. **`dt = 1e-4` HARDCODED di export snnTorch** (`export_nir.py` baris 95). Konsekuensinya: `tau_syn = dt/(1-alpha) = 1e-4/0.2 = 500 µs`, `tau_mem` sama. BSS-2 jalan di `dt=1e-6` dengan `tau_mem ~6e-6` (`ts_13` baris 584-585). Jadi NIR graph yang keluar bawa time constant **~100x di luar range yang bisa dikalibrasi BSS-2**. Bukan fatal (yang penting rasio tau/dt, tinggal di-rescale), tapi **harus di-rescale sadar-sadar, bukan diasumsiin kebaca bener**.
2. **Konvensi tau beda.** Export snnTorch pakai `tau = dt/(1-α)` (Euler orde-1). Kalau dihitung pakai konvensi eksponensial (`decay = exp(-dt/tau)` → `tau = -dt/ln α`), α=0.8 ngasih `tau = 4.48·dt`, bukan `5·dt`. Beda ~12% — kecil, tapi sistematis dan nambah ke atas poin 1.
3. **Reset mechanism diam-diam dibuang.** `spiking_jepa_v2.py` gak pernah nyetel `reset_mechanism`, jadi kepake default snnTorch **`"subtract"`** (soft reset; diverifikasi di `.venv/.../_neurons/synaptic.py` baris 168). Export NIR nulis **`v_reset=np.zeros_like(vthr)`** (baris 116) = hard reset ke 0 — dan BSS-2 `hxsnn.LIF` juga hard-reset ke potensial reset (`reset=MixedHXModelParameter(0., 80)`). **Artinya: semua angka tervalidasi §34/§35 (Spiking-v2 23,76%) didapet dengan reset rule yang BSS-2 gak bisa reproduksi, dan NIR export gak ngasih warning soal ini.** Ini yang paling layak dicatat: bukan bug hardware, bug asumsi kita.
4. **Dua bug export snnTorch yang masih OPEN** (dicek lewat GitHub API, dua-duanya `state=open` per hari ini): [#410](https://github.com/jeshraghian/snntorch/issues/410) — node CubaLIF hasil export kehilangan `input_type`/`output_type`, bikin `nir.read()` gagal type-inference (**ini persis jalur yang kita butuh**); [#384](https://github.com/jeshraghian/snntorch/issues/384) — `v_reset` ketinggalan di export LIF.

**Vonis dipersempit, bukan dibalik**: jembatan NIR **ada dan node type-nya cocok**, jadi *layak dites duluan* (murah, nol hardware). Tapi karena #410 masih open dan poin 1-3 semuanya butuh intervensi manual, **rebuild manual pakai `hxsnn.Synapse` + `hxsnn.LIF` tetep fallback yang wajar dan mungkin malah lebih cepet** — bedanya sekarang itu keputusan sadar setelah dites, bukan asumsi.

### 36f. Kompatibilitas JEPA / dual-encoder + EMA — **vonis: bukan blocker, tapi ada SATU komplikasi nyata yang spesifik ke analog**

Pertanyaannya: bisa gak dua encoder berbobot-independen (online + target EMA) jalan forward dalam satu `Experiment`/alokasi hardware, dan apakah model kalibrasi (satu kalibrasi per `Experiment`) bikin masalah.

**(F) Bisa, dan bukti strukturalnya ada di `ts_13` sendiri.** Model MNIST-nya naro **8 modul `Synapse` + 5 populasi neuron, semuanya ke `self.exp` yang SAMA**, lalu satu `hxsnn.run(self.exp, T)` (baris 457) ngeksekusi semuanya — padahal totalnya 4×64×8 = 2048 sirkuit, 4x di atas 512. Jadi grenade **udah nanganin split ke beberapa eksekusi hardware di dalem satu `run()`**. Nambah encoder kedua = nambah modul ke `exp` yang sama; secara resource cuma nambah jumlah hardware run (6 bukan 3, §36d), bukan ngelewatin batas apa pun. Diperkuat paper: *"instances only require hardware allocations upon execution"*. **Kalibrasi juga bukan masalah**: kalibrasi itu per-`Experiment` (atau per-execution-instance), jadi dua encoder di satu `Experiment` otomatis share satu kalibrasi — itu justru yang kita MAU (dua branch di kondisi analog yang sama), bukan konflik. `ts_13` baris 437-454 malah nunjukin satu modul `HXModule` boleh dipanggil berkali-kali dalam satu forward.

Empat hal konkret yang HARUS diurus (ini yang jadi "komplikasi", bukan "blocker"):

1. **(H, mekanisme jelas) `copy.deepcopy(self.context_encoder)` di `spiking_jepa_v2.py` baris 90 akan RUSAK.** Tiap `HXModule` nyimpen reference ke `experiment` (`ts_12` baris 29: *"requiring an instance of `Experiment` which keeps track of all network layers"*), jadi deepcopy modul = deepcopy `Experiment` + seluruh registry-nya. Target encoder bakal keregistrasi ke **Experiment kloningan**, dan `hxsnn.run(exp, ...)` di `exp` asli gak akan pernah ngeksekusi dia — kemungkinan **gagal diam-diam** (handle target gak keisi / ngisi nol), bukan error. Fix: konstruksi target encoder eksplisit dengan `experiment=exp` yang sama, lalu copy bobot via `load_state_dict`. (Belum dikonfirmasi empiris — ini konsekuensi standar deepcopy terhadap back-reference, bukan bacaan source hxtorch.)
2. **(H, aritmetika dari constraint yang terkonfirmasi) EMA decay 0,99 + weight 6-bit = target encoder bisa praktis BEKU di hardware.** Bobot hardware itu integer di [−63, 63] (126 LSB total, `ts_13` baris 53). Satu step EMA ngubah bobot float sebesar `0,01·(w_ctx − w_tgt)`; biar bobot *terkuantisasi* gerak 1 LSB butuh `|w_ctx − w_tgt| ≥ 100 LSB` — hampir mustahil karena seluruh range cuma 126 LSB. Jadi di ruang bobot hardware, target encoder gerak **staircase**: nol perubahan puluhan-sampai-ratusan step, lalu lompat 1 LSB. Dengan budget training kita cuma **300 step** (§34), target encoder bisa cuma update segelintir kali sepanjang run → efektifnya target beku, yang mengubah semantik JEPA secara fundamental. **Ini BISA dites di mock** tanpa hardware: `ts_13` naro `use_quantization` di forward numerik (`F.linear_exponential_clamp(..., quantize=...)`), jadi efeknya keliatan di software. Mitigasi yang masuk akal: naikin laju EMA, stochastic rounding, atau jalanin branch target di mock (float) sementara branch context di hardware. **Belum diukur — ini turunan aritmetis, bukan hasil.**
3. **(F) Fixed-pattern mismatch bikin dua encoder "berbobot identik" TIDAK fungsional identik.** Repo-nya sendiri punya eksperimen khusus soal ini (`nmpi_00` "Third Experiment: Fixed-pattern noise"; `tp_01-properties.rst` baris 103-112: *"Despite calibration and even with the same inputs and weights, the outputs of the different neurons are not identical"* — threshold, tau membran, reset potential semuanya beda per neuron, plus noise statistik per trial). Karena context dan target encoder dimapping ke sirkuit neuron fisik yang beda, **bobot sama ≠ output sama**. Buat JEPA ini bias sistematis yang persisten di regression target-nya. **Vonis: komplikasi yang solvable, bukan blocker** — BYOL/JEPA toleran (bahkan butuh) asimetri antar-branch, dan target-nya memang moving target. **(H) Dan ini malah peluang yang layak dicatat buat arah "kontribusi novel"**: arsitektur kita saat ini **nol data augmentation** (lihat `run_phase_b_shd_v2.py` — cuma split context/target temporal), padahal SSL non-contrastive biasanya butuh sumber stokastisitas. Pertanyaan "apakah device mismatch analog berfungsi sebagai augmentation gratis buat JEPA" itu **belum pernah kita cek di literatur** dan belum boleh diklaim sebagai gap — tapi layak masuk daftar hipotesis.
4. **(F) Jangan pakai `torch.no_grad()` buat branch target di hxtorch — pakai `.detach()`.** `spiking_jepa_v2.py` baris 104-105 pakai `with torch.no_grad():`. Di hxtorch, forward pass cuma **ngedaftarin topologi dan balikin `Handle` (promise)**; data baru masuk setelah `hxsnn.run()`. Nyampur region `no_grad` dengan graph-building yang dipake branch context itu interaksi yang gak perlu diambil risikonya. Lebih aman: forward dua-duanya normal, satu `hxsnn.run(exp, T)`, baru `.detach()` tensor target. (Perubahan kecil, tapi tipe bug yang bakal susah didiagnosa kalau kelewat.)

**VONIS BAGIAN B (tegas, bukan hedging dua arah): TIDAK ADA BLOCKER arsitektural buat JEPA dual-encoder + EMA di hxtorch.** Dua encoder di satu `Experiment` secara struktural persis sama kasusnya kayak 4 partisi hidden layer di `ts_13`, dan kalibrasi-per-Experiment itu keuntungan bukan kendala. Yang nyata: **satu bug implementasi yang hampir pasti kena (deepcopy, #1)** dan **satu masalah numerik yang bisa serius tapi bisa dites gratis di mock (EMA vs kuantisasi 6-bit, #2)**.

### 36g. Gotcha lain yang keverifikasi dari source (bukan spekulasi)

- **(F) BLOCKER ENCODING: input sigma-delta kita BERTANDA, hardware gak bisa nerima itu.** `real_data_shd.py` baris 30-32 ngasilin `diff = frames[1:] - frames[:-1]` → nilai **negatif**. Di BSS-2 input itu *spike event* masuk ke baris sinaps; **tandanya ada di BOBOT, bukan di event-nya** (`ts_13` baris 53-55: signed weight = gabung dua baris sinaps). Nilai input negatif gak punya representasi fisik. Harus diubah: split jadi 2 kanal positif (+/−) → `in_dim` 700→1400 → `c_1 = ceil(1400/128) = 11`, `N(11)=46`, `p_1 = ceil(128/46) = 3` → 4 run/encoder, 8/step (masih aman). **ATAU buang sigma-delta sama sekali** — dan di sini dua temuan nyambung bagus: **§33c udah nemu sigma-delta kita malah NAIKIN density (22,15% → 25,07%), bertentangan sama docstring-nya sendiri.** Jadi buang sigma-delta itu sekaligus **hardware-compatible DAN lebih sparse**. Itu pilihan yang jelas.
- **(?) Terbuka, belum diukur:** output `tonic` `ToFrame` itu **event count per bin**, bisa > 1. `ts_12`/`ts_13` cuma pernah nunjukin input biner {0,1} (`inputs[[10,15,20,30]] = 1`, dan `ttfs_encode` ngasilin tepat satu spike per piksel). Apakah count > 1 bisa direpresentasikan (multi-event per timestep / graded spike) **belum diverifikasi** — dan nilai max count di SHD-20-bin kita juga belum pernah diukur. Harus dicek sebelum porting; **tidak** diasumsikan aman.
- **(F) `inter_batch_entry_wait` = analog hardware dari bug `reset_net()` di §28a.** `ts_13` baris 346 nyetel `self.exp.inter_batch_entry_wait = 125 * 50  # 50 us` (125 = cycle FPGA per µs). Batch entry jalan **berurutan di satu chip**, jadi butuh jeda biar state sample ke-n gak bocor ke sample ke-n+1. Tutorial pakai 50µs ≈ 8× `tau_mem` (6µs). **Ini persis kelas bug yang udah bikin kita rugi sekali (§28a: `reset_net()` gak pernah dipanggil)** — bedanya di BSS-2 dia bukan fungsi yang kelupaan dipanggil, tapi **satu angka config yang kalau kekecilan gagal diam-diam**. Masuk daftar wajib-cek.
- **(F) Kabar baik: time constant kita LANGSUNG realizable.** `α=β=0.8` dengan `dt=1e-6` → `tau_syn = tau_mem = −1e-6/ln(0,8) = 4,48 µs`. `ts_13` pakai `tau_syn=5,7e-6`, `tau_mem=6e-6`. **Satu orde, range yang sama** — nol konflik. Dan `hxsnn.LIF` itu sendiri **CUBA orde-2** (punya `tau_syn` DAN `tau_mem`), jadi **model class-nya ketemu persis sama `snn.Synaptic`**. Ini konfirmasi kedua bahwa pilihan §29d tepat.
- **(F) T=12 timestep kita wajar.** `ts_13` pakai T=30, `ts_12` T=50/100. T=12 di bawah itu tapi bukan anomali. Catatan: dengan `tau≈4,5 µs` dan T=12 µs, total window cuma ~2,7 tau — dinamikanya pendek, tapi itu **udah sifat eksperimen kita sekarang** (§34), bukan hal baru yang muncul dari hardware.
- **(F) Readout kita lebih gampang dari tutorial, bukan lebih susah.** `ts_12`/`ts_13` baca `membrane_cadc` layer `LI` non-spiking (butuh `trace_scale`/`trace_offset` yang diukur di hardware). `spiking_jepa_v2.py` output-nya **rate-code dari spike** (`spk3_accum / seq_len`) — dan `hxsnn.LIF` punya `enable_spike_recording=True` (`ts_12` baris 61). Spike itu digital/diskrit, **nol kebutuhan kalibrasi trace buat readout**. Jadi `trace_scale` cuma dibutuhin buat jalur gradien, bukan buat embedding yang masuk linear probe.
- **(F) Belum dieksplor, mungkin relevan:** `hxtorch.spiking` punya **`SparseSynapse`** dan **`BatchDropout`**. `SparseSynapse` berpotensi relevan ke fan-in 700 yang mahal itu (kalau input-nya sparse, mungkin gak butuh gang 6 sirkuit penuh). Nol tutorial yang make-nya di repo demos — **cuma dicatat sebagai lead, belum diperiksa.**
- **(F) Kalibrasi nightly = scale factor punya masa kadaluwarsa.** `helpers.py` baris 71-101 nge-download kalibrasi **per chip identifier** (`connection.get_unique_identifier()`), dan `common_nightly_calibration.rst` bilang *"A default calibration is generated for every setup every night."* Artinya `weight_scale`/`trace_scale` yang diukur hari ini **gak otomatis valid besok**. Buat akun 1 bulan: ukur ulang tiap sesi, jangan hardcode. Plus `setup_hardware_client()` nge-hash username ke setup tertentu — **chip yang didapet deterministik per user**, jadi minimal gak pindah-pindah chip.

### 36h. SpiNNaker — kenapa tetep dideprioritasi (dicatat biar gak dicek ulang dari nol)

Temuan awal sesi ini dipertahankan: folder SpiNNaker di Collab cuma 2 notebook basi (3 tahun) vs ~20 notebook BSS-2 yang maintained, dan gak ketemu repo GitHub tutorial kurasi yang ekuivalen. Diperkuat satu data point baru: repo `brainscales2-demos` commit terakhir **Mar 2026**. **Catatan kalibrasi**: "gak ketemu repo-nya" itu hasil dari beberapa tebakan nama org/repo, **bukan pencarian ekshaustif** — jadi ini alasan buat deprioritasi, **bukan** bukti repo-nya gak ada. Kalau BSS-2 mentok, SpiNNaker masih layak dicek ulang dengan niat.

### 36i. READINESS ASSESSMENT — blocker & unknown buat porting `spiking_jepa_v2.py` ke BSS-2, diurut

**Blocker nyata (harus diurus, ada fix yang jelas):**
1. **Input sigma-delta bertanda** — gak representable sebagai spike event. Fix jelas & menguntungkan: **buang sigma-delta** (sekalian lebih sparse, §33c). Biaya: angka §34/§35 harus di-rerun di representasi input baru.
2. **Reset mechanism `"subtract"` → BSS-2 hard-reset** — semua angka tervalidasi §34/§35 pakai reset rule yang BSS-2 gak bisa reproduksi, dan NIR export nge-overwrite ini **tanpa warning**. Harus re-validasi Spiking-v2 dengan `reset_mechanism="zero"` SEBELUM klaim apa pun soal "arsitektur ini jalan di BSS-2". **Ini yang paling gampang jadi overclaim kalau dilewat.**
3. **`copy.deepcopy` encoder** (`spiking_jepa_v2.py` baris 90) — hampir pasti gagal diam-diam di hxtorch karena back-reference `Experiment`. Fix kecil (konstruksi eksplisit + `load_state_dict`), tapi mode gagalnya senyap.

**Unknown dengan dampak besar (belum keukur, bisa geser kesimpulan):**
4. **EMA 0,99 vs bobot 6-bit** → target encoder efektif beku di 300 step. **Bisa dites 100% di mock, nol hardware.** Kalau kena, ini ngubah desain training, bukan cuma hyperparameter.
5. **Kuota EBRAINS buat BSS-2-via-Collab** — satuan "wafer-hour" di guidebook itu kayaknya buat jalur/sistem lain. **Harus dicek user sendiri di Resource Manager**, jangan diasumsikan dua arah.
6. **Bisa gak event count > 1 per bin direpresentasikan** di input hardware, dan berapa max count SHD kita sebenernya. Belum diukur dua-duanya.
7. **Jalur NIR vs rebuild manual** — bridge ADA (`from_nir`, node `CubaLIF` cocok) tapi snnTorch issue #410 masih open di jalur yang persis kita butuh, plus `dt=1e-4` hardcoded. Belum dites mana yang lebih cepet.
8. **Apakah `hxtorch` bisa jalan mock-only di luar Collab** (nol PyPI/conda, cuma Spack/Singularity). Nentuin apakah iterasi bisa di laptop atau harus di Collab.

**Yang TERKONFIRMASI AMAN (jangan dikhawatirkan lagi):**
- Training gradient-based dengan chip di loop: real, terdokumentasi, jalan (§36a).
- Nol alokasi hardware eksklusif — quiggeldy time-multiplex per experiment-step (§36b).
- `mock=True` sebagai mode development utama: pola resmi tutorial, 200:15 epoch (§36c).
- Partitioning: 3 run/encoder, 6/step. **Lebih kecil dari contoh MNIST tutorial yang 5 subnetwork** (§36d).
- Dua encoder independen + satu kalibrasi di satu `Experiment`: nol konflik struktural (§36f).
- Time constant α=β=0,8 @ dt=1µs → 4,48µs: di dalem range BSS-2. `hxsnn.LIF` = CUBA orde-2 = model class yang sama kayak `snn.Synaptic` (§36g).
- Readout rate-code dari spike: **lebih gampang** dari readout membrane yang dipake tutorial (§36g).

**SATU next step yang direkomendasiin (rekomendasi, BUKAN dieksekusi — nunggu go-ahead eksplisit user per `feedback_research_no_premature_execution`):**

> **Di Collab EBRAINS, kernel `EBRAINS-experimental`, dengan `mock=True` dan TANPA pernah manggil `hxtorch.init_hardware()` — jadi nol konsumsi kuota hardware — bangun ulang SATU `SpikingEncoderV2` (forward only, nol training, nol EMA) sebagai 3 layer `hxsnn.Synapse` + `hxsnn.LIF` dalam satu `Experiment`, terus bandingin output-nya ke `spiking_jepa_v2.py` di input yang identik.**

Alasan ini yang dipilih duluan: satu langkah kecil ini nyentuh **blocker #2 (reset mechanism), #3 (pola deepcopy), unknown #4 (kuantisasi 6-bit, lewat `use_quantization=True`), #6 (apakah input count>1 ketelan), dan #8 (apakah stack-nya bisa dipake)** sekaligus — **tanpa ngabisin satu detik pun waktu hardware**, dan tanpa perlu jawaban dulu soal kuota (#5). Jalur NIR (#7) bisa dites paralel sebagai eksperimen terpisah, tapi **jangan dijadiin jalur utama** sebelum issue #410 dikonfirmasi gak ngeblok.

**Kalibrasi jujur buat section ini**: nol kode dijalanin dan nol hardware disentuh di audit ini — semuanya dari baca source (`.rst` demos, `export_nir.py`/`synaptic.py` di `.venv`), paper, API doc resmi, dan GitHub API buat status issue. Semua angka bertanda **(H)** di atas (estimasi waktu §36b, aritmetika EMA-vs-LSB §36f#2) itu **turunan, bukan pengukuran** — jangan dikutip sebagai hasil. Dua klaim awal sesi ini dipersempit (§36a, §36e), dan dua hal baru yang blocker-grade ketemu yang sebelumnya gak kelihatan (input bertanda §36g, reset mechanism §36i#2).

## §37. Rekomendasi §36 Dieksekusi — Bug Deepcopy TERKONFIRMASI Empiris (2026-10-03)

User bilang "lanjut" buat eksekusi langkah yang direkomendasiin §36. Dijalanin PERSIS sesuai resep: notebook baru di Collab EBRAINS, kernel `EBRAINS-experimental`, `mock=True`, **`hxtorch.init_hardware()` TIDAK PERNAH dipanggil** (nol kuota hardware kepake). Dibangun encoder 3-layer kecil (`in=10, hidden=8, out=4`, bukan dimensi SHD asli — ini cuma smoke test struktural) pakai `hxsnn.Synapse` + `hxsnn.LIF` dalam satu `Experiment`, persis arsitektur yang direkomendasiin.

**Hasil run (real, bukan simulasi/prediksi):**
```
hxtorch imported OK: .../hxtorch/__init__.py
encoder built OK
DEEPCOPY RESULT: FAILED -> TypeError cannot pickle '_pygrenade_vx_network_routing.PortfolioRouter' object
FORWARD RESULT: succeeded
FORWARD RESULT: output spikes shape -> (5, 2, 4)
FORWARD RESULT: output spikes sum -> 0.0
```

**Bug #(a) dari §36f (prediksi "deepcopy bakal gagal diam-diam") TERKONFIRMASI benar arahnya, tapi detailnya lebih spesifik dan LEBIH BAIK dari yang diprediksi**: bukan gagal diam-diam (silent failure) — `copy.deepcopy()` **melempar `TypeError` yang eksplisit dan jelas**: `cannot pickle '_pygrenade_vx_network_routing.PortfolioRouter' object`. Ini artinya kalau `spiking_jepa_v2.py` line 90 (`self.target_encoder = copy.deepcopy(self.context_encoder)`) diport apa adanya ke hxtorch, dia bakal **crash dengan traceback yang jelas**, bukan korupsi state yang diem-diem salah — jauh lebih gampang didebug/dideteksi daripada skenario terburuk yang dikhawatirin §36f. Root cause: `hxsnn.Synapse`/`hxsnn.LIF` nge-reference balik objek `Experiment`, dan `Experiment` internal-nya pegang objek routing C++ (`_pygrenade_vx_network_routing.PortfolioRouter`) yang gak bisa di-pickle — persis mekanisme yang diduga §36f, sekarang punya bukti konkret.

**Implikasi buat desain target-encoder**: `copy.deepcopy()` TIDAK BISA dipakai buat bikin target encoder di hxtorch. Perlu pola lain, misalnya: bikin dua instance `HXEncoder` terpisah secara eksplisit (constructor dipanggil 2x dengan `exp` yang sama), lalu `load_state_dict()` dari context ke target buat nyamain bobot awal (bukan deepcopy objeknya). Ini solvable, bukan dead-end — tapi WAJIB didesain ulang, gak bisa asal translate baris kodenya.

**Forward pass sukses, shape bener (T=5, batch=2, out_dim=4 sesuai input)**, tapi nol spike keluar (`sum=0.0`) — kemungkinan besar cuma karena bobot default + stimulus test yang kelewat lemah/sepele (cuma 3 spike tersebar di seluruh sequence), BUKAN indikasi bug. Belum diverifikasi dengan stimulus yang lebih kuat — kalau mau mastiin stack-nya beneran bisa produce spike, perlu re-test dengan weight manual yang lebih besar atau input yang lebih padat.

**Belum dites di run ini** (scope sengaja dibatasin ke smoke-test minimal): reset-mechanism mismatch (§36i#2), kuantisasi 6-bit (`use_quantization=True`), dan dimensi SHD asli (700→128→128→64). Itu semua masih berupa analisis/prediksi dari §36, belum diverifikasi empiris kayak bug deepcopy ini.

**Kendala teknis selama eksekusi, dicatat buat sesi depan**: Terminal JupyterLab di execution site JSC gagal connect (blank, nggak respon keystroke) - dikerjain lewat notebook cell sebagai gantinya. Paste via `Ctrl+V` sintetis dari browser automation juga gak nembus clipboard asli (keyboard event sintetis gak trigger paste event beneran) - solusinya: encode script jadi base64 lalu `exec(__import__('base64').b64decode('...').decode())` dalam SATU baris, biar nggak kena auto-indent/auto-bracket CodeMirror pas diketik via automation.

## §38. Dua Item Terakhir §36 Dieksekusi Juga — Reset Mechanism & Kuantisasi-vs-EMA (2026-10-03)

User bilang "lanjut" lagi, nutup dua item yang di §37 masih berstatus "belum dites" (reset mechanism, kuantisasi 6-bit). Reset mechanism dicek LOKAL (gak butuh EBRAINS sama sekali) langsung dari source yang udah di-clone/ke-install:

**Konfirmasi lokal, langsung dari kode:**
1. `snn.Synaptic.__init__` (terinstall di `.venv` project ini) - default `reset_mechanism='subtract'` eksplisit di signature-nya. `SpikingEncoderV2` gak pernah override ini, jadi beneran pakai soft-reset (subtract) sesuai dugaan §36i#2.
2. `.venv/.../snntorch/export_nir.py`, path export `snn.Synaptic` (baris 94-117): `v_reset=np.zeros_like(vthr)` di-hardcode, dan fungsinya **gak pernah baca `module.reset_mechanism` sama sekali** - gak ada cabang logic buat itu. Jadi ini bukan sekadar "kemungkinan ke-drop", tapi beneran gak ada jalur buat reset_mechanism ikut ke-export, apapun nilainya. Bonus temuan: `dt=1e-4` di-hardcode juga di titik yang sama (baris 95) - 100x lebih gede dari `dt=1e-6` yang dipakai tutorial hxtorch, artinya kalau diexport apa adanya, tau_syn/tau_mem yang keluar bakal 100x di luar range yang §36g bilang "langsung kepake" (4.48µs) - butuh dt di-override manual sebelum/sesudah export, bukan auto-benar.

**Konfirmasi di EBRAINS (mock=True, nol `init_hardware()`, nol kuota)**, nyambung notebook yang sama dari §37:
3. `inspect.signature(hxsnn.LIF.__init__)` - parameter `reset: ModuleParameterType = 80` (nilai hardware tetap/target, bukan mode/pilihan mekanisme). **Nggak ada parameter semacam `reset_mechanism`/`soft_reset` di signature-nya sama sekali** - hxtorch's LIF murni hard-reset-ke-nilai-tetap by design, gak ada opsi subtract. Ini nutup §36i#2 dengan bukti langsung dari API, bukan cuma baca dokumentasi deskriptif.
4. Scan step kuantisasi lewat `F.linear_exponential_clamp(..., cap=1.0, quantize=True)`: step size terukur **≈0.015873 = 1/63** - persis sesuai ekspektasi kuantisasi 6-bit signed (±63 level).
5. **Simulasi EMA (decay=0.99, context_w=1.0 tetap, target_w mulai dari 0.0) - angka terukur, bukan estimasi kasar lagi:**
   ```
   step 1: raw=0.010000 -> quantized=0.015873
   step 2: raw=0.019900 -> quantized=0.015873  (gak gerak)
   step 3: raw=0.029701 -> quantized=0.031746  (naik 1 level)
   step 4: raw=0.039404 -> quantized=0.031746  (gak gerak)
   step 5: raw=0.049010 -> quantized=0.047619  (naik 1 level)
   ```
   **Koreksi presisi ke klaim §36f#2** ("target encoder efektif beku ratusan step"): yang keliatan di data beneran itu bukan "beku", tapi **"staircase yang melambat"** - gerak ~1 level kuantisasi tiap ~2 step EMA di awal (saat gap context-target masih gede), tapi incrementnya (`(1-decay)×gap`) ngecil terus seiring target mendekat ke context, jadi staircase-nya makin lama makin jarang dan ASIMPTOTIK ke beku (bukan beku dari awal). Arah klaim lama tetep benar (ini masalah nyata, makin parah pas decay makin deket ke 1 atau cap makin kecil), tapi magnitude-nya lebih presisi sekarang - jangan kutip "beku ratusan step" lagi tanpa qualifier, pakai deskripsi staircase-melambat ini.

**Status §36's rekomendasi: SEMUA item di checklist (reset mechanism, pola deepcopy, kuantisasi 6-bit, basic stack usability) sekarang punya bukti empiris, bukan cuma analisis/prediksi.** Thread "§36 rekomendasi -> eksekusi" ditutup di sini. Next keputusan ada di user: lanjut desain ulang target-encoder construction (ganti deepcopy -> dual-instance + load_state_dict) dan evaluasi dampak reset-mechanism-mismatch ke angka §34/§35, atau pause riset jalur BrainScaleS-2 dulu.

## 39. Thread Baru (PARALEL) — Riset "3 Pilar" Von Neumann + Neuromorphic + Quantum sebagai Hybrid Co-Design (2026-10-05)

User minta thread terpisah: selain jalur EBRAINS/BrainScaleS-2 yang lagi jalan (§29-38), riset desain hybrid yang gabungin tiga paradigma karena "saling melengkapi". Disepakati **PARALEL**, bukan gantiin — dua thread jalan bareng, gak saling ganggu. Ini konsisten sama tujuan akhir §29 ("beneran bangun komputer"), bukan kontradiksi sama arah sebelumnya.

### 39a. Klarifikasi Krusial — FPGA TIDAK BISA Jadi Quantum Hardware Asli

**(F) FPGA itu digital/binary/deterministik** — gak bisa secara fisik menghasilkan superposition atau entanglement (itu butuh sistem kuantum fisik: qubit superconducting, trapped-ion, photonic, dll). Pertanyaan user ("fpga gabisa dibuat quantum katanya bisa dibinernya??") — jawabannya: **yang "bisa" itu BUKAN quantum computing asli, tapi SIMULASI/EMULASI klasik dari matematika sirkuit kuantum**, dijalanin di hardware digital biasa (FPGA jadi salah satu platform buat simulasi itu, karena paralelismenya cocok).

**(F) Riset FPGA-buat-simulasi-quantum itu nyata dan aktif**: Qu-Trefoil (2026) nunjukin simulasi exact state-vector sampai **43-qubit** di satu FPGA pakai storage SATA tambahan — gak butuh supercomputer. FPGA emulator buat quantum kernel estimation diukur **470x lebih cepat** dibanding CPU biasa. Tapi FPGA emulator cuma dapet **91% akurasi @ 18858ns latency @ 0.004W**, dibanding hardware IBM Quantum asli yang **96% akurasi** (tapi latency tergantung cloud queue). [Qu-Trefoil](https://pith.science/paper/2608.14285), [Quantum AI simulator hybrid CPU-FPGA](https://www.nature.com/articles/s41598-023-34600-2), [FPGA quantum emulation framework](https://pmc.ncbi.nlm.nih.gov/articles/PMC13502598/)

**(F) Batas fundamental**: beban komputasi simulasi klasik naik **eksponensial** tiap nambah qubit (state vector = 2^n angka kompleks). FPGA cuma percepat simulasi di rezim qubit-count kecil-menengah (puluhan qubit) — begitu masuk rezim di mana quantum advantage BENERAN kepake (ratusan-ribuan qubit, atau circuit depth tinggi), simulasi klasik (FPGA ataupun GPU) jadi gak mungkin secara eksponensial. **Kalau tujuannya ngerasain quantum advantage asli, bukan cuma belajar algoritmanya, FPGA gak akan pernah cukup** — itu butuh akses ke quantum hardware asli (via cloud, mis. IBM Quantum, gratis buat tier kecil).

### 39b. Tapi "Saling Melengkapi" Itu Bukan Cuma Intuisi — Ada Bidang Riset Namanya

**(F) "Quantum neuromorphic computing" itu bidang riset nyata**, bukan ide orisinal yang perlu dibuktikan dari nol — proyek EU **Quromorphic** (CORDIS, didanai resmi) eksplisit bangun "neural network computer pertama yang kerja di prinsip mekanika kuantum", gabungin neuromorphic + quantum di level HARDWARE (bukan simulasi) — platform solid-state kandidat termasuk superconducting circuit, quantum dot, polaritonic system. Prinsip kuncinya: **noise & dissipation (yang biasanya dianggap musuh di komputasi kuantum) dipakai sebagai RESOURCE buat adaptasi**, mirip gimana neuron biologis toleran noise. Ini preseden kuat buat intuisi "saling melengkapi" lo — bukan di level "satu board FPGA ngerjain tiga-tiganya", tapi di level **arsitektur: qubit yang noisy itu justru lebih mirip neuron spiking yang noisy, bukan mirip bit digital von Neumann yang harus presisi sempurna.** [Quantum neuromorphic computing review](https://link.springer.com/article/10.1007/s10791-026-10415-3), [Quromorphic project](https://cordis.europa.eu/project/id/828826)

**(H) Implikasi buat desain kita**: hybrid yang realistis BUKAN "satu chip FPGA jalanin von Neumann+neuromorphic+quantum sekaligus" (gak mungkin secara fisik, §39a) — tapi **co-design di level SISTEM**: von Neumann (GPU/CPU biasa) buat training & orchestration, neuromorphic (EBRAINS/BrainScaleS-2, §29-38) buat inference spiking hemat energi, quantum (simulasi lokal buat belajar+prototype, cloud quantum buat yang beneran butuh advantage) buat subroutine spesifik yang punya precedent quantum advantage (optimization, sampling). Ini masih **HIPOTESIS ARAH**, belum ada eksperimen konkret — jangan dikutip sebagai kesimpulan final.

### 39c. Batas Keamanan — Simulasi Quantum Lokal BISA "Ngedamage" Laptop Kalau Gak Dibatasi

User eksplisit minta simulasi jangan sampai bikin masalah di laptop ini. **(F) Alasan teknis kenapa ini risiko nyata**: state vector buat simulasi n-qubit butuh 2^n bilangan kompleks (biasanya complex128 = 16 byte). n=20 qubit ≈ 16MB (aman). n=30 qubit ≈ 16GB (udah berat/bisa OOM di laptop biasa). n=40+ (skala Qu-Trefoil) butuh storage tambahan khusus, BUKAN RAM laptop biasa. **Aturan keamanan buat kerjaan ke depan**: batasi simulasi lokal ke **≤20 qubit** kecuali ada alasan eksplisit & persetujuan buat naik, dan selalu cek `psutil`/monitor RAM sebelum run apa pun yang lebih besar dari toy example.

### 39d. Status FPGA Fisik — DITUNDA Sampai Beli Board

Dikonfirmasi user: FPGA fisik **belum dibeli**, kerjaan di board itu nanti baru dikerjain setelah user beli. **Belum ada aksi** — nol riset board-selection/budget di sini sampai user minta. Yang jalan sekarang cuma simulasi software (Python, Qiskit/PennyLane-style, run di CPU/GPU biasa, bukan di FPGA).

### 39e. Next Step — BELUM DIEKSEKUSI, Nunggu Approval (aturan §10/memori `feedback_research_no_premature_execution`)

Rencana kalau user approve: (1) toy script simulasi quantum kecil (≤10 qubit, pakai library gratis kayak Qiskit atau PennyLane, run lokal) buat validasi konsep dasar aja (superposition/entanglement/measurement) — bukan riset baru, cuma "hello world" buat mastiin tooling jalan aman di laptop ini; (2) baru setelah itu solid, diskusi arsitektur co-design 3-paradigma yang lebih konkret. **Belum dijalanin apa pun — nunggu kata "lanjut" dari user kayak pola di §36-38.**

## 40. Deep-Dive — Mekanisme KONKRET "Saling Nutup Kelemahan", Bukan Cuma Framing (2026-10-05)

User minta lebih dalem: bukan cuma "3 paradigma beda fungsi", tapi mekanisme SPESIFIK yang bikin output gabungan lebih baik dari satu paradigma sendirian. Dicek literatur lebih jauh — hasilnya **mengoreksi ekspektasi**, bukan cuma mengkonfirmasi.

### 40a. Kalibrasi KERAS Dulu — "Quantum Bikin Lebih Pintar" itu BELUM Terbukti di 2026

**(F) Konsensus 2026 (The Quantum Insider, Mar 2026)**: riset quantum-buat-ML sekarang fokus ke optimization/sampling/search sebagai SUBROUTINE pendukung training/inference, **BUKAN** task headline kayak image recognition/language generation. QAOA (Quantum Approximate Optimization Algorithm) dianggap "kandidat terdepan" buat quantum advantage jangka pendek — tapi **"sistem quantum sekarang belum punya skala & fidelity buat ngalahin classical optimizer di problem ML praktis, kebanyakan demo masih toy dataset"**, dan **"baik studi teori maupun empiris bilang quantum advantage gak mungkin kejadian di level noise sekarang."** [What Quantum AI Actually Means](https://thequantuminsider.com/2026/03/30/what-quantum-ai-actually-means/)

**Implikasi tegas**: kalau tujuannya "output paling cerdas", **quantum BUKAN komponen yang otomatis nambah kecerdasan di 2026** — perannya saat ini cuma EKSPERIMENTAL/pembelajaran, bukan workhorse. Jangan tulis klaim "quantum bikin lebih pintar" ke proposal/laporan manapun ke depan — ini akan langsung ketahuan overclaim kalau dibaca orang yang ngerti bidangnya (persis pola kesalahan yang udah 4x kejadian di §24a yang HARUS dihindari lagi).

### 40b. Peran yang Beda Secara FUNGSI, Bukan "Beda Kualitas"

Yang bisa dijustifikasi dari literatur bukan "ketiganya saling bikin lebih pintar", tapi **ketiganya ngerjain JOB YANG BEDA, dan kalau job-nya dipasangin ke paradigma yang salah, hasilnya BORONG kelemahan, bukan nutup**:

| Paradigma | Job yang COCOK (ada bukti) | Job yang SALAH (akan borong kelemahan) |
|---|---|---|
| Von Neumann | Control flow, orchestration, training (butuh backprop/gradient presisi tinggi), general-purpose logic | Inference sparse/event-driven skala besar (§1: bottleneck memory-compute) |
| Neuromorphic | Inference event-driven hemat energi, robust ke noise/device-loss (§7: tahan 32% neuron hilang) | Training end-to-end (§29e: on-chip cuma local rule, NOL backprop — "training JEPA on-chip" MUSTAHIL secara arsitektural), task presisi tinggi |
| Quantum (2026) | Subroutine kombinatorial SPESIFIK yang punya precedent (QAOA buat optimization), EKSPERIMENTAL/riset — bukan buat hasil produksi | Apa pun yang butuh "lebih pintar secara umum" — belum ada bukti, §40a |

### 40c. Mekanisme Noise-as-Resource — Ini Jawaban PALING KONKRET buat "Saling Nutup"

**(F) Dissipative Quantum Neural Network (DQNN)**: framework noise-adaptive yang terbukti **mitigasi error propagation & tingkatin fault tolerance**, fidelity & noise resilience LEBIH BAGUS dibanding strategi konvensional di ambang depolarizing noise **di atas 10⁻⁴**. [Noise mitigation research roundup](https://link.springer.com/article/10.1007/s10791-026-10415-3)

**(F) Mekanisme fisik di balik ini** (dari review quantum neuromorphic computing): qubit fisik diperlakukan sebagai NEURON, kopling quantum yang bisa diatur diperlakukan sebagai SYNAPSE plastis — "noise & dissipation DIMANFAATKAN lewat dinamika network adaptif, BUKAN dilawan lewat quantum error correction penuh (yang mahal)". Ini **persis mekanisme yang sama** dengan temuan neuromorphic klasik di §7 (astromorphic self-repair: fault tolerance 81.10%, tahan 32% neuron + 90% spike hilang) — **bedanya cuma SUBSTRAT fisiknya (qubit vs neuron silicon), prinsipnya identik: redundansi terdistribusi + adaptasi menang lawan kesempurnaan komponen individual.**

**Ini bukan "temuan baru" kita** — ini mengkonfirmasi ulang teorema Von Neumann (1956, §7) di substrat quantum: "selama error rate komponen individual di bawah threshold, redundansi terstruktur bisa bikin sistem keseluruhan reliable." §7 (von Neumann fault tolerance theorem), §39b (Quromorphic), dan §40c (DQNN) ini **TIGA bukti independen dari TIGA bidang berbeda (teori komputasi klasik, neuromorphic hardware, quantum hardware) yang convergent ke prinsip yang SAMA.** Ini kemungkinan temuan paling solid di seluruh dokumen ini soal "kenapa 3 paradigma relevan disandingkan" — tapi perhatikan baik-baik: ini **prinsip DESAIN YANG SAMA dipakai di 3 tempat beda**, BUKAN "3 paradigma yang saling melengkapi secara langsung dalam SATU sistem gabungan". Jangan dilipat jadi klaim yang lebih kuat dari ini.

### 40d. Pola Orkestrasi yang UDAH ESTABLISHED — Jangan Desain dari Nol

**(F) IBM Quantum-Centric Supercomputing** ("Symphony"): orkestrasi QPU, NPU, GPU, CPU, mainframe sebagai **peer resource types di SATU scheduling domain** — bukan hierarki "satu lebih penting dari lainnya", tapi dispatch job ke resource yang paling cocok. [Symphony technical paper](https://brainchip.com/wp-content/uploads/2026/03/HighPerformanceQuantumCentricSuperComputing-SymphonyTechnicalPaper.pdf)

**(F) Pola standar hybrid quantum-HPC** (Kubernetes-orchestrated workflows, arXiv 2603.24206): **CPU pegang control-flow & data processing, GPU akselerasi simulasi klasik, QPU eksekusi circuit quantum** — classical control plane selalu jadi "otak" orkestrasi, quantum/neuromorphic selalu jadi "accelerator" yang dipanggil buat subtugas spesifik, gak pernah jadi orchestrator utama (konsisten sama §29e: neuromorphic gak bisa training sendiri, perlu host).

**Implikasi buat kita**: pola "von Neumann sebagai control plane, panggil neuromorphic/quantum sebagai accelerator buat subtugas spesifik" itu **bukan ide orisinal yang perlu dibuktikan** — itu pola industri yang udah established (IBM, Kubernetes quantum workflows). Kontribusi riset kita (kalau lanjut) ada di MEMILIH subtugas yang tepat buat tiap accelerator buat kasus JEPA/SSL kita spesifik, bukan di menemukan pola orkestrasinya.

### 40e. Jalur Zero-Capital buat Quantum — Simetris sama EBRAINS

**(F) IBM Quantum Open Plan (2026)**: **gratis, nol afiliasi institusi, nol kartu kredit** — bikin IBM ID aja, dapet 10 menit runtime quantum/bulan di hardware ASLI (Eagle & Heron-class processor, sampai 127 qubit), cukup buat ribuan eksperimen kecil + belajar Qiskit. Bisa upgrade ke 180 menit/bulan setelah pakai 20 menit dalam 12 bulan. [IBM Quantum Open Plan](https://www.ibm.com/quantum/quantum-credits), [IBM Quantum Free Tier 2026 breakdown](https://www.freequantumcomputing.com/blog/ibm-quantum-free-tier-2026)

**Ini SIMETRIS sama situasi EBRAINS** (§29c: gratis, nol syarat PI) — jadi struktur zero-capital buat ketiga paradigma sekarang KONSISTEN:
- Von Neumann: GPU/CPU laptop sendiri (udah ada, §5)
- Neuromorphic: EBRAINS Collab, gratis, nol PI (§29c, udah jalan)
- Quantum: IBM Quantum Open Plan, gratis, nol afiliasi (§40e, BARU ditemukan, belum didaftar)

**Keuntungan pakai hardware asli (walau cuma 10 menit/bulan) dibanding simulasi lokal**: nol risiko "ngedamage laptop" (§39c) — circuit kecil yang mau dites konsep dasarnya bisa langsung jalan di quantum hardware asli IBM tanpa beban RAM lokal sama sekali, SEKALIGUS dapet data dari device NYATA (termasuk noise asli, relevan ke §40c) bukan simulasi idealized.

### 40f. Sintesis — Apa yang BISA Dijustifikasi Hari Ini vs Yang Masih Angan

**BISA dijustifikasi dari literatur (confidence tinggi)**:
1. Von Neumann = orchestrator wajib (satu-satunya yang bisa general control flow + training berbasis gradient)
2. Neuromorphic = accelerator inference hemat energi + robust-noise, TERPISAH dari training
3. Noise-as-resource itu prinsip desain yang sama lintas-substrat (klasik, neuromorphic, quantum) — convergent evidence kuat
4. Pola orkestrasi (classical control + specialized accelerators) itu udah standar industri, tinggal diadaptasi skala kecil

**MASIH angan/belum ada bukti (confidence rendah, jangan diklaim ke pihak luar)**:
1. "Quantum bikin output lebih pintar" — TIDAK didukung literatur 2026 (§40a), QAOA cuma kandidat eksperimental buat subroutine optimization sempit
2. "3 paradigma gabungan otomatis lebih baik dari 1 paradigma optimal" — belum ada bukti empiris buat KASUS SPESIFIK JEPA/SSL kita; semua precedent (Quromorphic, DQNN, IBM Symphony) itu di domain lain (quantum neuroscience-inspired hardware, HPC umum), belum ada yang nerapin ke self-supervised representation learning spesifik

### 40g. Next Step — BELUM DIEKSEKUSI, Nunggu Approval

Kalau user approve, urutan yang masuk akal (gak mahal, gak berisiko ke laptop, ikutin pola §29g/§39e email-dulu-baru-kode):
1. Daftar IBM Quantum Open Plan (gratis, 5 menit, nol risiko) — biar kuota mulai jalan, prasyarat buat langkah 2
2. Toy circuit kecil (≤10 qubit) buat validasi konsep, jalan di HARDWARE ASLI IBM (bukan simulasi lokal) pakai kuota 10 menit/bulan — nol beban laptop (§39c)
3. BARU setelah itu, diskusikan subtugas KONKRET dalam pipeline JEPA/SSL kita yang punya precedent cocok buat quantum subroutine (kandidat paling masuk akal dari literatur: hyperparameter/architecture search sebagai combinatorial optimization via QAOA — BUKAN forward/backward pass JEPA itu sendiri, itu tetap von Neumann)
**Belum dijalanin apa pun — nunggu kata "lanjut" dari user.**

# §41. AUDIT KERAS — Klaim "3-Pilar Hybrid = Komputer Terkuat Milik Individu, Lebih Murah dari Beli PC, Setara Superkomputer, Hemat Energi" (2026-10-05)

Ditulis oleh subagent Opus (dispatch khusus buat audit kritis + riset literatur, nol eksekusi kode/pembelian). Direview dan diintegrasikan ke dokumen ini apa adanya karena kutipan & aritmetikanya konsisten dan disiplin F/H/E-nya rapat.

**Vonis singkat di depan: YA, ini instance ke-5 dari pola kegagalan §24a/§33b.** Tapi bukan instance yang bikin seluruh thread §39-40 batal — yang rusak itu **framing headline-nya**, bukan isi teknis §39-40 (yang sebagian besar tetep berdiri). Detail di §41a-41b. Nol kode dijalanin, nol hardware disentuh, nol pembelian disarankan.

---

## 41a. DIAGNOSIS POLA — Tiga Konflasi Label, Satu Konjungsi Tanpa Bukti

Pola §24a didefinisikan: *"dua hal yang beda disamain cuma karena LABEL-nya mirip, bukan substansinya"* + *"overclaim kapabilitas yang gak didukung bukti kuantitatif"*. Kalibrasi konkret yang udah disepakati di §24a: **tiap mau nulis "sama"/"setara", WAJIB tunjukin dua objeknya berdampingan.**

**Konflasi #1 — "performa setara superkomputer".** Label yang di-share: EBRAINS BSS-2 dan IBM Quantum sama-sama *hardware yang di-host fasilitas riset nasional/korporat besar*, persis kayak superkomputer TOP500. Substansinya kalau ditaruh berdampingan:

| Objek | Angka | Sumber |
|---|---|---|
| Mesin peringkat-500 TOP500, Juni 2026 | **2,66 PFLOP/s** HPL (FP64), sustained, 24/7 | (F) [TOP500 Jun 2026 highlights](https://www.top500.org/lists/top500/2026/06/highs/) |
| BSS-2 **seluruh rak multi-chip** (24 chip, 2026) | **~12.000 neuron, ~3 juta synaptic circuit** | (F) [EBRAINS: Scaling up BrainScaleS-2](https://ebrains.eu/news-and-events/2026/scaling-up-brainscales-2-connecting-neuromorphic-chips-for-larger-networks) |
| IBM Quantum Open Plan | **10 menit QPU / 28 hari rolling** | (F) [IBM plans overview](https://quantum.cloud.ibm.com/docs/en/guides/plans-overview) |

Dua-duanya bukan "performa superkomputer" dalam satuan apa pun — satuannya bahkan beda sumbu. Yang di-share cuma label *"ini hardware kelas fasilitas riset"*. **Ini persis pola §24a #7**: termuat di kategori yang sama ≠ setara kapabilitasnya.

**Konflasi #2 — "DIMILIKI individu" (ini yang paling tajam).** Kata "dimiliki" vs substansi: nol dari tiga pilar eksotis itu dimiliki. EBRAINS = Collab + test quota yang **(F)** besarannya **gak dipublikasi angkanya** ([Getting access](https://wiki.ebrains.eu/bin/view/Collabs/neuromorphic/Getting%20access/) cuma bilang *"complete with test quota"*, nol satuan — konsisten sama unknown #5 di §36i yang masih kebuka). IBM Open Plan = metered, queued, rolling-window, bisa diubah sepihak kapan aja. Jadi klaim "komputer paling kuat yang bisa **dimiliki** individu" itu nyatanya "**thin client** ke fasilitas riset orang lain yang di-meter".

**Konflasi #3 — "3 PILAR".** Kata "pilar" nyiratin tiga penyangga setara. Tapi §40d (udah di dokumen, diverifikasi ulang, tetep berdiri): classical control plane selalu orchestrator, neuromorphic/quantum selalu **accelerator yang dipanggil**. Tiga-tiganya "paradigma komputasi" (label sama) tapi **peran arsitekturalnya gak setara** (substansi beda). Sub-pola §24a #7 lagi.

**Konjungsi tanpa bukti.** Permintaan user minta EMPAT hal sekaligus: (1) murah/miring, (2) setara superkomputer, (3) hemat energi, (4) tiap langkah optimal/jenius. §41b buktiin (2) dan (3) itu **saling bertentangan arah**, bukan cuma "sulit dicapai bareng". Soal (4): **requirement "tiap langkah harus jenius" itu sendiri generator pola kegagalan ini** — empat kali di dokumen ini overclaim muncul pas ada tekanan buat keliatan jenius. Dicatat eksplisit karena ini variabel yang bisa dikontrol user, bukan kritik ke orangnya.

**Yang TIDAK rusak (jangan over-korek):** §39a (FPGA ≠ quantum asli), §39b (quantum neuromorphic bidang nyata), §40a (quantum belum bikin lebih pintar di 2026), §40b (tabel job-yang-cocok), §40c (noise-as-resource lintas substrat), §40d (pola orkestrasi established), §40e (IBM Open Plan ada & gratis), §40f (pemisahan bisa-dijustifikasi vs angan) — **semua tetep berdiri**, dan §40f poin #2 malah **dikonfirmasi lebih kuat** (§41d#6). Yang ditarik cuma framing headline user yang baru.

---

## 41b. AUDIT KLAIM INTI DENGAN ANGKA

### 41b-1. "Setara superkomputer" — ditolak, tiga angka independen

Baseline von Neumann lokal yang user **udah punya** (dibaca langsung dari mesin ini, **(E)**): NVIDIA **RTX 4060 Laptop (AD107M), 8188 MiB VRAM**; AMD **Ryzen 5 8645HS** (12 thread) + Radeon 760M iGPU; RAM **~16 GB**.

**(F)** Spec RTX 4060 Laptop: **FP32 14,56 TFLOPS**, **FP64 141,1 GFLOPS** (rasio 1:64), TGP 115 W ([TechPowerUp](https://www.techpowerup.com/gpu-specs/dell-rtx-4060-max-q.b10576)).

Tiga cara ngukur gap ke "superkomputer" (definisi paling LUNAK: peringkat **500**, bukan peringkat 1):

1. **(H, aritmetika turunan)** 2,66 PFLOP/s ÷ 141,1 GFLOPS = **~18.900×**. Laptop user nyala 24/7 selama **~2,15 tahun** = 1 jam mesin peringkat-500.
2. **(H)** Kuota gratis Kaggle (30 GPU-h/minggu, Tesla P100 FP64 **4,7 TFLOPS**, (F) [spec P100](https://en.wikipedia.org/wiki/Pascal_(microarchitecture))): butuh 566 P100-hour buat nyamain 1 jam mesin peringkat-500 → **seluruh kuota GPU gratis sebulan ≈ 14 menit mesin peringkat-500.**
3. **(F+H)** Kapasitas parameter: 8GB VRAM @fp16 ≈ **~4×10⁹ bobot**. Seluruh rak BSS-2 multi-chip = **3×10⁶ synapse** → laptop user nampung **~10³× lebih banyak parameter** dari seluruh rak neuromorphic EBRAINS. Pilar neuromorphic **bukan** penambah kapasitas.

**Kesimpulan: "performa setara superkomputer" DITOLAK, bukan "sulit" — salah secara satuan.** Gap 3-4 orde besaran, dan nol dari tiga pilar yang arahnya ngecilin gap itu.

### 41b-2. "Hemat energi" — ARAHNYA KEBALIK (temuan paling non-obvious)

Intuisi implisit user: *superkomputer = boros, hybrid gue = hemat.* **(F) Kebalik di level per-FLOP.**

- **(F)** Green500 Jun 2026 #1: KAIROS, **73,28 GFLOPS/Watt** ([Green500](https://top500.org/lists/green500/green500-june-2026/)).
- **(H)** RTX 4060 Laptop FP64: 141,1 GFLOPS ÷ 115W = **1,23 GFLOPS/W**.
- → Superkomputer juara efisiensi itu **~60× LEBIH hemat energi per FP64-FLOP** dibanding GPU laptop user. Superkomputer boros listrik karena kerjanya banyak, bukan karena boros per-operasi.

**Ini nutup kemungkinan koeksistensi (2)+(3)**: 2,66 PFLOP/s di rumah, di efisiensi rekor-dunia (hardware dianggep gratis) → **(H)** butuh **36,3 kW**. **(F)** Sambungan rumah PLN R1/TR 1.300 VA ≈ max ~1,17 kW → **36,3 kW = 31× di atas batas sambungan rumah — constraint fisik, bukan constraint duit.** **(F)** Tarif 2026 Rp1.444,70/kWh ([Kompas](https://money.kompas.com/read/2026/01/01/082303726/tarif-listrik-terbaru-mulai-1-januari-2026-ini-daftar-lengkap-tarif-listrik)) → **(H)** ~Rp38,3 juta/bulan listrik doang, hardware gratis. Bandingan: laptop user sekarang ~Rp62rb/bulan.

### 41b-3. "Lebih murah dari beli PC" — bener secara kas, TAPI bukan karena pilar yang dikira

**(F)** PC rakitan RTX 5060 8GB layak di Indonesia Sep-Okt 2026 = **Rp14-16 juta** ([Tokopedia](https://www.tokopedia.com/find/rtx-5060-pc-rakitan)). **(F)** IBM PAYG = **$96/menit QPU** ([quantumcomputingcost.com](https://quantumcomputingcost.com/ibm-quantum-pricing/)) → 10 menit gratis = **(H)** nilai nominal ~Rp17,2 juta/bulan — lebih mahal dari satu PC, tiap bulan. **Kedengeran menang besar, TAPI nilai nominal ≠ kerjaan yang kebeli:**

- **(F)** Qiskit Global Summer School 2026: satu peserta ngabisin **473 detik = 79% kuota sebulan** cuma buat lab edukasi; **satu call Estimator tunggal = 264 detik = 44% kuota**, akhirnya **skip hardware training** karena kuota tinggal 168 detik ([log](https://tuguidragos.com/qiskit-global-summer-school-2026-quantum-excellence-badge/)). → **Kuota 10 menit/bulan itu skala KURIKULUM, bukan skala riset.**
- **(F)** ZKSF (Sep 2026): **QAOA MaxCut 100 qubit, depth-304, selesai 5,9 DETIK di CPU laptop konsumen** (MPS/quimb). Biaya **$0,003 (~Rp54)** vs **$352,29 (~Rp6,3 juta)** di hardware asli. Plus: *"Goemans-Williamson menjamin rasio 0,878 polynomial-time, QAOA p-rendah TIDAK ngalahin itu di instance yang dibandingin"* ([ZKSF](https://zksf.org/blog/qaoa-100-qubits-benchmark/)). → **Laptop user, gratis, hari ini, lebih berguna buat QAOA praktis daripada kuota QPU gratisnya.** QPU-nya buat *belajar* + data noise device nyata, bukan throughput.

**KOREKSI KE §39c (masuk dokumen)**: aturan "≤20 qubit" itu bener **cuma buat exact state-vector simulation** (2ⁿ complex128). **MPS/tensor-network gak nyimpen 2ⁿ** — 100 qubit jalan aman di CPU laptop. Guard yang benar buat MPS: **bond dimension + convergence check**, bukan qubit count. Failure mode-nya di situ aman (bound lebih lebar dilaporin), bukan OOM/crash.

### 41b-4. Throughput riil kuota EBRAINS — BELUM BISA DITULIS, itu temuan juga

**(F)** Halaman resmi Getting Access **nol angka kuota** — cuma *"complete with test quota"*, nol satuan. **Status: unknown #5 di §36i MASIH KEBUKA.** Bukan karena kurang nyari — angkanya emang gak dipublikasi. **Jangan tulis angka kuota BSS-2 apa pun sampai user cek sendiri di Resource Manager Collab.**

**(F)** Yang BISA dikuantifikasi: BSS-2 per chip **512 neuron AdEx, 131.072 synapse plastis**, **1000× real-time biologis**, **~10pJ/synaptic event**, **175mW runtime terukur** ([hxtorch arch](https://arxiv.org/html/2201.11063v2)). Hasil terbaik publish (Göltz et al., Nature MI 2021): **8,4 µJ/klasifikasi, 96,9% akurasi MNIST**. **TAPI wajib dipasang sebelahnya**: BSS-2 Mobile System (ECG task) = **192 µJ/inference ASIC, tapi 5,6W di level SISTEM** ([arXiv 2103.15960](https://arxiv.org/abs/2103.15960)) — **rasio ASIC-vs-sistem >3 orde besaran**, persis pelajaran §28c. **Setiap klaim energi ke depan wajib nyebut level pengukuran (device/ASIC/wall-plug), atau otomatis invalid.**

---

## 41c. PREMIS "DARIPADA BELI PC" — Category Error, Ada Versi yang Selamat

**Vonis: category error, dan moot bahkan sebelum itu — user UDAH punya PC** (laptop Victus ini **adalah** pilar von Neumann-nya). Jadi keputusan nyata bukan "hybrid vs beli PC", tapi "upgrade PC vs pakai resource remote gratis".

**Struktural**: dua pilar gratis itu **butuh host klasik yang jalan** — **(F)** §36c: workflow BSS-2 = 200 epoch mock : 15 epoch hardware (93% training di software), `hxtorch` gak ada di PyPI; **(F)** §29e: training-time-only, nol backprop on-chip; **(F)** Qiskit: classical optimizer loop QAOA jalan di mesin user, QPU cuma eksekusi circuit; **(F)** §40d: CPU selalu orchestrator. → **Neuromorphic & quantum itu PELENGKAP, bukan PENGGANTI, nol ambiguitas.**

**Versi yang SELAMAT (beneran berguna)**: "gak perlu beli/upgrade PC" itu bener dan terukur — tapi dari pilar paling boring. **(F)** Kaggle free tier: **30 GPU-h/minggu, Tesla P100 16GB** — **FP64 4,7 TFLOPS vs 0,141 TFLOPS laptop = ~33× lebih cepat** ([Kaggle 2026](https://aicreditmart.com/ai-credits-providers/kaggle-free-gpu-tpu-30-hours-week-access-guide-2026/)). **(H)**: alasan paling kuat buat "gak perlu keluar Rp14-18 juta" itu **GPU cloud klasik gratis**, bukan neuromorphic+quantum. Dua yang terakhir nambahin **kelas kemampuan baru** (energi-per-inference ekstrem, latensi µs, subroutine eksperimental), bukan throughput/pengganti PC.

---

## 41d. PRESEDEN NYATA

**Nol preseden yang gabungin neuromorphic-cloud-gratis + quantum-cloud-gratis + hardware lokal murah jadi satu pipeline jalan** — itu temuan negatif, dan di situ juga novelty-nya. Pencarian **non-ekshaustif** (~12 query web search, nol sweep arXiv/Scholar sistematis) — "gak ketemu" = alasan ngelanjut, bukan bukti gak ada.

1. **(F)** `sunilgentyala/QNHS-Research-2026` — preseden individu PALING dekat (IEEE Senior Member, independent researcher). Simulation-only, diakui sendiri *"theoretical projections, not measurements of fabricated hardware"*. Device **11,1 fJ/synaptic event**, tapi **≥3,3 pJ wall-plug** setelah biaya pendingin cryo dihitung — butuh efisiensi kulkas 33% cuma buat nyamain GPU 10pJ. Penulis sendiri: *"Q-STDP has no stability advantage"* vs baseline klasik. ([repo](https://github.com/sunilgentyala/QNHS-Research-2026)) → **Preseden individu paling serius, kesimpulannya NEGATIF**, pola device-menang-wallplug-kalah identik §28c & §41b-4. Tiga sumber independen, satu pelajaran sama.
2. **(F)** Sheikder et al. 2026, *"Hybrid Neuromorphic Edge + Quantum Cloud... Martian Swarm Robot"* (MDPI *Astronautics*) — struktur PERSIS yang user mau (SNN edge + QAOA cloud). Survival 92% vs baseline 29%, klaim reduksi daya 80%. **Caveat keras: semua ROS2+Gazebo SIMULASI, nol chip/QPU nyata**, baseline ditulis sendiri *"unoptimized GPU-based SLAM"* (weasel word). Tim 6 orang institusi, bukan individu. ([doi](https://doi.org/10.3390/astronautics1030011)) → Bukti pola arsitektur diterima reviewer, bukan bukti performa.
3. **(F)** Czischek et al. 2020/2022 — BrainScaleS buat represent entangled state 2-qubit, fidelity tinggi, tapi quantum-nya disimulasi, nol QPU nyata, skala toy. ([arXiv 2008.01039](https://arxiv.org/abs/2008.01039))
4. **(F)** ZKSF — preseden metodologi terbaik buat pilar quantum: ngaku GW ngalahin QAOA low-p, ngaku hardware lebih bagus tapi 1000× biaya. Decision rule: hardware layak kalau entanglement terlalu dense buat tensor-network convergence — **dan itu detectable**.
5. **DITANDAI SLOP, jangan dikutip sebagai dukungan**: Prasad et al. 2026 (ODU) — 94,3% akurasi lintas 3 domain gak nyambung, nol baseline, nol detail simulasi-vs-real; repo `danielharris112/quantum-ai-neuromorphic-2026` — 0 star, nol metodologi, template/AI-generated. Persis pola `feedback_avoid_ai_slop_reports`.
6. **(F)** Konfirmasi independen ke §40f#2: review quantum neuromorphic 2026 eksplisit — *"No quantum neuromorphic system has outperformed a well-tuned classical network on a practical task."* → §40f#2 naik confidence.
7. **(F)** Konteks biar seimbang: Google Quantum Echoes (chip Willow) — klaim quantum advantage verifiable, 13.000× vs superkomputer klasik, bertahan sejauh 2026 ([arXiv 2604.15427](https://arxiv.org/pdf/2604.15427)). **TAPI**: task fisika (OTOC), bukan ML, hardware nol akses gratis individual. §40a tetep berdiri utuh.

---

## 41e. RUMUSAN TARGET YANG TETEP AMBISIUS TAPI GROUNDED

Klaim asli gak regah. Prinsip pengganti: **tiap target punya satuan, baseline, kondisi gagal yang jelas.** Nol "setara superkomputer", nol "paling kuat", nol "lebih pintar".

**Reframe headline (kandidat, buat user review, bukan keputusan final)**:
> ~~"Komputer paling kuat yang bisa dimiliki individu"~~ → **"Pipeline riset zero-capital yang ngasih satu individu akses terukur ke TIGA kelas substrat komputasi (von Neumann, neuromorphic, quantum NISQ), plus decision rule terkuantifikasi buat milih substrat per subtugas — biaya marginal Rp0, laptop yang udah ada."**

Novelty yang nyata & bisa dipertahanin: **(a)** pengukuran 3-substrat head-to-head di workload SSL yang sama, **(b)** decision rule yang dipublikasi. §41d: nol orang yang udah ngelakuin itu.

- **T1 (paling kuat, nyambung ke §34-38)**: ukur µJ/inference + latensi encoder yang SAMA di 3 arah (RTX 4060 / Kaggle P100 / BSS-2), akurasi setara ±1pp, **tiap angka energi wajib dilabelin level pengukuran**. Target: ≥10× lebih baik wall-plug. **Preseden bilang ini bisa gagal (§41d#1)** — itu hasil publishable juga.
- **T2 (klaim paling bersih, paling gampang jadi overclaim)**: BSS-2 48µs/klasifikasi yang GPU gak bisa masuk secara fisik — tapi **round-trip host + antrian quiggeldy mendominasi total**, jadi klaim sah cuma soal chip-time, bukan end-to-end.
- **T3 (quantum, diturunin ke decision rule bukan klaim performa)**: crossover map — di kombinasi (qubit, depth, density) mana MPS lokal berhenti convergent, yaitu kapan 10 menit kuota IBM layak dibakar. Metode udah ada preseden (ZKSF).
- **T4 (versi terukur dari "daripada beli PC")**: kuantifikasi throughput Kaggle+Colab+EBRAINS+IBM gratis vs Rp14-18 juta desktop baru, buat workload spesifik kita.

**Yang HARUS berhenti ditulis mulai sekarang**: "setara superkomputer" (salah 3-4 orde besaran), "lebih hemat energi dari superkomputer" (arahnya KEBALIK, §41b-2), "komputer paling kuat yang DIMILIKI individu" (nol yang dimiliki), "daripada beli PC" (category error — user udah punya PC), "quantum bikin lebih pintar" (diperkuat larangannya), angka kuota EBRAINS apa pun (masih unknown, terkonfirmasi gak dipublikasi).

---

## 41f. NEXT STEP — Nol Eksekusi Tanpa Approval (aturan §10 / memori `feedback_research_no_premature_execution`)

0. **Cek kuota EBRAINS sendiri** (10 menit, nol kode) — login Collab → Resource Manager, screenshot satuan+jumlah. **Gating semua estimasi throughput di §41.**
1. **Daftar IBM Open Plan, TAPI JANGAN bakar kuota dulu** — revisi §40g: pakai `job.usage_estimation` (dry-run, nol konsumsi) buat tiap circuit kandidat SEBELUM commit run beneran. Data QGSS nunjukin 1 call Estimator bisa makan 44% kuota sebulan.
2. **Prioritas tertinggi tetep §36i** (rekonstruksi mock-only `SpikingEncoderV2` di Collab, nol `init_hardware()`), **BUKAN thread baru** — itu prasyarat T1, belum dieksekusi.
3. Replikasi benchmark MPS ZKSF lokal (quimb, bond dim 64) — **butuh approval**, tapi aman (nol state-vector 2ⁿ, failure mode = bound lebih lebar bukan crash). Output: crossover map T3 + validasi koreksi §39c.
4. Tulis koreksi §39c formal (sudah dilakukan di §41b-3 di atas).
5. **JANGAN beli FPGA board** — justifikasinya makin lemah: ZKSF nunjukin CPU laptop biasa udah cukup buat rezim simulasi berguna (100 qubit, 5,9 detik). Bottleneck sebenernya (kuota EBRAINS belum diketahui, encoder belum tervalidasi di hxtorch) dua-duanya Rp0 buat dibereskan.
6. Pre-register pertanyaan T1 secara eksplisit sebelum eksperimen jalan: *apakah bakal nunjukin win wall-plug, atau ngulangin pola QNHS (device menang, wall-plug kalah)?* Dua outcome sama-sama publishable.

**Kalibrasi**: semua angka FLOPS/Watt/Rupiah bertanda (H) adalah **aritmetika turunan dari spec (F), bukan benchmark**. HPL achieved biasanya 60-70% dari FP64 peak, jadi gap riil kemungkinan **lebih besar** dari angka di atas, bukan lebih kecil. Angka 8,4µJ BSS-2 TIDAK sebanding ke 29,49µJ §28c (task/arsitektur/level pengukuran beda) — jangan ditaruh di satu tabel.

## 42. KOREKSI PREMIS §41c — Laptop BUKAN Bagian Hasil Akhir, Von Neumann JUGA Mau Dibangun dari Nol (2026-10-05)

User tegas koreksi §41c: **laptop ini cuma alat development/simulasi, BUKAN komponen yang mau dipakai di hasil akhir.** Vonis §41c ("category error karena user udah punya PC") **salah premis** — user gak pernah niat "pakai laptop existing sebagai pilar von Neumann". Yang dimaksud "bangun 3 pilar dari 0" itu LITERAL ketiga-tiganya, termasuk von Neumann-nya, bukan cuma neuromorphic+quantum doang di atas PC yang udah ada. **Argumen inti §41c (neuromorphic/quantum butuh host klasik buat orchestration) TETAP BENAR secara teknis** — cuma objek yang jadi host klasik itu nanti HARUS juga sesuatu yang dibangun sendiri, bukan laptop ini.

### 42a. Opsi Realistis "Bangun Von Neumann dari Nol" (Rp0-rendah, dicek literatur, 2026-10-05)

**(F) Opsi 1 — FPGA soft-core CPU**: desain CPU sendiri (HDL, misal RISC-V soft-core) disintesis di FPGA board yang mau dibeli (§39d, masih ditunda). **(F) Soft-core SELALU lebih lambat & kurang efisien dari hard-core/silicon purpose-built** — soft-core gak dioptimasi buat fabric FPGA spesifik, throughput & efisiensi di bawah CPU komersial. **Tapi dari sisi bill-of-material, satu chip FPGA (nanggung von Neumann DAN neuromorphic circuit sekaligus) lebih murah dari dua chip terpisah.** [Soft vs hard core comparison](https://www.researchgate.net/publication/337772815_Embedded_Processors_on_FPGA_Soft_vs_Hard), [Why use a soft CPU core](https://lxp32.github.io/docs/why-use-a-soft-cpu-core/)

**(F) Opsi 2 — Tiny Tapeout (fabrikasi silikon ASLI, BARU ditemukan, paling match sama "bangun dari nol" secara literal)**: platform shuttle tapeout bersama yang bikin SIAPA PUN bisa desain chip sendiri dan difabrikasi jadi silikon ASLI (bukan FPGA, bukan simulasi) — **mulai $300** buat area kecil di shuttle run bersama, **GRATIS buat anggota IEEE** di shuttle TTSKY26b (2026). Tools desain gratis (Wokwi buat pemula, Verilog/HDL buat yang lebih serius), dari digital basic sampai analog/mixed-signal custom. **Catatan status 2026**: Efabless (partner fab sebelumnya) tutup operasi, tapi Tiny Tapeout pindah ke partner baru (`wafer.space`, didirikan Tim Ansell) dan tetap jalan — proyek gak mati, cuma ganti supplier. [Tiny Tapeout news](https://tinytapeout.com/news/), [status setelah Efabless tutup](https://www.eenewseurope.com/en/tiny-tapeout-sees-industrial-boost-as-it-recovers-from-efabless-closure/), [Hackster: shuttle IHP terbaru](https://www.hackster.io/news/tiny-tapeout-opens-an-ihp-shuttle-for-your-open-source-chip-designs-but-beware-the-new-terms-77e0b292cae4)

**Ini TEMUAN PALING RELEVAN buat niat user** — literal "punya chip sendiri yang difabrikasi beneran", bukan simulasi/soft-core di atas silikon orang lain. Tapi **(H) kalibrasi wajib**: chip Tiny Tapeout itu KECIL (area terbatas di shared wafer, biasanya skala ribuan-puluhan ribu gate, bukan jutaan/miliaran transistor kayak CPU komersial) — cocok buat prototipe/proof-of-concept desain (misal: satu core von Neumann minimal, atau satu blok neuron spiking kecil), **BUKAN** buat chip yang performanya kompetitif lawan CPU/GPU komersial. Timeline shuttle run juga berbulan-bulan (submit desain → tunggu batch fabrikasi → nunggu lagi buat dapet chip fisik balik), bukan instant.

**(F) Opsi 3 — Discrete logic / relay / TTL computer** (gaya proyek hobbyist Ben Eater): paling murah, paling "dari nol" secara tangan (rakit gerbang logika sendiri dari komponen diskrit), tapi performa di rentang **kHz**, jauh di bawah bahkan komputer 1980an — murni edukasi/pembuktian konsep "aku ngerti cara kerja CPU dari gerbang logika", bukan buat workload riil.

### 42b. Implikasi ke Angka §41 — Belum Berubah, Tapi Premisnya Sekarang Beda

Angka ekonomi/performa di §41b (gap 3-4 orde besaran ke superkomputer, listrik 36kW buat 2,66 PFLOP/s, dst.) **TETAP BENAR secara fisika** — fabrikasi sendiri gak mengubah hukum fisika/termodinamika. **(H) Justru kemungkinan makin berat buat opsi "bangun sendiri"**: chip yang difabrikasi individual (Tiny Tapeout node lama/kecil, bukan node fabrikasi terdepan yang dipakai Nvidia/TSMC) hampir pasti **kalah efisiensi per-FLOP** dibanding silicon komersial yang dibikin node fab termaju + tim engineer ribuan orang. Jadi value proposition-nya **BUKAN** "lebih murah/lebih kuat dari beli" — value-nya adalah **kepemilikan desain + proses belajar + "ini chip yang gue desain sendiri, bukan yang orang lain bikin"**, yang konsisten sama [[user_core_motivation_novelty]] (dorongan utama: nemuin/bikin sesuatu yang belum ada, bukan optimasi biaya).

### 42c. Pertanyaan Terbuka ke User — Belum Dijawab, Nentuin Bentuk Proyek

Tiga opsi di §42a itu proyek yang BEDA JAUH skalanya (waktu, biaya, kompleksitas desain). Belum ada keputusan user soal mana yang dimaksud "bangun von Neumann dari 0" — **jangan diasumsikan**, ini harus dijawab eksplisit sebelum riset lanjut ke arah manapun.

**Update (2026-10-05, sama hari)**: user minta langsung jalan ("mana yang terbaik dan masuk akal, lo kerjain"), rekomendasi §42c sebelumnya (FPGA soft-core dulu, baru Tiny Tapeout) diambil. Belum beli board FPGA (§39d masih berlaku) — jadi yang dikerjain sekarang itu tahap yang gak butuh hardware fisik sama sekali: desain RTL + simulasi lokal, baru disintesis ke FPGA fisik begitu board-nya ada.

### 42d. Milestone 1 — Single-Cycle RV32I Core (subset), Simulasi PASS (2026-10-05)

**(E)** Ditulis dari nol (Verilog RTL asli, bukan nyalin soft-core yang udah ada) di `fpga_von_neumann/rtl/`: `alu.v`, `regfile.v` (32×32-bit, x0 hardwired 0), `control.v`, `imem.v`, `dmem.v`, `cpu.v` (datapath single-cycle). Instruction subset v1: ADD/SUB/AND/OR/XOR/SLT (R-type), ADDI/ANDI/ORI (I-type), LW, SW, BEQ, JAL — **cukup buat program nyata sekecil apa pun, belum RV32I lengkap** (belum ada BNE/BLT/dkk, belum ada SLL/SRL/shift, belum ada subtraksi immediate terpisah — dicatat eksplisit sebagai TODO di komentar `cpu.v`, bukan disembunyikan).

**(E) Toolchain**: `iverilog 13.0` + `yosys 0.66` diinstall dari repo CachyOS (`pacman -S iverilog yosys`, 41MB download, reversible). Program tes di-encode manual pakai `tools/asm.py` (encoder RV32I bikinan sendiri dari layout bit ISA langsung — R/I/S/B/J type — bukan pakai assembler/toolchain RISC-V yang udah ada, konsisten sama scope "bangun dari nol").

**(E) Hasil simulasi** (`iverilog` + `vvp`, program 9 instruksi: addi/addi/add/sw/lw/beq/addi(skip)/addi/jal-infinite-loop):
```
x1 = 5  (expect 5)
x2 = 10 (expect 10)
x3 = 15 (expect 15)
x4 = 15 (expect 15, via lw)
x5 = 42 (expect 42, NOT 999)
PASS: single-cycle RV32I core (ADD/ADDI/LW/SW/BEQ/JAL subset) executed correctly.
```
**x5=42 (bukan 999) itu bukti BEQ beneran nge-skip instruksi berikutnya** — bukan kebetulan kebetulan ketolong bug lain (kalau branch logic-nya salah, hasilnya 999, bukan 42, jadi test ini genuinely diskriminatif).

**Kalibrasi — jangan overclaim dari hasil ini**: ini SIMULASI software (Icarus Verilog), bukan hardware fisik — belum ada FPGA yang disentuh. Instruction set-nya SANGAT minimal (9 opcode dari ~47 opcode RV32I dasar). Memori instruksi/data cuma 1KB masing-masing (256 word), dan baca memori dibuat kombinasional (bukan synchronous-read kayak block-RAM FPGA asli) — ini bakal butuh disesuaikan pas tahap sintesis FPGA nanti (block RAM vendor biasanya synchronous-read), dicatat sebagai gap yang HARUS dibereskan sebelum §39d (beli FPGA) lanjut ke sintesis, bukan sekarang.

**Next step (belum dieksekusi)**: (1) tambah BNE/BLT/BGE/shift biar instruction set lebih lengkap, (2) coba sintesis `yosys` buat liat netlist/resource usage ESTIMASI (tanpa perlu FPGA fisik — yosys bisa `synth_generic` buat sanity check sintesis murni software), (3) riset board FPGA murah yang muat buat core ini + sirkuit neuromorphic (LIF neuron) bareng di satu chip, buat keputusan beli (§39d).

## 43. Software-Optimization buat Desain "Gatekeeper" Neuromorphic — Target Terukur, Bukan "Segacor Mungkin" Tanpa Batas (2026-10-05)

User minta didalemin: optimasi desain gatekeeper (§42-diskusi, belum ditulis formal sebagai section sebelum ini) itu dari sisi SOFTWARE/riset algoritma, bukan ganti hardware, dan "dibuat segacor mungkin". **Kalibrasi wajib di depan**: "segacor mungkin" itu arahnya oke, tapi harus diterjemahin jadi angka yang bisa dikejar/dilewatin — pola §41a (tekanan buat "jenius" itu sendiri generator overclaim) tetap berlaku di sini. Jadi tiap klaim di bawah dipasangin baseline measurable.

### 43a. Baseline Terukur — Pola "Cascade Early-Exit" itu Udah Established, Substrat-Agnostic

**(F)** Keyword-spotting accelerator (cascade/early-exit): **69% input bisa exit lebih awal, reduksi power rata-rata 22%** dibanding selalu jalanin network penuh. **(F) CalexNet (2025)**: metode kalibrasi drop-in buat cascade early-exit APAPUN backbone-nya (frozen), nol perubahan arsitektur saat inference. [Low-Power Branch CNN Early Exit](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC12349502/), [CalexNet](https://arxiv.org/pdf/2509.08318)

**Implikasi**: pola "gate murah + model mahal" itu **bukan ide baru kita** — ada preseden kuantitatif (69%/22%) di domain BERBEDA (keyword spotting, bukan JEPA/SSL). **Target realistis buat gate neuromorphic kita**: minimal nyamain atau ngelewatin rasio ini (gate exit rate tinggi, power reduction signifikan), diukur di workload SSL kita sendiri — bukan angka yang diimpor mentah dari domain lain.

### 43b. Lever Software Konkret #1 — Threshold/Surrogate-Gradient Optimization SNN (2026, state-of-the-art)

**(F)** Dua paper AAAI 2026 relevan langsung: **"Optimization Method for Surrogate Function in SNN Based on Membrane Potential Distribution"** dan **"DS-ATGO: Dual-Stage Synergistic Learning via Forward Adaptive Threshold and Backward Gradient Optimization"** — dua-duanya nyerang masalah yang sama: surrogate gradient SANGAT sensitif ke skala threshold tegangan membran (V_thr), training bisa gagal/unstable kalau threshold-nya gak di-tuning bener. [Stabilizing Direct Training of SNN](https://arxiv.org/pdf/2511.08708)

**(F) Properti yang kita BUTUH buat gate** (bukan buat klasifikasi akurat): *"bounded, reset-driven dynamics... bisa ngasih onset timing yang tajam TANPA false positive berkepanjangan di antara burst"* — ini PERSIS requirement gate (jangan bangunin GPU gara-gara noise, tapi begitu ada event beneran langsung tajam responsnya). Ini bukan properti "akurasi klasifikasi" (yang udah kebukti kalah di §34/§35) — ini properti BEDA (precision/recall timing), yang gak diaudit di §34/§35 sama sekali. **Gate dan classifier itu dua masalah optimasi berbeda** — kekalahan spiking di klasifikasi (§35) TIDAK otomatis berarti kalah di gating.

### 43c. Lever Software Konkret #2 — Distilasi dari JEPA Sendiri (ini yang paling elegan & paling match sama fondasi riset kita)

**(F)** *"Exploiting neuro-inspired dynamic sparsity for energy-efficient intelligent perception"* (Nature Communications, 2025): predictive coding + attention-gating dipakai BARENG buat jaga sparsity + efisiensi energi — otak fokusin resource ke input yang "gak terduga" (**surprise**), bukan ke semua input rata. [Nature Comms](https://www.nature.com/articles/s41467-025-65387-7)

**(H, ide sintesis, BELUM ada di literatur — ini kandidat kontribusi asli)**: JEPA kita **UDAH PUNYA** sinyal "surprise" built-in — prediction error antara predictor dan target encoder (§3: definisi JEPA itu sendiri). **Jangan desain sinyal gating dari nol** — **distill prediction-error JEPA yang udah dilatih (dense, §34/§35 pemenang) ke SNN gate yang kecil**: SNN kecil dilatih buat memprediksi "kapan prediction-error dense-JEPA bakal tinggi" dari input MENTAH (raw sensor, belum diproses), tanpa harus menjalankan dense encoder yang mahal itu duluan. Kalau SNN prediksi error-nya bakal tinggi (= ada sesuatu yang "mengejutkan"/salient) → **baru** bangunin dense encoder yang mahal buat analisis penuh. Kalau SNN prediksi rendah (= business as usual, gak ada yang baru) → dense encoder TETAP TIDUR.

**Kenapa ini lebih kuat dari ide gating generik**: (1) sinyal supervisinya udah ADA (prediction-error dense-JEPA kita sendiri, nol butuh label baru/dataset baru), (2) nyambung LANGSUNG ke tujuan asli proyek §29 ("beneran bangun komputer" buat arsitektur JEPA-like), bukan tempelan terpisah, (3) §40c (noise-as-resource) jadi relevan lagi — SNN gate yang "salah" sesekali (gak fatal, cuma nge-trigger dense encoder yang gak perlu) itu toleran, beda dari training klasifikasi yang butuh presisi tinggi.

### 43d. Target Terukur buat Validasi (Belum Dieksekusi — Rencana)

Prinsip sama kayak §41e (T1-T4): tiap klaim "segacor" wajib punya satuan + baseline + kondisi gagal.

- **Metrik gate**: (a) **duty-cycle reduction** — berapa % waktu dense encoder "boleh tidur" dibanding selalu-nyala, pakai gate ini vs baseline §43a (69%/22% dari domain lain sebagai starting reference, BUKAN target yang dijamin tercapai di domain kita). (b) **Recall event penting** — berapa % event yang beneran "surprising" (prediction-error tinggi di ground truth) yang BERHASIL ditangkep gate, gak boleh ada yang keslip (miss itu lebih mahal dari false-positive buat use-case ini).
- **Baseline pembanding wajib**: dense encoder jalan terus-menerus (always-on) — itu baseline "0% hemat", dan gate acak/random (buka 50% waktu tanpa logic) — itu baseline "0% informasi", biar tau gate kita beneran belajar sesuatu bukan cuma random filtering.
- **Gagal yang diterima**: kalau SNN gate gagal distill prediction-error dense-JEPA dengan baik (recall rendah / correlation rendah ke ground-truth surprise), itu HASIL PUBLISHABLE juga (persis pola §27/§28/§35) — bukan berarti ide gating-nya mati, tapi berarti distilasi dense→spiking buat sinyal ini spesifik lebih sulit dari diharapkan, dicatat sebagai temuan.

**Status: semua di §43 ini RENCANA, nol eksekusi kode.** Prasyarat sebelum jalan (urutan dari §41f tetep berlaku): §36i (encoder tervalidasi di hxtorch) dan keputusan FPGA/Tiny Tapeout (§42c) harus clear duluan, karena gate ini butuh encoder dense-JEPA yang established buat jadi sumber supervisi prediction-error-nya.

## 44. "Sleep Consolidation" — Mekanisme yang Mungkin Nutup Keterbatasan §29e (On-Chip Learning Nol Backprop) (2026-10-05)

User nyebut "AI yang bisa terus belajar sendiri, kayak bigsleep" — dicek literatur, ternyata ada bidang riset NYATA namanya persis **"Sleep"/sleep-consolidation paradigm** buat continual learning, bukan istilah yang user karang.

### 44a. Grounding Literatur

**(F)** Paradigma "Sleep" (Google + Cornell): model terus belajar, **distill memori jangka-pendek yang rapuh jadi pengetahuan jangka-panjang yang stabil lewat REPLAY**, plus proses "Dreaming" buat recursive self-improvement. **(F) SCM (Sleep-Consolidated Memory)**: siklus sleep memicu 3 proses — **NREM consolidation (penguatan Hebbian + synaptic downscaling)**, **REM dreaming (generasi asosiasi baru)**, dan **forgetting intentional berbasis-nilai (pruning)**. **(F) SleepGate**: forgetting gate network ringan yang evict/compress cache entry basi, plus consolidation module yang gabungin entry terkait jadi representasi ringkas. [SCM paper](https://arxiv.org/html/2604.20943v1), [Sleep-like replay reduces catastrophic forgetting (Nature Comms 2022)](https://www.nature.com/articles/s41467-022-34938-7), [Language Models Need Sleep](https://arxiv.org/html/2606.03979v1)

**(F) Konsep intinya**: *"perlu ada fase OFFLINE antara ngeliat sesuatu dan berubah karenanya"* — model gak update bobot secara live terus-menerus pas lagi "jaga" (inference/respond), tapi ngumpulin pengalaman dulu, baru pas "tidur" dia proses ulang (replay) dan putusin apa yang disimpen permanen.

### 44b. KENAPA INI PENTING — Nutup Lubang §29e, Bukan Cuma Ide Tempelan

**(F) §29e (udah lama berdiri di dokumen ini)**: on-chip learning neuromorphic (Loihi 2 dkk) cuma bisa local/3-factor rule, **NOL backprop** — makanya "training JEPA on-chip" divonis **mustahil secara arsitektural**. Ini kelemahan struktural yang udah berkali-kali dicatat (§29e, §40b, §41c).

**(H, sintesis — ini yang bikin "bigsleep" bukan cuma analogi puitis)**: **NREM consolidation di paradigma Sleep itu ESENSINYA Hebbian strengthening + synaptic downscaling — DUA-DUANYA LOCAL RULE, bukan backprop.** Local rule itu PERSIS jenis learning yang neuromorphic hardware BISA jalanin secara native (§29e sendiri bilang "on-chip cuma local/3-factor rule" — tadinya ditulis sebagai KETERBATASAN, tapi local-rule itu justru SYARAT yang dipenuhi sama mekanisme consolidation gaya sleep). **Jadi "AI yang terus belajar sendiri" itu mungkin BUKAN lewat neuromorphic nyoba genjot backprop (yang emang gak bisa, §29e) — tapi lewat neuromorphic ngerjain consolidation ala-sleep (Hebbian/local), yang dari awal memang itu yang dia bisa.**

### 44c. Desain Gabungan — "Awake Gate" (§43) + "Sleep Consolidation" (§44), Siklus Day/Night

1. **Fase AWAKE** (§43, udah direncanain): gate SNN hemat energi nungguin input terus-menerus, trigger dense-JEPA cuma pas ada event "surprising" (prediction-error tinggi). Setiap kali trigger kejadian, **simpan snapshot event itu** (bukan langsung update bobot permanen di saat itu — ini bedanya sama online learning biasa).
2. **Fase SLEEP** (periodik, misal pas idle/baterai dicas/jadwal tertentu): sistem OFFLINE, replay snapshot event yang kekumpul dari fase awake, jalanin **local Hebbian update + synaptic downscaling** (NREM-style) buat konsolidasiin pola yang sering muncul jadi bobot permanen, **plus forgetting/pruning** buat event yang ternyata gak penting (value-based, kayak SCM).
3. **Kenapa dua fase ini COCOK dipisah ke substrat beda**: AWAKE butuh latensi rendah + energi rendah terus-menerus → neuromorphic (§43, sesuai job-table §40b). SLEEP butuh waktu buat "mikir ulang" tapi gak real-time → bisa VON NEUMANN (replay buffer, batch processing) ATAU neuromorphic (kalau update-nya emang local-rule-compatible) — **ini pertanyaan terbuka yang butuh riset lanjut, bukan kesimpulan final.**

### 44d. Kalibrasi — Ini HIPOTESIS SINTESIS, Belum ada yang Nerapin ke Kombinasi Kita

**(H) Belum ketemu paper yang eksplisit gabungin**: (a) SNN event-gate hemat-energi + (b) sleep-consolidation Hebbian + (c) JEPA/predictive-coding sebagai sinyal surprise, jadi SATU sistem. Masing-masing komponen (a), (b), (c) itu **(F) ada literaturnya sendiri-sendiri** (§43, §44a, §3) — tapi kombinasi tiganya **belum ketemu preseden**, mirip situasi §41d (nol preseden kombinasi penuh). Pencarian buat klaim ini **cuma 1 putaran query** (lebih sedikit dari §41d yang 12 query) — **confidence "belum ada" di sini LEBIH RENDAH** dari §41d, jangan disamain levelnya.

**Ini BUKAN klaim "kita nemuin AI yang bisa belajar sendiri"** — ini kerangka hipotesis yang: (1) konsisten sama kendala arsitektural yang udah ketemu (§29e), (2) punya precedent kuat per-komponen, (3) BELUM divalidasi sebagai gabungan. Status sama kayak §43: **rencana, nol eksekusi**, dan prasyaratnya lebih panjang dari §43 (butuh §43 jalan dulu buat punya snapshot event yang bisa di-replay).

### 44e. Next Step (Belum Dieksekusi)

Setelah §43 (gate) jalan dan ngasilin snapshot event: (1) riset lebih dalam apakah Hebbian/STDP consolidation bisa diimplementasi di hxtorch/BrainScaleS-2 secara konkret (cek API, bukan asumsi — pola yang sama kayak §36), (2) desain eksperimen kecil: snapshot event dari gate §43, replay offline, ukur apakah consolidation ala-SCM beneran ningkatin retensi tanpa catastrophic forgetting dibanding baseline "update langsung tiap event" (online naive) vs "nol update" (static). Baseline ketiga ini WAJIB biar tau consolidation-nya beneran nambah nilai, bukan cuma kedengeran keren.

## 45. Blocker Nyata — EBRAINS "Access Denied" Pas Coba Login (2026-10-05)

User coba login ke EBRAINS (`iam.ebrains.eu`, client `ebrains-web`) buat eksekusi §41f step 0 (cek kuota), hasilnya **"Access Denied — You do not currently have access to this application."** (screenshot, tab "Sign in to EBRAINS").

**(E) Diagnosis dari pesan errornya**: bunyi errornya "akses ke APLIKASI ini", bukan "kredensial salah"/"akun gak ketemu" — beda arti. Konsisten sama proses yang dari awal udah dicatat di §29c/§29g: **bikin akun EBRAINS itu BUKAN otomatis dapet akses ke Collab/service neuromorphic computing** — perlu langkah tambahan: (a) halaman resmi [Getting access](https://wiki.ebrains.eu/bin/view/Collabs/neuromorphic/Getting%20access/), dan/atau (b) email ke `neuromorphic@humanbrainproject.eu` (rencana dari §29g, **belum pernah dikonfirmasi udah dikirim/dibalas**).

**Status: BLOCKER TERBUKA.** Belum jelas apakah (1) akun EBRAINS user belum selesai dibikin, (2) akun udah ada tapi belum request akses ke Collab neuromorphic, atau (3) proses §29g (email) emang belum pernah dieksekusi. **Jangan diasumsikan** — next action: user coba halaman "Getting access" + kirim email kalau belum, laporan hasilnya ke sini sebelum lanjut §41f step 0.

**Update sama hari — dua teori ditolak, eskalasi ke email**: User klarifikasi akun guest EBRAINS ini **BERHASIL dipake 2026-10-03** (2 hari sebelumnya, §36-38, bukan cuma dicoba — beneran eksekusi smoke test di Collab). Jadi **dua teori awal salah**: (1) bukan soal "belum pernah request akses" (udah pernah jalan), (2) dicoba masuk lewat link Collab yang sama yang berhasil sebelumnya — **masih Access Denied**, jadi juga bukan soal "salah app/tab". Guest access katanya 1 bulan (§36) tapi baru 2 hari udah ketolak — entah itu emang expired lebih cepat dari diinfokan, atau sebab lain yang belum diketahui.

**Keputusan: berhenti nebak, eskalasi langsung ke `neuromorphic@humanbrainproject.eu`** (rencana §29g yang belum pernah dieksekusi beneran) — tanyain status akun secara langsung, bukan nambah teori ketiga tanpa bukti. **Belum dikirim per penulisan ini** — nunggu konfirmasi user udah kirim + jawaban mereka.

**Keputusan arsitektur**: EBRAINS cuma nge-block pilar NEUROMORPHIC (§36i, §43, §44) — von Neumann (§42) dan quantum (IBM Quantum, belum didaftar) independen total, gak butuh EBRAINS. Jadi blocker ini **bisa ditinggal paralel**, lanjut kerjain yang gak keblokir sambil nunggu balesan email.

## 46. Core von Neumann v2 — Instruction Set Diperluas (BNE/BLT/BGE/SLL/SRL), Masih PASS (2026-10-05)

**(E)** Lanjutin §42d sambil nunggu EBRAINS. Perubahan RTL: `alu.v` nambah `ALU_SLL`/`ALU_SRL`; `control.v` nambah decode SLL/SRL (R-type funct3 001/101) dan branch control disederhanain (cuma `br=1`, kondisi spesifik dipindah ke komparator baru); `cpu.v` dapet comparator khusus branch (`branch_eq`, `branch_lt` via `$signed`) yang baca `funct3` langsung buat milih BEQ/BNE/BLT/BGE — **ganti total dari trik lama** (alu_op=SUB, cek result==0, yang cuma bisa BEQ, §42d).

**(E) Assembler juga di-upgrade**: `tools/asm.py` sekarang two-pass dengan label (`p.label(...)`, offset branch/jump dihitung otomatis dari label, bukan hitung manual byte-offset) — biar program makin panjang gak makin gampang salah hitung.

**(E) Hasil simulasi, program 23 instruksi** (nambahin: SLL, SRL, BNE-not-taken, BLT-taken, BGE-taken, masing-masing dengan "trap value" kalau logic-nya salah — 666/222/444 gak boleh pernah muncul):
```
x1=5 x2=10 x3=15 x4=15 x5=42(bukan 999) x6=3 x7=2 x9=12(SLL) x10=3(SRL)
x11=111(bukan 666, BNE) x12=333(bukan 222, BLT) x13=555(bukan 444, BGE)
PASS: single-cycle RV32I core (ADD/ADDI/LW/SW/SLL/SRL/BEQ/BNE/BLT/BGE/JAL) executed correctly.
```
Semua 4 jenis branch (BEQ/BNE/BLT/BGE) sekarang punya test yang diskriminatif (bukan cuma 1 arah) — kalau comparator-nya salah sedikit aja, trap value bakal muncul dan test FAIL.

**(E) Coba sintesis `yosys` buat estimasi resource** — temuan tak terduga: `synth -top cpu` ngilangin HAMPIR SEMUA logic (stat akhir cuma nunjukin 9 wire, 0 cell). Diselidiki: ternyata `cpu.v` cuma punya port `clk`/`rst`, **nol output eksternal** — yosys optimizer (correctly) nganggep semua logic internal "gak diobservasi dari luar" dan nyapu bersih pas full synth. Dikonfirmasi dengan jalanin cuma `proc` (tanpa full `synth`/`opt`) — di situ kelihatan jelas 5 submodule beneran keinstantiate dengan cell asli (`$mux`, `$memrd`, `$memwr_v2`, dll). **Bukan bug, bukan salah desain** — ini konsekuensi logis dari cpu.v yang memang didesain cuma buat disimulasi (testbench baca internal lewat hierarchical reference `dut.rf.regs[...]`, bukan lewat port).

**Next step (belum dieksekusi)**: tambah port debug/memory-mapped I/O ke `cpu.v` (misal expose nilai register tertentu atau bus keluar) SEBELUM nyoba resource-count yang valid — tanpa itu, angka sintesis apa pun dari core ini gak ada artinya. Ini juga relevan buat FPGA fisik nanti: real board butuh cara buat lihat hasil (LED, UART, dll), bukan cuma `$display` testbench.

## 47. §45 TERJAWAB — Akar Masalah "Access Denied" Ketemu dari Email Asli EBRAINS (2026-10-05)

User share isi email balasan asli dari **Dr. Björn Kindler (Neuromorphic Computing, Heidelberg)**, dibalas 2026-10-02 ke email institusi user (`@mhs.unipma.ac.id`, Universitas PGRI Madiun/UNIPMA). **Fakta krusial yang sebelumnya gak kecatet di mana pun**:

- **Username akun guest**: `guestmarhabynovlafifalutv`, batas waktu 1 bulan dari 2026-10-02.
- **Collab test NMC spesifik user**: `https://wiki.ebrains.eu/bin/view/Collabs/nmc-test-guestmarhabynovlafifalutv` — **ini halaman yang punya kuota test**, target asli §41f step 0 yang selama ini belum ketemu.
- **Prosedur login resmi dari EBRAINS sendiri** (bukan dari web browsing gua): (1) buka `https://wiki.ebrains.eu` (BUKAN `iam.ebrains.eu` langsung, BUKAN portal `ebrains-web` generic), (2) klik "Masuk" pojok kanan atas, (3) **pakai link "lupa kata sandi"** buat SET password pertama kali (akun guest dibikinin Björn, nol password awal yang user tau) — ini kemungkinan BESAR akar masalah §45/§46: kalau password belum pernah di-set lewat "forgot password", login manapun bakal gagal/aneh.
- **Urutan wajib SEBELUM pakai "Lab"**: akses "Drive" (menu kiri Collab) MINIMAL SEKALI dulu, baru klik "Lab". Kalau urutannya kebalik, kemungkinan bisa nyebabin masalah akses juga (gak eksplisit dibilang nyebabin Access Denied, tapi dicatat sebagai prasyarat resmi).
- **Server pas pertama buka Lab**: pilih "Citra Docker resmi EBRAINS xx.xx untuk Collaboratory.Lab (direkomendasikan)", kernel harus diganti ke "EBRAINS-...." (bukan default Python 3).
- **Kontak troubleshooting EBRAINS**: `support@ebrains.eu` (beda dari `neuromorphic@humanbrainproject.eu` yang buat request akses awal).

**Diagnosis final §45/§46**: "Access Denied" di `iam.ebrains.eu` client `ebrains-web` itu **BUKAN berarti akun gak valid/expired** — itu karena user coba masuk lewat **pintu yang salah** (portal generic `ebrains-web`) alih-alih jalur resmi yang dikasih Björn (`wiki.ebrains.eu` → Masuk → lupa password → Collab spesifik). Akun guest-nya kemungkinan besar **masih valid** (baru 3 hari dari batas 1 bulan).

**Next step buat user (bukan Claude, ini akun pribadi)**: ikutin PERSIS urutan dari email — `wiki.ebrains.eu` → Masuk → "lupa kata sandi" pakai username `guestmarhabynovlafifalutv` → set password baru → login → buka langsung `https://wiki.ebrains.eu/bin/view/Collabs/nmc-test-guestmarhabynovlafifalutv` → akses Drive dulu → baru Lab. **Kalau ini kelar, §41f step 0 (cek kuota) otomatis kejawab** karena halaman Collab itu literally yang dicari.

**RESOLVED (2026-10-05)**: user konfirmasi login berhasil ikutin prosedur di atas. **Blocker §45 ditutup — login/akses portal beres.**

## 48. Pengejaran "Angka Kuota" DIHENTIKAN — Salah Nyambungin Dua Sumber (2026-10-05)

Setelah login beres (§47), dicoba cari angka kuota via `JobManager` di Collab `nmc-test-guestmarhabynovlafifalutv` — halaman kosong (nol job history, nol indikator "Quota" yang keliatan), dicek 2x (user screenshot + Claude navigasi sendiri ke URL yang sama, hasil konsisten kosong).

**Root cause kenapa pengejaran ini salah arah**: email asli Björn Kindler (§47) cuma bilang *"temukan Collaboratory uji NMC Anda dengan beberapa contoh... serta kuota uji"* — dia bilang kuota ADA di Collab itu, **TAPI GAK PERNAH bilang itu ditampilin sebagai angka di JobManager pojok kanan atas**. Detail UI itu **diambil dari halaman dokumentasi umum EBRAINS** (`Getting access`) yang digabung sama email Björn **tanpa dasar kuat** — dua sumber beda yang gak otomatis nyambung di kasus guest-account spesifik ini. **Pola kesalahan yang sama kayak §45/§46**: gabung-gabungin sumber informasi secara lompat tanpa verifikasi, bukan kali ini doang.

**Keputusan: berhenti nyari angka kuota.** Kemungkinan besar itu alokasi internal (dikelola backend/quiggeldy, §36b) yang emang gak diekspos ke user sebagai angka yang bisa di-browse — bukan sesuatu yang "ketemu kalau nyari lebih keras". **Status §41f step 0 DITUTUP sebagai "tidak applicable/tidak dapat diverifikasi lewat UI"**, bukan "ditemukan nol". Kalau limit beneran kena pas run eksperimen, itu bakal kelihatan dari error-nya sendiri saat itu — itu jadi sinyal yang lebih reliable daripada nebak dari UI yang gak jelas.

**Next step**: lanjut ke `Lab` (bukan JobManager) — ikutin instruksi asli Björn: buka tutorial notebook `index` di folder `BrainScaleS > brainscales2-demos`. Ini juga prasyarat buat §36i.

## 49. Masuk ke Lab Collab Beneran, Baca Tutorial `ts_12-hxtorch_snn_intro` — Konfirmasi Rencana §37 Valid (2026-10-05)

**(E)** Login beres (§47-48), masuk ke JupyterLab (`lab.jsc.ebrains.eu/user/guestmarhabynovlafifalutv/...`), kernel otomatis ke-set `EBRAINS-25.10` (bukan default Python 3, sesuai anjuran). Dibaca `index.ipynb` (overview), lalu `ts_00-single_neuron.ipynb` (PyNN-level API, bukan yang kita butuhin langsung), lalu ketemu dan dibaca **`ts_12-hxtorch_snn_intro.ipynb`** — ini tutorial resmi buat API `hxtorch.snn` yang PERSIS dipake di `spiking_jepa_v2.py` kita.

**(F) Temuan kunci, konfirmasi langsung dari tutorial resmi (bukan dari ingatan/asumsi)**:
1. **Mode `mock=True` TIDAK butuh kalibrasi atau `init_hardware()`** — contoh kode tutorial: `exp = hxsnn.Experiment(dt=1e-6, mock=True)` langsung dipakai, nol `load_calib()`, nol `hxtorch.init_hardware()` dipanggil sebelumnya. **Ini mengkonfirmasi rencana §37 valid** ("mock=True dan TANPA pernah manggil hxtorch.init_hardware()" — sebelumnya diturunkan dari analisis kode, sekarang dikonfirmasi dari tutorial resmi EBRAINS sendiri).
2. **Mode hardware ASLI (mock=False) butuh kalibrasi** — pola: `save_nightly_calibration('spiking_calix-native.pkl')` lalu `exp.default_execution_instance.load_calib("spiking_calix-native.pkl")`. **Detail ini belum pernah kecatet di §36-38** — relevan kalau nanti lanjut dari mock ke hardware asli, BELUM relevan buat tahap mock-only yang lagi direncanain.
3. API pattern persis yang dipake di kode kita: `hxsnn.Synapse(in_features, out_features, experiment=exp)` → `hxsnn.LIF(size, experiment=exp, enable_cadc_recording=True, enable_madc_recording=True, enable_spike_recording=True, record_neuron_id=0, ...)`.
4. **CADC vs MADC**: CADC (Columnar ADC) sample SEMUA neuron paralel, resolusi temporal rendah; MADC (Membrane ADC) cuma sample 2 neuron, tapi resolusi temporal tinggi. Trade-off eksplisit, bukan "CADC selalu dipakai".
5. Utility `linear_saturating` (`from hxtorch.snn.transforms.weight_transforms import linear_saturating`) — buat scaling bobot software↔hardware, relevan ke isu kuantisasi 6-bit (§38).
6. **Bobot hardware range -63 sampai 63** (6-bit signed), dikonfirmasi eksplisit di komentar kode tutorial — konsisten sama temuan kuantisasi 6-bit di §38, bukan kontradiksi.

**Status**: belum eksekusi kode apa pun di Collab (baru baca tutorial, nol run cell). Next: lanjut baca sisa `ts_12` (bagian training loop LI neuron belum kebaca full), atau langsung mulai kerjain §36i (rekonstruksi `SpikingEncoderV2` mock-only) pakai API pattern yang udah dikonfirmasi ini.

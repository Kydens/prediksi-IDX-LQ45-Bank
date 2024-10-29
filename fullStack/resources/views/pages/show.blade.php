@extends('layouts.AppLayouts')

@section('content')
    <div class="wrapper d-flex flex-column gap-3">
        <div class="card">
            <div class="card-body overflow-hidden">
                <h5 class="card-title text mx-0 mt-0 mb-2">Prediksi</h5>
                <p class="card-text"><strong>Prediksi pergerakan sinyal</strong> ini dilakukan dengan menggunakan algoritma
                    <strong>Random Forest Regression</strong> dan <strong>XGBoost Regression</strong> yang merupakan metode
                    Machine Learning (Bagian dari Artificial Intelligence). Hasil dari prediksi akan berupa <strong>status
                        pergerakan sinyal saham</strong> di masa mendatang untuk 1 pekan, 1 bulan, 3 bulan, 6 bulan, atau 1
                    tahun.
                </p>
                <p class="card-text mb-4">
                    Penentuan status pergerakan akan menggunakan bantuan indikator teknikal <strong>Bollinger Bands</strong>
                    dengan <strong>Simple Moving Average</strong> periode <strong>window 20 hari</strong> dan <strong>50
                        hari</strong>. Bollinger Bands terdiri dari 3 pita yaitu <strong class="text-danger">pita atas
                        (Upperbands)</strong> dan
                    <strong class="text-success">pita bawah
                        (Lowerbands)</strong> akan menunjukkan untuk menentukan <strong class="text-danger">titik tertinggi
                        (Oversold)</strong> dan
                    <strong class="text-success">titik terendah (Overbought)</strong>, serta pita tengah atau disebut dengan
                    <strong>pita SMA (Simple Moving Average)</strong>.
                </p>
                <p class="card-text fw-bold mb-1">Saham-saham IDX LQ45 perbankan berikut yang dapat anda prediksi : </p>
                <div class="row mb-3">
                    <div class="col-8 d-flex flex-wrap">
                        <a href="{{ route('dashboard.viewTicker', ['ticker' => 'ARTO.JK']) }}"
                            class="btn btn-primary m-1">Bank
                            Jago</a>
                        <a href="{{ route('dashboard.viewTicker', ['ticker' => 'BBCA.JK']) }}"
                            class="btn btn-primary m-1">Bank
                            Central Asia</a>
                        <a href="{{ route('dashboard.viewTicker', ['ticker' => 'BBTN.JK']) }}"
                            class="btn btn-primary m-1">Bank
                            Tabungan Negara</a>
                        <a href="{{ route('dashboard.viewTicker', ['ticker' => 'BBNI.JK']) }}"
                            class="btn btn-primary m-1">Bank
                            Negara Indonesia</a>
                        <a href="{{ route('dashboard.viewTicker', ['ticker' => 'BBRI.JK']) }}"
                            class="btn btn-primary m-1">Bank
                            Republik Indonesia</a>
                        <a href="{{ route('dashboard.viewTicker', ['ticker' => 'BRIS.JK']) }}"
                            class="btn btn-primary m-1">Bank
                            Syariah Indonesia</a>
                        <a href="{{ route('dashboard.viewTicker', ['ticker' => 'BMRI.JK']) }}"
                            class="btn btn-primary m-1">Bank
                            Mandiri</a>
                    </div>
                </div>
                <small class="card-text">
                    Jika anda tertarik, anda dapat melihat urutan saham perusahaan-perusahaan berindeks LQ45 yang telah
                    diperbahui secara berkala melalui "<a href="https://www.idx.co.id/id/berita/pengumuman"
                        class="text-decoration-none text-primary" target="_blank">Pengumuman Bursa Efek Indonesia</a>".
                </small>
            </div>
        </div>
    </div>
@endsection

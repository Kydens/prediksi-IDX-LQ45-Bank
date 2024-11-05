@extends('layouts.AppLayouts')

@section('content')
    <div class="wrapper d-flex flex-column gap-3">
        <div class="card">
            <div class='card-body'>
                <h3 class="text-center pb-2 text mx-0 fw-bold w-100">Bantuan</h3>
                <p class="text-center mb-5">
                    Temukan jawaban dari pertanyaan-pertanyaan anda di bawah ini.
                </p>
                <div class="accordion shadow rounded" id="accordionExample">
                    <div class="accordion-item">
                        <h2 class="accordion-header" id="headingOne">
                            <button class="accordion-button" type="button" data-bs-toggle="collapse"
                                data-bs-target="#collapseOne" aria-expanded="true" aria-controls="collapseOne">
                                <strong>Apa itu Saham Indeks LQ45?</strong>
                            </button>
                        </h2>
                        <div id="collapseOne" class="accordion-collapse collapse show" aria-labelledby="headingOne"
                            data-bs-parent="#accordionExample">
                            <div class="accordion-body">
                                Saham indeks LQ45 merupakan indeks saham unggulan di Bursa Efek
                                Indonesia (BEI). Saham indeks LQ45 terdiri dari empat puluh lima saham paling
                                aktif yang memiliki likuiditas yang tinggi dalam Bursa Efek Indonesia (BEI). Salah satu
                                faktor penentu sebuah saham perusahaan dapat masuk ke dalam
                                indeks LQ45 adalah dengan kondisi keuangan dan prospek pertumbuhan
                                perusahaan di masa yang akan datang. Saham indeks LQ45 terjadi perubahan
                                setiap tiga bulan, yaitu setiap bulan Januari, April, Juli, dan Oktober berdasarkan
                                pengumuman evaluasi Bursa Efek Indonesia
                            </div>
                        </div>
                    </div>
                    <div class="accordion-item">
                        <h2 class="accordion-header" id="headingTwo">
                            <button class="accordion-button collapsed" type="button" data-bs-toggle="collapse"
                                data-bs-target="#collapseTwo" aria-expanded="false" aria-controls="collapseTwo">
                                <strong>Apa itu Bollinger Bands dan Simple Moving Average (SMA)?</strong>
                            </button>
                        </h2>
                        <div id="collapseTwo" class="accordion-collapse collapse" aria-labelledby="headingTwo"
                            data-bs-parent="#accordionExample">
                            <div class="accordion-body">
                                <p>Bollinger Bands dikembangkan oleh John Bollinger pada awal tahun 1980,
                                    metode ini umum digunakan sebagai indikator teknikal untuk menganalisis market
                                    keuangan dan mendeteksi sinyal jual atau beli. Bollinger Bands terdiri dari dua buah
                                    pita,
                                    yaitu pita atas (Upper Bands) dan
                                    pita bawah (Lower Bands). Kedua pita tersebut dijadikan sebagai batas atas dan
                                    bawah dengan dihitung pada jarak dua standar deviasi dari Simple Moving
                                    Average (SMA) (Di dalam pasar keuangan, Bollinger Bands menggunakan period
                                    pergerakan rata-rata waktu atau window dengan 20 hari).</p>
                                <p>Indikator Bollinger Bands dapat menunjukkan
                                    kondisi pasar
                                    saham sedang dalam Overbought (kondisi dimana harga pasar saham sedang berada pada titik
                                    tertinggi) atau Oversold (kondisi dimana harga pasar saham sedang berada pada titik
                                    tertinggi)</p>
                                <p>Simple Moving Average (SMA) adalah salah satu indikator teknikal sederhana
                                    untuk menganalisis teknikal dengan mengambil harga saham yang dirata-ratakan
                                    selama periode tertentu. Simple Moving Average digunakan untuk membatasi
                                    fluktuasi harga saham sehingga dapat menentukan sinyal beli atau jual.</p>
                            </div>
                        </div>
                    </div>
                    <div class="accordion-item">
                        <h2 class="accordion-header" id="headingThree">
                            <button class="accordion-button collapsed" type="button" data-bs-toggle="collapse"
                                data-bs-target="#collapseThree" aria-expanded="false" aria-controls="collapseThree">
                                <strong>Bagaimana cara menggunakan prediksi pergerakan sinyal?</strong>
                            </button>
                        </h2>
                        <div id="collapseThree" class="accordion-collapse collapse" aria-labelledby="headingThree"
                            data-bs-parent="#accordionExample">
                            <div class="accordion-body">
                                <p>Pengguna memilih saham IDX LQ45 Perbankan yang ingin diprediksi melalui halaman <a
                                        href="{{ route('dashboard.show') }}">"Prediksi"</a>. Setelah pengguna memilih saham
                                    yang dituju, aplikasi prediksi pergerakan sinyal ini
                                    perlu ditentukan jumlah periode
                                    pergerakan rata-rata waktu atau window, yaitu 20 hari atau 50 hari.</p>
                                <p>Aplikasi akan menampilkan tingkat error dan keakuratan prediksi, serta status dari hasil
                                    prediksi
                                    pergerakan sinyal Naik Signifikan,
                                    Turun Signifikan, Naik, Turun, dan Stabil.</p>
                            </div>
                        </div>
                    </div>
                    <div class="accordion-item">
                        <h2 class="accordion-header" id="headingFour">
                            <button class="accordion-button collapsed" type="button" data-bs-toggle="collapse"
                                data-bs-target="#collapseFour" aria-expanded="false" aria-controls="collapseFour">
                                <strong>Seberapa akurat prediksi pergerakan sinyal ini?</strong>
                            </button>
                        </h2>
                        <div id="collapseFour" class="accordion-collapse collapse" aria-labelledby="headingFour"
                            data-bs-parent="#accordionExample">
                            <div class="accordion-body">
                                <p>Prediksi pergerakan sinyal saham IDX LQ45 Perbankan menggunakan nilai ukur RMSE, MAE, dan
                                    R2 untuk mencari tingkat error dan tingkat keakuratan prediksi.</p>
                                <p><strong>RMSE</strong> atau Root Mean Squared Error adalah nilai ukur hasil dari
                                    pengakaran jumlah kuadrat
                                    dari kesalahan antara nilai aktual dengan nilai prediksi, yang dibagi dengan jumlah
                                    periode prediksi. Semakin kecil
                                    nilai error, maka semakin bagus model prediksinya</p>
                                <p><strong>MAE</strong> atau Mean Absolute Error adalah nilai ukur hasil dari
                                    rata-rata diferensiasi absolute antara nilai prediksi dengan nilai aktual. Semakin kecil
                                    nilai error, maka semakin bagus model prediksinya.</p>
                                <p><strong>R2</strong> atau R-Squared adalah nilai ukur hasil dari
                                    pengurangan 1 dengan jumlah kuadrat residual yang dibagi dengan jumlah kuadrat total.
                                    Nilai R2 berada pada 0 hingga 1. Semakin dekat dengan nilai 1, maka semakin akurat hasil
                                    prediksinya.</p>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>

        <div class="text-center">
            <small>Pertanyaan anda tidak terjawab? Hubungi <a href="mailto:nathsyahed@gmail.com">disini</a>.</small>
        </div>
    </div>
@endsection

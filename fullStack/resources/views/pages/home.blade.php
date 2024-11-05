@extends('layouts.AppLayouts')

@section('content')
    <div class="wrapper d-flex flex-column gap-3">
        <div class="card">
            <div class="card-body overflow-hidden">
                <h5 class="card-title text mx-0 mt-0 mb-2">IDX LQ45 Perbankan</h5>
                <p class="card-text">Saham IDX LQ45 merupakan indeks saham untuk perusahaan-perusahan indonesia
                    yang memiliki
                    likuiditas
                    tertinggi, kapitalisasi pasar yang besar, frekuensi perdagangan tertinggi, dan kondisi keuangan yang
                    stabil.</p>
                <p class="card-text mb-5">Dengan melihat rasio dan penerapan GCG (Good
                    Corporate Governance) atau Tata Kelola Perusahaan yang baik, saham-saham
                    sektor perbankan cenderung
                    stabil dan tergolong sehat. Dan apabila jika saham-saham sektor perbankan tersebut termasuk dalam saham
                    dengan indeks LQ45, akan membuatnya lebih terjamin.</p>
                <h5 class="fw-bold">Tertarik Untuk Prediksi Saham LQ45 Perbankan?</h5>
                <p class="card-text">
                    <a href="{{ route('dashboard.show') }}" class="btn btn-dark">Ayo Prediksi Sahamnya!</a>
                </p>
            </div>
        </div>

        @if (count($articlesTopStories) > 0)
            <div class="card">
                <div class="card-body">
                    <div class="d-flex justify-content-between">
                        <h5 class="text mx-0 mt-0 mb-0 fw-bold">Berita Teratas</h5>
                        <a href="{{ route('dashboard.news') }}" class="btn btn-dark">Berita Lainnya</a>
                    </div>
                    <hr class="mb-4">
                    <div class="row px-3 d-flex gap-3">
                        @foreach ($articlesTopStories as $article)
                            <div class="card p-3 shadow-sm" style="width: 24%; border-radius: 10px; overflow: hidden;">
                                <a href="{{ $article['url'] }}" target="_blank">
                                    <img srcset="{{ $article['urlToImage'] }}" class="card-img-top mb-3 rounded"
                                        alt="{{ $article['title'] }}"
                                        style="width: 100%; max-height: 200px; object-fit: cover; object-position: center; transition: transform 0.3s">
                                </a>
                                <div class="card-body d-flex flex-column justify-content-between p-0">
                                    <div class="wrapper-card-body">
                                        <a href="{{ $article['url'] }}" class="text-decoration-none text-dark"
                                            target="_blank">
                                            <h5 class="card-title fw-bold mb-2"
                                                style="font-size: 18px; display: -webkit-box; -webkit-line-clamp: 2; -webkit-box-orient: vertical; overflow: hidden;">
                                                {{ $article['title'] }}
                                            </h5>
                                        </a>
                                        <p class="card-text text-muted small mb-3">
                                            {{ $article['publishedAt'] }}
                                        </p>
                                    </div>
                                    <a href="{{ $article['url'] }}" class="btn w-100 text-white"
                                        style="font-size: 14px; background-color:#0067ac;" target="_blank">
                                        Baca Selengkapnya
                                    </a>
                                </div>
                            </div>
                        @endforeach
                    </div>
                </div>
            </div>
        @endif
    </div>
@endsection

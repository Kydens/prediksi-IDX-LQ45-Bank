@extends('layouts.AppLayouts')

@section('content')
    <div class="wrapper d-flex flex-column gap-3">
        <div class="card">
            <div class="card-body overflow-hidden">
                <h5 class="card-title text mx-0 mt-0 mb-2">IDX LQ45 Perbankan</h5>
                <p class="card-text">Saham IDX LQ45 merupakan saham perusahaan indonesia yang memiliki likuiditas tertinggi,
                    kapitalisasi pasar yang besar, frekuensi perdagangan tertinggi, dan kondisi keuangan yang stabil.
                    Saham-saham perbankan cenderung stabil dan tergolong sehat.</p>
                <p class="card-text fw-bold mb-1">Saham-saham IDX LQ45 perbankan berikut : </p>
                <div class="row">
                    <div class="col-8 d-flex flex-wrap">
                        <a href="{{ route('dashboard.ticker', ['ticker' => 'ARTO.JK']) }}" class="btn btn-primary m-1">Bank
                            Jago</a>
                        <a href="{{ route('dashboard.ticker', ['ticker' => 'BBCA.JK']) }}" class="btn btn-primary m-1">Bank
                            Central Asia</a>
                        <a href="{{ route('dashboard.ticker', ['ticker' => 'BBTN.JK']) }}" class="btn btn-primary m-1">Bank
                            Tabungan Negara</a>
                        <a href="{{ route('dashboard.ticker', ['ticker' => 'BBNI.JK']) }}" class="btn btn-primary m-1">Bank
                            Negara Indonesia</a>
                        <a href="{{ route('dashboard.ticker', ['ticker' => 'BBRI.JK']) }}" class="btn btn-primary m-1">Bank
                            Republik Indonesia</a>
                        <a href="{{ route('dashboard.ticker', ['ticker' => 'BRIS.JK']) }}" class="btn btn-primary m-1">Bank
                            Syariah Indonesia</a>
                        <a href="{{ route('dashboard.ticker', ['ticker' => 'BMRI.JK']) }}" class="btn btn-primary m-1">Bank
                            Mandiri</a>
                    </div>
                </div>
            </div>
        </div>

        @if (count($articles) > 0)
            <div class="card">
                <div class="card-body">
                    <h5 class="text mx-0 mt-0 mb-4 fw-bold">Berita Terkini</h5>
                    <div class="row d-flex flex-wrap justify-content-around gap-3">
                        @foreach ($articles as $article)
                            <div class="card p-3 shadow-sm" style="width: 23%; border-radius: 10px; overflow: hidden;">
                                <img src="{{ $article['urlToImage'] }}" class="card-img-top mb-3"
                                    alt="{{ $article['title'] }}" style="height: 200px; object-fit: fit;">
                                <div class="card-body p-0">
                                    <h5 class="card-title fw-bold mb-2" style="font-size: 18px;">{{ $article['title'] }}
                                    </h5>
                                    <p class="card-text text-muted small mb-3">
                                        {{ $article['publishedAt'] }}</p>
                                    <a href="{{ $article['url'] }}" class="btn btn-primary w-100" style="font-size: 14px;">
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

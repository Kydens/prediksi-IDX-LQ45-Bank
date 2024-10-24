@extends('layouts.AppLayouts')

@section('content')
    <div class="wrapper">
        <div class="card">
            {{-- <h5 class="card-header">Featured</h5> --}}
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
    </div>
@endsection

@extends('layouts.AppLayouts')

@section('content')
    <div class="wrapper">
        <div class="card">
            {{-- <h5 class="card-header">Featured</h5> --}}
            <div class="card-body overflow-hidden">
                <h5 class="card-title text mx-0 mt-0 mb-2">IDX LQ45 Perbankan</h5>
                <p class="card-text">With supporting text below as a natural lead-in to additional content.</p>
                <div class="row">
                    <div class="col-6">
                        <a href="{{ url('/ARTO.JK') }}" class="btn btn-primary m-1">Bank Jago</a>
                        <a href="{{ url('/BBCA.JK') }}" class="btn btn-primary m-1">Bank Central Asia</a>
                        <a href="{{ url('/BBTN.JK') }}" class="btn btn-primary m-1">Bank Tabungan Negara</a>
                        <a href="{{ url('/BBNI.JK') }}" class="btn btn-primary m-1">Bank Negara Indonesia</a>
                        <a href="{{ url('/BBRI.JK') }}" class="btn btn-primary m-1">Bank Republik Indonesia</a>
                        <a href="{{ url('/BRIS.JK') }}" class="btn btn-primary m-1">Bank Syariah Indonesia</a>
                        <a href="{{ url('/BMRI.JK') }}" class="btn btn-primary m-1">Bank Mandiri</a>
                    </div>
                </div>
            </div>
        </div>
    </div>
@endsection

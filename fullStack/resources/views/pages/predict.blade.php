@extends('layouts.AppLayouts')

@section('content')
    <div>
        <div class='mb-2 d-flex align-items-center gap-2'>
            <h3>Prediksi Data</h3>
        </div>
        <a class='btn btn-dark mb-3' href='{{ route('dashboard.ticker', $ticker) }}'>Kembali</a>
        <div class='col card p-4 d-flex flex-column gap-3'>
            <div class='col'>
                <canvas id='chrtTicker' class='pb-4' style='max-height: 500px; max-width: 100%;'></canvas>
                <button id='resetZoom' class='btn btn-primary'>Reset Zoom</button>
            </div>
            <div class="col">
                <hr>
            </div>
            <div class="col">
                <div class="py-4">
                    <div class="mb-4">
                        <h5 class="text-center fw-bold">Detail Prediksi Data Untuk Hari ke-{{ $days }} dengan Window
                            {{ $window }}</h5>
                    </div>
                    <div class='d-flex justify-content-center'>
                        <div class='col-md-10'>
                            <table id='tablePred' class='display cell-border' width='100%'></table>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    </div>
@endsection

@section('js')
    <script>
        $(document).ready(function() {
            function tablePredict(apiData, table) {
                var combinedData = [];

                for (var i = 0; i < apiData.data.index_pred.length; i++) {
                    combinedData.push([
                        apiData.data.index_pred[i],
                        apiData.data.status[i]
                    ]);
                }

                var table = new DataTable(table, {
                    'dom': 'rftip',
                    pageLength: 10,
                    columns: [{
                            title: 'Tanggal'
                        },
                        {
                            title: 'Prediksi'
                        },
                    ],
                    data: combinedData,
                })

                return table;
            }

            function chartPredict(apiData, canvas) {
                var close_actual = apiData.data.df_combined.slice(0, apiData.data.df_combined.length - apiData.data
                    .close_pred
                    .length);
                var close_pred = Array(apiData.data.df_combined.length - apiData.data.close_pred.length).fill(null)
                    .concat(
                        apiData.data.close_pred);
                var upper_band = Array(apiData.data.df_combined.length - apiData.data.upper_band.length).fill(null)
                    .concat(
                        apiData.data.upper_band);
                var middle_band = Array(apiData.data.df_combined.length - apiData.data.sma_band.length).fill(null)
                    .concat(
                        apiData.data.sma_band);
                var lower_band = Array(apiData.data.df_combined.length - apiData.data.lower_band.length).fill(null)
                    .concat(
                        apiData.data.lower_band);

                var chartLine = new Chart(canvas, {
                    type: 'line',
                    data: {
                        labels: apiData.data.index_combined,
                        datasets: [{
                                label: 'Harga Penutupan Aktual (Close)',
                                data: close_actual,
                                fill: false,
                                borderColor: 'blue',
                                backgroundColor: '#ffffff',
                                pointRadius: 0,
                                borderWidth: 2,
                            },
                            {
                                label: 'Harga Penutupan Aktual dan Prediksi (Close)',
                                data: close_pred,
                                fill: false,
                                borderColor: 'orange',
                                backgroundColor: '#ffffff',
                                pointRadius: 0,
                                borderWidth: 2,
                            },
                            {
                                label: 'Upperband',
                                data: upper_band,
                                fill: false,
                                borderColor: 'red',
                                backgroundColor: '#ffffff',
                                pointRadius: 0
                            },
                            {
                                label: 'Middleband (SMA)',
                                data: middle_band,
                                fill: false,
                                borderColor: 'black',
                                backgroundColor: '#ffffff',
                                pointRadius: 0
                            },
                            {
                                label: 'Lowerband',
                                data: lower_band,
                                fill: false,
                                borderColor: 'green',
                                backgroundColor: '#ffffff',
                                pointRadius: 0
                            }
                        ]
                    },
                    options: {
                        scales: {
                            y: {
                                beginAtZero: true,
                            },
                        },
                        plugins: {
                            zoom: {
                                zoom: {
                                    drag: {
                                        enabled: true,
                                        borderColor: 'rgba(0,0,0,0.3)',
                                        borderWidth: 1,
                                    },
                                    mode: 'xy',
                                },
                                pan: {
                                    enabled: true,
                                    modifierKey: 'ctrl',
                                },
                            },
                            title: {
                                display: true,
                                text: apiData.data.ticker,
                            },
                            transition: {
                                zoom: {
                                    animation: {
                                        duration: 4000,
                                        easing: 'easeOutCubic'
                                    },
                                },
                            },
                        },
                    }
                });

                return chartLine;
            }

            function resetChart(chartLine) {
                chartLine.resetZoom();
            }

            const tooltipTriggerList = document.querySelectorAll('[data-bs-toggle="tooltip"]')
            const tooltipList = [...tooltipTriggerList].map(tooltipTriggerEl => new bootstrap.Tooltip(
                tooltipTriggerEl))

            var chartCanvas = $('#chrtTicker');
            var tablePrediction = $('#tablePred');
            var data = @json($apiData);

            var chartLine = chartPredict(data, chartCanvas);
            var tablePredict = tablePredict(data, tablePrediction)



            $('#resetZoom').click(function() {
                resetChart(chartLine);
            })
        })
    </script>
@endsection

</body>

</html>

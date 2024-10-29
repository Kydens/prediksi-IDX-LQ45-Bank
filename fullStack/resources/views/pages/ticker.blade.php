@extends('layouts.AppLayouts')

@section('content')
    <div class="wrapper">
        <h3>Harga Penutupan Sahan {{ $ticker }}</h3>
        <div class="col">
            <div class="card p-4">
                <div class="col mb-2">
                    <canvas id='chrtTicker' class="pb-4" style="max-height: 500px; min-width: 100%;"></canvas>
                    <button id='resetZoom' class='btn btn-primary'>Reset Zoom</button>
                </div>
                <div class="col">
                    <hr>
                </div>
                <div class="col">
                    <form action="{{ route('dashboard.predict', ['ticker' => $ticker]) }}" method="GET">
                        @csrf
                        <input type="hidden" value="{{ $ticker }}">
                        <div class="row">
                            <div class="col-md-5 mb-3">
                                <label for="days" class="form-label">Prediksi Hari Ke-n</label>
                                <select id="days" name="days" class="form-select" required>
                                    <option value='' selected>Prediksi Hari ke-n</option>
                                    <option value='7'>7 hari (1 Pekan)</option>
                                    <option value='30'>30 hari (1 Bulan)</option>
                                    <option value='90'>90 hari (3 Bulan)</option>
                                    <option value='120'>180 hari (6 Bulan)</option>
                                    <option value='365'>365 hari (1 Tahun)</option>
                                </select>
                            </div>
                            <div class="col-md-5 mb-3">
                                <label for="window" class="form-label">Window Simple Moving Average (SMA)</label>
                                <select id="window" name="window" class="form-select" required>
                                    <option value='' selected>Window</option>
                                    <option value='20'>20 hari Rolling</option>
                                    <option value='50'>50 hari Rolling</option>
                                </select>
                            </div>
                        </div>
                        <button class="btn btn-dark" type="submit">Predict</button>
                    </form>
                </div>
            </div>
        </div>
    </div>
@endsection

@section('js')
    <script>
        function chartTicker(apiData, canvas) {
            var chartLine = new Chart(canvas, {
                type: 'line',
                data: {
                    labels: apiData.data.index,
                    datasets: [{
                        label: 'Harga Penutupan (Close)',
                        data: apiData.data.close,
                        fill: false,
                        borderColor: 'blue',
                        backgroundColor: '#ffffff',
                        pointRadius: 0
                    }]
                },
                options: {
                    scales: {
                        y: {
                            beginAtZero: true,
                        }
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
                        transition: {
                            zoom: {
                                animation: {
                                    duration: 4000,
                                    easing: 'easeOutCubic'
                                },
                            },
                        },
                    },
                },
            });

            return chartLine;
        }

        function resetChart(chartLine) {
            chartLine.resetZoom();
        }

        $(document).ready(function() {
            var chartCanvas = $('#chrtTicker');
            var data = @json($apiData);

            var chartLine = chartTicker(data, chartCanvas);

            $('#resetZoom').on('click', function() {
                resetChart(chartLine);
            })
        });
    </script>
@endsection

</body>

</html>

<!DOCTYPE html>
<html lang="en">

<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <meta http-equiv="X-UA-Compatible" content="ie=edge">
    <title>Document</title>
</head>

<body>
    <div>
        <h3>Prediction Data</h3>
        {{-- <ul>
            @for ($i = 0; $i < count($apiData['data']['index']); $i++)
                <li>{{ $apiData['data']['index'][$i] }}</li>
                <li>{{ $apiData['data']['close'][$i] }}</li>
            @endfor
        </ul> --}}
        <canvas id='chrt'></canvas>
        <button onClick='resetChart()'>Reset Chart</button>
    </div>

    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>

    <script src="https://cdnjs.cloudflare.com/ajax/libs/chartjs-plugin-zoom/2.0.1/chartjs-plugin-zoom.min.js"
        integrity="sha512-wUYbRPLV5zs6IqvWd88HIqZU/b8TBx+I8LEioQ/UC0t5EMCLApqhIAnUg7EsAzdbhhdgW07TqYDdH3QEXRcPOQ=="
        crossorigin="anonymous" referrerpolicy="no-referrer"></script>

    <script>
        var ctx = document.getElementById('chrt');
        var apiData = @json($apiData)

        const close_actual = apiData.data.df_combined.slice(0, apiData.data.df_combined.length - apiData.data.close_pred
            .length);
        const close_pred = Array(apiData.data.df_combined.length - apiData.data.close_pred.length).fill(null).concat(apiData
            .data.close_pred);
        const upper_band = Array(apiData.data.df_combined.length - apiData.data.upper_band.length).fill(null).concat(apiData
            .data.upper_band);
        const middle_band = Array(apiData.data.df_combined.length - apiData.data.sma_band.length).fill(null).concat(apiData
            .data.sma_band);
        const lower_band = Array(apiData.data.df_combined.length - apiData.data.lower_band.length).fill(null).concat(apiData
            .data.lower_band);

        chartLine = new Chart(ctx, {
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
                    pan: {
                        enabled: true,
                        modifierKey: 'ctrl',
                    },
                    zoom: {
                        zoom: {
                            drag: {
                                enabled: true,
                            },
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

        function resetChart() {
            chartLine.resetZoom()
        }
    </script>
</body>

</html>

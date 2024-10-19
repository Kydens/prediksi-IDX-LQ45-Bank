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
    </div>

    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <script>
        var ctx = document.getElementById('chrt');
        var apiData = @json($apiData)

        new Chart(ctx, {
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
                }
            },
            plugins: {
                zoom: {
                    zoom: {
                        wheel: {
                            enabled: true,
                        },
                        pinch: {
                            enabled: true,
                        }
                    }
                }
            }
        });
    </script>
</body>

</html>

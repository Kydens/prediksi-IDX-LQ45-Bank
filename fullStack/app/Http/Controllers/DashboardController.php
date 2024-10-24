<?php

namespace App\Http\Controllers;

use Illuminate\Http\Request;
use Illuminate\Support\Facades\Http;

class DashboardController extends Controller
{
    public function index()
    {
        return view('pages.home');
    }

    public function ticker($ticker)
    {
        $response = Http::get(
            $_ENV['MICROSERVICES_API_URL'] . $ticker
        );
        $apiData = json_decode($response, true);

        return view('pages.ticker', compact('apiData', 'ticker'));
    }

    public function predict(Request $request, $ticker)
    {
        $request->validate([
            'days'=>'required|integer',
            'window'=>'required|integer'
        ]);

        $days = $request->input('days');
        $window = $request->input('window');

        $response = Http::get(
            $_ENV['MICROSERVICES_API_URL'] . $ticker . '/predict?days=' . $days . '&window=' . $window
        );
        $apiData = json_decode($response, true);

        return view('pages.predict', compact('apiData', 'ticker','days','window'));
    }
}

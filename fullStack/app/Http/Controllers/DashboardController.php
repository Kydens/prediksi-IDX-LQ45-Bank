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
            'http://localhost:5000/api/' . $ticker
        );
        $apiData = json_decode($response, true);

        return view('pages.ticker', compact('apiData'));
    }

    public function predict(Request $request, $ticker)
    {
        $days = $request->query('days');
        $window = $request->query('window');

        $response = Http::get(
            'http://localhost:5000/api/' . $ticker . '/predict?days=' . $days . '&window=' . $window
        );
        $apiData = json_decode($response, true);

        return view('pages.predict', compact('apiData'));
    }
}

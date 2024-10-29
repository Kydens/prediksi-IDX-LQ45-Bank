<?php

namespace App\Http\Controllers;

use Illuminate\Http\Request;
use Illuminate\Support\Facades\Http;

class DashboardController extends Controller
{
    public function index()
    {
        $apiKey = env('APIKEY_SERPAPIGOOGLENEWS');
        $articles = [];

        $guzzleClient = new \GuzzleHttp\Client(['base_uri'=>'https://serpapi.com', 'verify'=>false]);

        try {
            $response = $guzzleClient->get('/search?google_news',['query'=>['q'=>'saham','gl'=>'id','hl'=>'id','api_key'=>$apiKey]]);

            $data = json_decode($response->getBody(), true);

            if(isset($data['top_stories']) && is_array($data['top_stories'])) {
                $articles = $data['top_stories'];

                $articles = array_map(function($article) {
                    return [
                        'title' => $article['title'] ?? '',
                        'url' => $article['link'] ?? '',
                        'urlToImage' => $article['thumbnail'] ?? '',
                        'publishedAt' => $article['date'],
                        'source' => $article['source'] ?? '',
                    ];
                }, $articles);
            }

        } catch (\GuzzleHttp\Exception\GuzzleException $e) {
            dd($e->getMessage());
        }

        return view('pages.home', compact('articles'));
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

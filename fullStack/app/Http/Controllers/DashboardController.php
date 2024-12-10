<?php

namespace App\Http\Controllers;

use App\Support\ArrayPaginator;
use Exception;
use Illuminate\Http\Request;
use Illuminate\Support\Facades\Http;

class DashboardController extends Controller
{
    private $serpapiAPIKey;
    private $urlMicroservices;


    public function __construct()
    {
        $this->serpapiAPIKey = config('services.api.serpapi_api_key');
        $this->urlMicroservices = config('services.api.microservices_api');
    }

    public function index()
    {
        $allArticlesTopStories = $this->serpapiAPITopStories($this->serpapiAPIKey);
        $articlesTopStories = ArrayPaginator::paginate($allArticlesTopStories, 8);

        return view('pages.home', compact('articlesTopStories'));
    }

    public function show()
    {
        return view ('pages.show');
    }

    public function news()
    {
        try {
            $allArticlesTopStories = $this->serpapiAPITopStories($this->serpapiAPIKey);
            $allArticlesGNews = $this->serpapiAPIGNews($this->serpapiAPIKey);
        } catch (Exception $e) {
            session()->flash('error', 'Berita sedang tidak tersedia untuk saat ini.');
            return view('pages.news');
        }

        return view ('pages.news', compact('allArticlesTopStories', 'allArticlesGNews'));
    }

    public function help()
    {
        return view('pages.help');
    }

    public function viewTicker($ticker)
    {
        try {
            $response = Http::timeout(3)->get(
                $this->urlMicroservices . $ticker
            );

            if ($response->status() === 503) {
                return response()->view('layouts.error.503', [], 503);
            }

            $apiData = json_decode($response, true);
        } catch (Exception $e) {
            return response()->view('layouts.error.503', [],503);
        }

        return view('pages.ticker.show', compact('apiData', 'ticker'));
    }

    public function predictTicker(Request $request, $ticker)
    {
        $request->validate([
            'window'=>'required|integer'
        ]);

        $days = 7;
        $window = $request->input('window');

        try {
            $response = Http::timeout(3)->get(
            $this->urlMicroservices . $ticker . '/predict?days=' . $days . '&window=' . $window
            );

            if ($response->status() === 503) {
                return response()->view('layouts.error.503', [], 503);
            }

            $apiData = json_decode($response, true);
        } catch (Exception $e) {
            return response()->view('layouts.error.503', [],503);
        }


        return view('pages.ticker.predict', compact('apiData', 'ticker','days','window'));
    }

    private function serpapiAPIGNews(string $apiKey)
    {
        $articles = [];

        $guzzleClient = new \GuzzleHttp\Client([
            'base_uri'=>'https://serpapi.com',
            'verify'=>false
        ]);

        try {
            $response = $guzzleClient->get('/search',[
                'query'=>[
                    'tbm'=>'nws',
                    'q'=>'saham IDX',
                    'gl'=>'id',
                    'hl'=>'id',
                    'num'=>40,
                    'safe'=>'active',
                    'api_key'=>$apiKey,
                ]
            ]);

            $data = json_decode($response->getBody(), true);

            // dd($data);

            if(isset($data['news_results']) && is_array($data['news_results'])) {
                $articles = $data['news_results'];

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

        return $articles;
    }

    private function serpapiAPITopStories(string $apiKey)
    {
        $articles = [];

        $guzzleClient = new \GuzzleHttp\Client([
            'base_uri'=>'https://serpapi.com',
            'verify'=>false
        ]);

        try {
            $response = $guzzleClient->get('/search?google_news',[
                'query'=>[
                    'q'=>'saham LQ45',
                    'gl'=>'id',
                    'hl'=>'id',
                    'api_key'=>$apiKey,
                ]
            ]);

            $data = json_decode($response->getBody(), true);

            // dd($data);

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

        return $articles;
    }
}

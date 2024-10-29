<?php

namespace App\Http\Controllers;

use Illuminate\Http\Request;
use Illuminate\Support\Facades\Http;

class NewsController extends Controller
{
    public function getNews() {
        $apiKey = $_ENV['APIKEY_NEWSAPI'];
        $response = Http::get('https://newsapi.org/v2/everything',[
            'q'=>'saham',
            'country'=>'id',
            'apiKey'=>$apiKey,
        ]);

        $news = $response->json();

        return $news;
    }
}

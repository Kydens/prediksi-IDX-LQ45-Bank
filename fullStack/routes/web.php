<?php

use App\Http\Controllers\DashboardController;
use App\Http\Controllers\NewsController;
use Illuminate\Support\Facades\Route;

/*
|--------------------------------------------------------------------------
| Web Routes
|--------------------------------------------------------------------------
|
| Here is where you can register web routes for your application. These
| routes are loaded by the RouteServiceProvider and all of them will
| be assigned to the "web" middleware group. Make something great!
|
*/

// Route::get('/', function () {
//     return view('welcome');
// });


Route::get('/', [DashboardController::class,'index'])->name('dashboard.index');

Route::get('/idx45-bank', [DashboardController::class,'show'])->name('dashboard.show');

Route::get('/berita', [DashboardController::class,'news'])->name('dashboard.news');

Route::get('/idx45-bank/{ticker}', [DashboardController::class,'viewTicker'])->name('dashboard.viewTicker');

Route::get('/idx45-bank/{ticker}/predict', [DashboardController::class,'predictTicker'])->name('dashboard.predictTicker');

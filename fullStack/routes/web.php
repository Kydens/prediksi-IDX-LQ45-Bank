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

Route::get('/{ticker}', [DashboardController::class,'ticker'])->name('dashboard.ticker');

Route::get('/{ticker}/predict', [DashboardController::class,'predict'])->name('dashboard.predict');

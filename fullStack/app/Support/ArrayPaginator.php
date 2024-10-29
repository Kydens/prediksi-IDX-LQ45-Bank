<?php

namespace App\Support;

use Illuminate\Pagination\LengthAwarePaginator;

class ArrayPaginator extends LengthAwarePaginator
{
    public function __construct($items, $total, $perPage, $currentPage = null, array $options = [])
    {
        parent::__construct($items, $total, $perPage, $currentPage, $options);
    }

    public static function paginate($items, $perPage)
    {
        $currentPage = LengthAwarePaginator::resolveCurrentPage();

        $currentItems = array_slice($items, ($currentPage - 1) * $perPage, $perPage);

        return new static(
            $currentItems,
            count($items),
            $perPage,
            $currentPage,
            [
                'path' => LengthAwarePaginator::resolveCurrentPath(),
                'pageName' => 'page',
            ]
        );
    }
}

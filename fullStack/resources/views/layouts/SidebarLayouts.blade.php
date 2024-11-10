<div class="custom-sidebar-wrapper">
    <div class="sidebar">
        <div class="logo_details">
            <i class="bx bxl-audible icon"></i>
            <div class="logo_name">Prediction Stock Market</div>
            <i class="bx bx-menu" id="btn"></i>
        </div>
        <ul class="navbar-list">
            {{-- <li>
                <i class="bx bx-search"></i>
                <input type="text" placeholder="Search...">
                <span class="tooltip">Search</span>
            </li> --}}
            <li>
                <a href="{{ route('dashboard.index') }}">
                    <i class="bx bx-grid-alt" style="font-size: 24px"></i>
                    <span class="link_name">Dashboard</span>
                </a>
                <span class="tooltip">Dashboard</span>
            </li>
            <li>
                <a href="{{ route('dashboard.show') }}">
                    <i class="bx bx-stats" style="font-size: 24px"></i>
                    <span class="link_name">Pilihan Saham</span>
                </a>
                <span class="tooltip">Pilihan Saham</span>
            </li>
            <li>
                <a href="{{ route('dashboard.news') }}">
                    <i class="bx bx-news" style="font-size: 24px"></i>
                    <span class="link_name">Berita</span>
                </a>
                <span class="tooltip">Berita</span>
            </li>
            <li>
                <a href="{{ route('dashboard.help') }}">
                    <i class="bx bx-help-circle" style="font-size: 24px"></i>
                    <span class="link_name">Bantuan</span>
                </a>
                <span class="tooltip">Bantuan</span>
            </li>
            {{-- <li>
                <a href="#">
                    <i class="bx bx-folder"></i>
                    <span class="link_name">File Manger</span>
                </a>
                <span class="tooltip">File Manger</span>
            </li>
            <li>
                <a href="#">
                    <i class="bx bx-cart-alt"></i>
                    <span class="link_name">Order</span>
                </a>
                <span class="tooltip">Order</span>
            </li>
            <li>
                <a href="#">
                    <i class="bx bx-cog"></i>
                    <span class="link_name">Settings</span>
                </a>
                <span class="tooltip">Settings</span>
            </li> --}}
        </ul>
    </div>

</div>

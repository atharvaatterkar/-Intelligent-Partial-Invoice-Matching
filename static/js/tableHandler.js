class TableHandler {
    constructor() {
        this.data = [];
        this.currentPage = 1;
        this.rowsPerPage = 10;
        this.table = document.querySelector('.account-table tbody');
        this.pagination = document.querySelector('.pagination');
    }

    setData(data) {
        this.data = data;
        this.render();
    }

    createTableRow(item) {
        return `
            <tr>
                <td>${item.sourceInvoice}</td>
                <td><i class="bi bi-arrow-right"></i></td>
                <td>${item.targetInvoice}</td>
                <td>${item.confidence}%</td>
                <td>
                    <span class="badge text-bg-${item.confidence >= 70 ? 'success' : 'warning'}">
                        ${item.confidence >= 70 ? 'Exact Match' : 'Partial Match'}
                    </span>
                    <div class="action-buttons">
                        <button class="btn btn-success btn-sm">
                            <i class="bi bi-check"></i>
                        </button>
                        <button class="btn btn-danger btn-sm">
                            <i class="bi bi-x"></i>
                        </button>
                    </div>
                </td>
            </tr>
        `;
    }

    renderTable() {
        const start = (this.currentPage - 1) * this.rowsPerPage;
        const end = start + this.rowsPerPage;
        const pageData = this.data.slice(start, end);
        
        this.table.innerHTML = pageData.map(item => this.createTableRow(item)).join('');
    }

    renderPagination() {
        const totalPages = Math.ceil(this.data.length / this.rowsPerPage);
        
        if (totalPages <= 1) {
            this.pagination.style.display = 'none';
            return;
        }

        this.pagination.style.display = 'flex';
        const pages = Array.from({length: totalPages}, (_, i) => i + 1);
        
        const paginationHTML = `
            <li class="page-item ${this.currentPage === 1 ? 'disabled' : ''}">
                <a class="page-link prev-page" href="#">Prev</a>
            </li>
            ${pages.map(page => `
                <li class="page-item ${page === this.currentPage ? 'active' : ''}">
                    <a class="page-link page-number" href="#" data-page="${page}">${page}</a>
                </li>
            `).join('')}
            <li class="page-item ${this.currentPage === totalPages ? 'disabled' : ''}">
                <a class="page-link next-page" href="#">Next</a>
            </li>
        `;
        
        this.pagination.innerHTML = paginationHTML;
    }

    render() {
        this.renderTable();
        this.renderPagination();
    }

    bindEvents() {
        this.pagination.addEventListener('click', (e) => {
            e.preventDefault();
            if (e.target.classList.contains('page-number')) {
                this.currentPage = parseInt(e.target.dataset.page);
                this.render();
            } else if (e.target.classList.contains('prev-page') && this.currentPage > 1) {
                this.currentPage--;
                this.render();
            } else if (e.target.classList.contains('next-page') && 
                      this.currentPage < Math.ceil(this.data.length / this.rowsPerPage)) {
                this.currentPage++;
                this.render();
            }
        });
    }
}
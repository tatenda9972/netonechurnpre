// Admin Dashboard JavaScript

document.addEventListener('DOMContentLoaded', function() {
    // Initialize tooltips
    var tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
    var tooltipList = tooltipTriggerList.map(function (tooltipTriggerEl) {
        return new bootstrap.Tooltip(tooltipTriggerEl);
    });
    
    // Handle confirmation dialogs
    const confirmActions = document.querySelectorAll('[data-confirm]');
    confirmActions.forEach(element => {
        element.addEventListener('click', function(event) {
            if (!confirm(this.getAttribute('data-confirm'))) {
                event.preventDefault();
            }
        });
    });
    
    // Handle user edit modal
    const editUserModal = document.getElementById('editUserModal');
    if (editUserModal) {
        editUserModal.addEventListener('show.bs.modal', function(event) {
            const button = event.relatedTarget;
            const userId = button.getAttribute('data-user-id');
            const firstName = button.getAttribute('data-user-first-name');
            const lastName = button.getAttribute('data-user-last-name');
            const email = button.getAttribute('data-user-email');
            const isAdmin = button.getAttribute('data-user-is-admin') === 'True';
            
            editUserModal.querySelector('#edit_user_id').value = userId;
            editUserModal.querySelector('#edit_first_name').value = firstName;
            editUserModal.querySelector('#edit_last_name').value = lastName;
            editUserModal.querySelector('#edit_email').value = email;
            editUserModal.querySelector('#edit_is_admin').checked = isAdmin;
        });
    }
    
    // Handle prediction delete modal
    const deletePredictionModal = document.getElementById('deletePredictionModal');
    if (deletePredictionModal) {
        deletePredictionModal.addEventListener('show.bs.modal', function(event) {
            const button = event.relatedTarget;
            const predictionId = button.getAttribute('data-prediction-id');
            const predictionFile = button.getAttribute('data-prediction-file');
            
            deletePredictionModal.querySelector('#delete_prediction_id').value = predictionId;
            deletePredictionModal.querySelector('#delete_prediction_file').textContent = predictionFile;
        });
    }
    
    // Handle user delete modal
    const deleteUserModal = document.getElementById('deleteUserModal');
    if (deleteUserModal) {
        deleteUserModal.addEventListener('show.bs.modal', function(event) {
            const button = event.relatedTarget;
            const userId = button.getAttribute('data-user-id');
            const userName = button.getAttribute('data-user-name');
            
            deleteUserModal.querySelector('#delete_user_id').value = userId;
            deleteUserModal.querySelector('#delete_user_name').textContent = userName;
        });
    }
    
    // Handle user search filter
    const userSearchInput = document.getElementById('user-search');
    if (userSearchInput) {
        userSearchInput.addEventListener('input', function() {
            const searchTerm = this.value.toLowerCase();
            const tableRows = document.querySelectorAll('tbody tr');
            
            tableRows.forEach(row => {
                const name = row.cells[1].textContent.toLowerCase();
                const email = row.cells[2].textContent.toLowerCase();
                
                if (name.includes(searchTerm) || email.includes(searchTerm)) {
                    row.style.display = '';
                } else {
                    row.style.display = 'none';
                }
            });
        });
    }
    
    // Handle prediction search filter
    const predictionSearchInput = document.getElementById('prediction-search');
    if (predictionSearchInput) {
        predictionSearchInput.addEventListener('input', function() {
            const searchTerm = this.value.toLowerCase();
            const tableRows = document.querySelectorAll('tbody tr');
            
            tableRows.forEach(row => {
                const user = row.cells[1].textContent.toLowerCase();
                const filename = row.cells[2].textContent.toLowerCase();
                
                if (user.includes(searchTerm) || filename.includes(searchTerm)) {
                    row.style.display = '';
                } else {
                    row.style.display = 'none';
                }
            });
        });
    }
    
    // Reset filters
    const resetFiltersButton = document.getElementById('reset-filters');
    if (resetFiltersButton) {
        resetFiltersButton.addEventListener('click', function() {
            // Reset search input if exists
            const searchInput = document.getElementById('user-search') || document.getElementById('prediction-search');
            if (searchInput) {
                searchInput.value = '';
            }
            
            // Reset any dropdowns
            const dropdowns = document.querySelectorAll('select');
            dropdowns.forEach(dropdown => {
                dropdown.selectedIndex = 0;
            });
            
            // Show all table rows
            const tableRows = document.querySelectorAll('tbody tr');
            tableRows.forEach(row => {
                row.style.display = '';
            });
        });
    }
});
document.addEventListener('DOMContentLoaded', function() {
    const form = document.getElementById('news-form');
    const clearBtn = document.getElementById('clear-btn');
    const textarea = document.getElementById('news-text');
    const sampleButtons = document.querySelectorAll('.sample-btn');
    
    // Initialize animation indices for result page elements
    const termItems = document.querySelectorAll('.term-item');
    const tipItems = document.querySelectorAll('.tips ul li');
    
    // Set animation indices for term items
    if (termItems.length > 0) {
        termItems.forEach((item, index) => {
            item.style.setProperty('--index', index);
        });
    }
    
    // Set animation indices for tip items
    if (tipItems.length > 0) {
        tipItems.forEach((item, index) => {
            item.style.setProperty('--index', index);
        });
    }
    
    // Sample news articles for testing
    const sampleNews = {
        fake: "BREAKING NEWS: Scientists discover that drinking coffee turns people into zombies. A recent study conducted by the Institute of Beverage Research claims that consuming more than 3 cups of coffee per day can lead to zombie-like behavior. The government is considering banning coffee nationwide. Coffee shop owners are protesting, claiming this is a conspiracy by the tea industry.",
        real: "Researchers at Stanford University have published a new study in the Journal of Medicine showing that regular exercise can reduce the risk of heart disease by up to 30%. The study, which followed 5,000 participants over a 10-year period, found that even moderate physical activity for 30 minutes a day, five days a week, can have significant health benefits. The researchers recommend a combination of cardio and strength training for optimal results."
    };
    
    // Add animation classes to sample buttons
    if (sampleButtons.length > 0) {
        sampleButtons.forEach((button, index) => {
            button.style.setProperty('--index', index);
        });
    }
    
    // Form submission
    if (form) {
        form.addEventListener('submit', function(e) {
            if (!validateForm()) {
                e.preventDefault();
            } else {
                // Show loading indicator
                showLoading();
            }
        });
    }
    
    // Clear text area
    if (clearBtn) {
        clearBtn.addEventListener('click', function() {
            textarea.value = '';
            textarea.focus();
            // Add a subtle animation to the textarea
            textarea.classList.add('pulse-effect');
            setTimeout(() => {
                textarea.classList.remove('pulse-effect');
            }, 500);
        });
    }
    
    // Load sample news
    if (sampleButtons.length > 0) {
        sampleButtons.forEach(button => {
            button.addEventListener('click', function() {
                const newsType = this.getAttribute('data-type');
                
                // Add a subtle animation to the button
                this.classList.add('active');
                setTimeout(() => {
                    this.classList.remove('active');
                }, 300);
                
                // Animate the text change
                if (textarea.value !== '') {
                    // If there's text, fade it out first
                    textarea.classList.add('fade-out');
                    setTimeout(() => {
                        textarea.value = sampleNews[newsType];
                        textarea.classList.remove('fade-out');
                        textarea.classList.add('fade-in');
                        setTimeout(() => {
                            textarea.classList.remove('fade-in');
                        }, 300);
                    }, 200);
                } else {
                    // If empty, just fade in
                    textarea.value = sampleNews[newsType];
                    textarea.classList.add('fade-in');
                    setTimeout(() => {
                        textarea.classList.remove('fade-in');
                    }, 300);
                }
                
                textarea.focus();
            });
        });
    }
    
    // Form validation with improved feedback
    function validateForm() {
        const newsText = textarea.value.trim();
        
        if (newsText === '') {
            showError('Please enter some news text to analyze.');
            textarea.focus();
            return false;
        }
        
        if (newsText.length < 50) {
            showError('Please enter at least 50 characters for accurate analysis.');
            textarea.focus();
            return false;
        }
        
        return true;
    }
    
    // Improved error feedback
    function showError(message) {
        // Check if error message already exists
        const existingError = document.querySelector('.error-message');
        if (existingError) {
            existingError.remove();
        }
        
        const errorDiv = document.createElement('div');
        errorDiv.className = 'error-message';
        errorDiv.textContent = message;
        
        // Insert after textarea
        textarea.parentNode.insertBefore(errorDiv, textarea.nextSibling);
        
        // Animate the error message
        errorDiv.style.animation = 'fadeInDown 0.5s ease-out forwards';
        
        // Auto-dismiss after 5 seconds
        setTimeout(() => {
            if (errorDiv.parentNode) {
                errorDiv.style.animation = 'fadeOut 0.5s ease-out forwards';
                setTimeout(() => {
                    if (errorDiv.parentNode) {
                        errorDiv.remove();
                    }
                }, 500);
            }
        }, 5000);
    }
    
    // Enhanced loading indicator
    function showLoading() {
        const loadingOverlay = document.createElement('div');
        loadingOverlay.className = 'loading-overlay';
        
        const loadingContainer = document.createElement('div');
        loadingContainer.className = 'loading-container';
        
        const spinner = document.createElement('div');
        spinner.className = 'spinner';
        
        const loadingText = document.createElement('div');
        loadingText.className = 'loading-text';
        loadingText.textContent = 'Analyzing...';
        
        loadingContainer.appendChild(spinner);
        loadingContainer.appendChild(loadingText);
        loadingOverlay.appendChild(loadingContainer);
        document.body.appendChild(loadingOverlay);
        
        // Add fade-in animation
        setTimeout(() => {
            loadingOverlay.style.opacity = '1';
        }, 10);
    }
    
    // Add these styles for the loading overlay
    const style = document.createElement('style');
    style.textContent = `
        .loading-overlay {
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background-color: rgba(255, 255, 255, 0.8);
            display: flex;
            flex-direction: column;
            justify-content: center;
            align-items: center;
            z-index: 1000;
            opacity: 0;
            transition: opacity 0.3s ease;
        }
        
        .spinner {
            width: 50px;
            height: 50px;
            border: 5px solid #f3f3f3;
            border-top: 5px solid #3498db;
            border-radius: 50%;
            animation: spin 1s linear infinite;
            margin-bottom: 1rem;
        }
        
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
        
        .pulse-effect {
            animation: pulse 0.5s ease;
        }
        
        @keyframes pulse {
            0% { box-shadow: 0 0 0 0 rgba(52, 152, 219, 0.4); }
            70% { box-shadow: 0 0 0 10px rgba(52, 152, 219, 0); }
            100% { box-shadow: 0 0 0 0 rgba(52, 152, 219, 0); }
        }
        
        .fade-in {
            animation: fadeIn 0.3s ease forwards;
        }
        
        .fade-out {
            animation: fadeOut 0.2s ease forwards;
        }
        
        @keyframes fadeIn {
            from { opacity: 0; }
            to { opacity: 1; }
        }
        
        @keyframes fadeOut {
            from { opacity: 1; }
            to { opacity: 0; }
        }
        
        .error-message {
            color: #e74c3c;
            margin-top: 0.5rem;
            padding: 0.5rem;
            border-left: 3px solid #e74c3c;
            background-color: rgba(231, 76, 60, 0.1);
            border-radius: 0 3px 3px 0;
        }
        
        @keyframes fadeInDown {
            from { opacity: 0; transform: translateY(-10px); }
            to { opacity: 1; transform: translateY(0); }
        }
        
        @keyframes fadeOut {
            from { opacity: 1; }
            to { opacity: 0; }
        }
        
        .sample-btn.active {
            transform: scale(0.95);
            box-shadow: 0 0 0 3px rgba(52, 152, 219, 0.3);
            transition: all 0.3s ease;
        }
    `;
    document.head.appendChild(style);
});
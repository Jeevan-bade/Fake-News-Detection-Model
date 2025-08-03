document.addEventListener('DOMContentLoaded', function () {
    const form = document.getElementById('news-form');
    const clearBtn = document.getElementById('clear-btn');
    const textarea = document.getElementById('news-text');
    const sampleButtons = document.querySelectorAll('.sample-btn');
    const resultBox = document.getElementById('result-box');
    const errorContainer = document.getElementById('error-container');
    const loadingOverlay = document.querySelector('.loading-overlay');
    const submitButton = form.querySelector('button[type="submit"]');
    const loadingSpinner = submitButton.querySelector('.loading');

    const sampleNews = {
        fake: `BREAKING NEWS: Scientists discover that drinking coffee turns people into zombies. A recent study conducted by the Institute of Beverage Research claims that consuming more than 3 cups of coffee per day can lead to zombie-like behavior. The government is considering banning coffee nationwide. Coffee shop owners are protesting, claiming this is a conspiracy by the tea industry.`
    };

    function showLoading() {
        loadingSpinner.style.display = 'inline-block';
        loadingOverlay.style.display = 'flex';
        submitButton.disabled = true;
    }

    function hideLoading() {
        loadingSpinner.style.display = 'none';
        loadingOverlay.style.display = 'none';
        submitButton.disabled = false;
    }

    function showError(message) {
        errorContainer.innerHTML = `<div class="error-message">${message}</div>`;
        resultBox.style.display = 'none';
    }

    function clearError() {
        errorContainer.innerHTML = '';
    }

    function displayResult(result) {
        resultBox.style.display = 'block';
        resultBox.className = `result-box ${result.prediction.toLowerCase().replace(' ', '-')}`;
        
        resultBox.querySelector('.result-title').textContent = 
            result.prediction === 'Fake News' ? '🚫 Fake News Detected' : '✓ Real News Verified';
        
        resultBox.querySelector('.prediction').innerHTML = 
            `<strong>${result.prediction}</strong>`;
        
        resultBox.querySelector('.confidence').innerHTML = 
            `Confidence: <strong>${result.confidence}</strong>`;
        
        resultBox.querySelector('.explanation-text').innerHTML = 
            result.explanation.replace(/\n/g, '<br>');
        
        resultBox.querySelector('.terms-text').textContent = 
            result.key_terms ? result.key_terms.join(', ') : 'No key terms available';
        
        resultBox.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }

    if (sampleButtons.length > 0) {
        sampleButtons.forEach(button => {
            button.addEventListener('click', async function () {
                const type = this.getAttribute('data-type');
                clearError();
                resultBox.style.display = 'none';
                
                if (type === 'real') {
                    try {
                        const response = await fetch('/sample_real');
                        if (!response.ok) throw new Error('Failed to fetch sample real news');
                        const data = await response.json();
                        textarea.value = data.sample;
                    } catch (error) {
                        console.error('Error fetching sample real news:', error);
                        showError('Failed to load sample real news. Please try again.');
                    }
                } else {
                    textarea.value = sampleNews[type];
                }
                textarea.focus();
            });
        });
    }

    if (clearBtn) {
        clearBtn.addEventListener('click', function () {
            textarea.value = '';
            clearError();
            resultBox.style.display = 'none';
            textarea.focus();
        });
    }

    form.addEventListener('submit', async function (e) {
        e.preventDefault();
        const text = textarea.value.trim();
        clearError();

        if (!text) {
            showError('Please enter some news text to analyze.');
            textarea.focus();
            return;
        } 
        
        if (text.length < 50) {
            showError('Please enter at least 50 characters for accurate analysis.');
            textarea.focus();
            return;
        }

        showLoading();

        try {
            const response = await fetch('/predict', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ text: text })
            });

            if (!response.ok) {
                throw new Error('Server error occurred');
            }

            const result = await response.json();
            if (result.error) {
                showError(result.error);
            } else {
                displayResult(result);
            }
        } catch (error) {
            console.error('Error:', error);
            showError('An error occurred while analyzing the text. Please try again.');
        } finally {
            hideLoading();
        }
    });
});

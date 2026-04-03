document.addEventListener('DOMContentLoaded', () => {
    // Scroll reveal animation logic using Intersection Observer API
    const revealElements = document.querySelectorAll('.reveal');

    const revealOptions = {
        threshold: 0.15,
        rootMargin: "0px 0px -50px 0px"
    };

    const revealOnScroll = new IntersectionObserver((entries, observer) => {
        entries.forEach(entry => {
            if (!entry.isIntersecting) {
                return;
            } else {
                entry.target.classList.add('active');
                observer.unobserve(entry.target);
            }
        });
    }, revealOptions);

    revealElements.forEach(el => {
        revealOnScroll.observe(el);
    });
    
    // Add staggered delay for the initial header cards load
    const headerCards = document.querySelectorAll('.header-card');
    headerCards.forEach((card, index) => {
        card.style.transitionDelay = `${0.3 + (index * 0.15)}s`;
    });

    // Dynamic Grad-CAM generation logic
    const generateBtn = document.getElementById('generateBtn');
    const loadingIndicator = document.getElementById('loadingIndicator');
    const dynamicResult = document.getElementById('dynamicResult');
    const dynamicImage = document.getElementById('dynamicImage');
    const trueLabels = document.getElementById('trueLabels');
    const predLabels = document.getElementById('predLabels');

    if (generateBtn) {
        generateBtn.addEventListener('click', async () => {
            // Disable button and show loading state
            generateBtn.disabled = true;
            loadingIndicator.style.display = 'flex';
            dynamicResult.style.display = 'none';

            try {
                const response = await fetch('/api/random_gradcam', {
                    method: 'POST',
                });
                
                if (!response.ok) {
                    throw new Error(`Server returned ${response.status}`);
                }
                
                const data = await response.json();
                
                if (data.error) {
                    alert(`Error: ${data.error}`);
                } else {
                    dynamicImage.src = data.image;
                    trueLabels.textContent = data.true_labels;
                    predLabels.textContent = data.predicted_labels;
                    
                    // Reveal the newly generated view
                    dynamicResult.style.display = 'block';
                    dynamicResult.style.animation = 'fadeIn 0.5s';
                }

            } catch (error) {
                console.error("Grad-CAM Generation Failed:", error);
                alert("Failed to generate Grad-CAM. Check if the backend is running.");
            } finally {
                // Restore button state
                generateBtn.disabled = false;
                loadingIndicator.style.display = 'none';
            }
        });
    }
});

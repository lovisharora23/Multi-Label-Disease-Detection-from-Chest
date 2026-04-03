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

    // Static Grad-CAM generation cache (No backend required)
    const demoData = [{"image": "assets/demo/patient_0.png", "true_labels": "Atelectasis", "predicted_labels": "No Finding (Low Confidence)"}, {"image": "assets/demo/patient_1.png", "true_labels": "No Finding", "predicted_labels": "No Finding (Low Confidence)"}, {"image": "assets/demo/patient_2.png", "true_labels": "No Finding", "predicted_labels": "No Finding (Low Confidence)"}, {"image": "assets/demo/patient_3.png", "true_labels": "No Finding", "predicted_labels": "Emphysema, Infiltration, Pneumothorax"}, {"image": "assets/demo/patient_4.png", "true_labels": "Emphysema, Mass", "predicted_labels": "Atelectasis, Fibrosis, Hernia"}, {"image": "assets/demo/patient_5.png", "true_labels": "No Finding", "predicted_labels": "Cardiomegaly, Consolidation, Edema, Infiltration, Pneumonia"}, {"image": "assets/demo/patient_6.png", "true_labels": "No Finding", "predicted_labels": "Effusion, Emphysema, Pleural_Thickening, Pneumothorax"}, {"image": "assets/demo/patient_7.png", "true_labels": "No Finding", "predicted_labels": "Cardiomegaly, Hernia"}, {"image": "assets/demo/patient_8.png", "true_labels": "No Finding", "predicted_labels": "Consolidation, Effusion, Infiltration, Nodule, Pleural_Thickening, Pneumothorax"}, {"image": "assets/demo/patient_9.png", "true_labels": "No Finding", "predicted_labels": "Cardiomegaly, Consolidation, Edema, Effusion, Infiltration, Pneumonia"}, {"image": "assets/demo/patient_10.png", "true_labels": "Cardiomegaly, Effusion", "predicted_labels": "Cardiomegaly"}, {"image": "assets/demo/patient_11.png", "true_labels": "Infiltration", "predicted_labels": "No Finding (Low Confidence)"}, {"image": "assets/demo/patient_12.png", "true_labels": "No Finding", "predicted_labels": "Pneumothorax"}, {"image": "assets/demo/patient_13.png", "true_labels": "Fibrosis, Infiltration, Nodule, Pleural_Thickening", "predicted_labels": "Atelectasis, Fibrosis, Pleural_Thickening"}, {"image": "assets/demo/patient_14.png", "true_labels": "Atelectasis, Effusion", "predicted_labels": "Atelectasis, Cardiomegaly, Effusion, Pneumothorax"}, {"image": "assets/demo/patient_15.png", "true_labels": "No Finding", "predicted_labels": "No Finding (Low Confidence)"}, {"image": "assets/demo/patient_16.png", "true_labels": "Atelectasis, Edema, Infiltration", "predicted_labels": "Atelectasis, Effusion"}, {"image": "assets/demo/patient_17.png", "true_labels": "No Finding", "predicted_labels": "No Finding (Low Confidence)"}, {"image": "assets/demo/patient_18.png", "true_labels": "Infiltration", "predicted_labels": "No Finding (Low Confidence)"}, {"image": "assets/demo/patient_19.png", "true_labels": "No Finding", "predicted_labels": "Atelectasis, Consolidation, Effusion, Infiltration"}];
    let currentIndex = 0;

    const generateBtn = document.getElementById('generateBtn');
    const loadingIndicator = document.getElementById('loadingIndicator');
    const dynamicResult = document.getElementById('dynamicResult');
    const dynamicImage = document.getElementById('dynamicImage');
    const trueLabels = document.getElementById('trueLabels');
    const predLabels = document.getElementById('predLabels');

    if (generateBtn) {
        generateBtn.addEventListener('click', () => {
            // Disable button and show artificial loading state for effect
            generateBtn.disabled = true;
            loadingIndicator.style.display = 'flex';
            dynamicResult.style.display = 'none';

            setTimeout(() => {
                const data = demoData[currentIndex];
                currentIndex = (currentIndex + 1) % demoData.length;
                
                dynamicImage.src = data.image;
                trueLabels.textContent = data.true_labels;
                predLabels.textContent = data.predicted_labels;
                
                // Reveal the newly generated view
                dynamicResult.style.display = 'block';
                dynamicResult.style.animation = 'fadeIn 0.5s';
                
                generateBtn.disabled = false;
                loadingIndicator.style.display = 'none';
            }, 600); // 0.6 second dramatic loading delay
        });
    }
});

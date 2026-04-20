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
    
    // ==========================================================================
    // BROWSER-SIDE AI INFERENCE (ONNX RUNTIME)
    // ==========================================================================
    const LABELS = [
        'Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Effusion', 
        'Emphysema', 'Fibrosis', 'Hernia', 'Infiltration', 'Mass', 'Nodule', 
        'Pleural_Thickening', 'Pneumonia', 'Pneumothorax'
    ];

    let session = null;

    // Load ONNX Session
    async function initModel() {
        try {
            if (!session) {
                session = await ort.InferenceSession.create('./model.onnx', { executionProviders: ['webgl', 'wasm'] });
                console.log('✔ AI Model Loaded Successfully');
            }
        } catch (e) {
            console.error('❌ Failed to load AI model:', e);
        }
    }

    const xrayUpload = document.getElementById('xrayUpload');
    const aiLoading = document.getElementById('aiLoading');
    const aiResults = document.getElementById('aiResults');
    const previewImage = document.getElementById('previewImage');
    const findingBars = document.getElementById('findingBars');
    const heatmapCanvas = document.getElementById('heatmapCanvas');
    const heatmapToggle = document.getElementById('heatmapToggle');

    if (xrayUpload) {
        xrayUpload.addEventListener('change', async (e) => {
            const file = e.target.files[0];
            if (!file) return;

            previewImage.src = URL.createObjectURL(file);
            aiLoading.style.display = 'flex';
            aiResults.style.display = 'none';
            
            await initModel();
            
            const img = new Image();
            img.src = previewImage.src;
            img.onload = async () => {
                const tensor = await preprocess(img);
                const { probs, heatmap } = await runInference(tensor);
                
                displayResults(probs);
                renderHeatmap(heatmap);
                
                aiLoading.style.display = 'none';
                aiResults.style.display = 'block';
                
                // Toggle heatmap visibility
                heatmapCanvas.style.display = heatmapToggle.checked ? 'block' : 'none';
                
                aiResults.scrollIntoView({ behavior: 'smooth' });
            };
        });
    }

    if (heatmapToggle) {
        heatmapToggle.addEventListener('change', () => {
            heatmapCanvas.style.display = heatmapToggle.checked ? 'block' : 'none';
        });
    }

    async function preprocess(img) {
        const canvas = document.createElement('canvas');
        canvas.width = 224;
        canvas.height = 224;
        const ctx = canvas.getContext('2d');
        ctx.drawImage(img, 0, 0, 224, 224);
        
        const imageData = ctx.getImageData(0, 0, 224, 224).data;
        const redArr = [], greenArr = [], blueArr = [];

        for (let i = 0; i < imageData.length; i += 4) {
            redArr.push((imageData[i] / 255 - 0.485) / 0.229);
            greenArr.push((imageData[i+1] / 255 - 0.456) / 0.224);
            blueArr.push((imageData[i+2] / 255 - 0.406) / 0.225);
        }

        const input = new Float32Array([...redArr, ...greenArr, ...blueArr]);
        return new ort.Tensor('float32', input, [1, 3, 224, 224]);
    }

    async function runInference(inputTensor) {
        const feeds = { input: inputTensor };
        const results = await session.run(feeds);
        
        const prediction = results.prediction.data;
        const heatmap = results.heatmap.data; // 1x1x7x7
        
        const probs = Array.from(prediction).map(v => 1 / (1 + Math.exp(-v)));
        return { probs, heatmap };
    }

    function renderHeatmap(data, probItems) {
        heatmapCanvas.width = 224;
        heatmapCanvas.height = 224;
        const ctx = heatmapCanvas.getContext('2d');
        ctx.clearRect(0, 0, 224, 224);

        if (!heatmapToggle.checked) return;

        // 1. Draw soft heatmap overlay
        const size = 7;
        const max = Math.max(...data);
        const range = max || 1;
        const offCanvas = document.createElement('canvas');
        offCanvas.width = size;
        offCanvas.height = size;
        const offCtx = offCanvas.getContext('2d');
        const offData = offCtx.createImageData(size, size);

        for (let i = 0; i < data.length; i++) {
            const val = data[i] / range;
            const idx = i * 4;
            offData.data[idx] = 255; offData.data[idx + 1] = 255; offData.data[idx + 2] = 0; // Yellow-ish
            offData.data[idx + 3] = val * 100; 
        }
        offCtx.putImageData(offData, 0, 0);
        ctx.imageSmoothingEnabled = true;
        ctx.drawImage(offCanvas, 0, 0, size, size, 0, 0, 224, 224);

        // 2. Draw Bounding Boxes for top findings
        probItems.forEach((item, index) => {
            const bbox = calculateBBox(data, 0.6); // Threshold 60% of max
            if (bbox) {
                const [x1, y1, x2, y2] = bbox.map(v => (v * 224) / 7);
                const w = (x2 - x1) || 30;
                const h = (y2 - y1) || 30;

                ctx.strokeStyle = '#ffff00';
                ctx.lineWidth = 2;
                ctx.strokeRect(x1, y1, w, h);

                // Draw Label Tag
                ctx.fillStyle = '#ffff00';
                const labelText = `${index + 1} - ${item.label}`;
                ctx.font = 'bold 10px Inter';
                const textWidth = ctx.measureText(labelText).width;
                ctx.fillRect(x1, y1 - 15, textWidth + 10, 15);
                
                ctx.fillStyle = '#000';
                ctx.fillText(labelText, x1 + 5, y1 - 4);
            }
        });
    }

    function calculateBBox(data, thresh) {
        const size = 7;
        const max = Math.max(...data);
        const limit = max * thresh;
        
        let minR = size, maxR = -1, minC = size, maxC = -1;
        let found = false;

        for (let r = 0; r < size; r++) {
            for (let c = 0; c < size; c++) {
                if (data[r * size + c] >= limit) {
                    minR = Math.min(minR, r); maxR = Math.max(maxR, r);
                    minC = Math.min(minC, c); maxC = Math.max(maxC, c);
                    found = true;
                }
            }
        }
        // Expand box slightly for better visual
        if (found) return [Math.max(0, minC-0.5), Math.max(0, minR-0.5), Math.min(7, maxC+1.5), Math.min(7, maxR+1.5)];
        return null;
    }

    function displayResults(probs, heatmap) {
        findingBars.innerHTML = '';
        const sorted = probs
            .map((p, i) => ({ label: LABELS[i], prob: p }))
            .sort((a, b) => b.prob - a.prob);

        const filtered = sorted.filter(item => item.prob > 0.25).slice(0, 3);
        
        // Trigger heatmap and BBox rendering
        renderHeatmap(heatmap, filtered);

        if (filtered.length > 0) {
            filtered.forEach(item => {
                const percentage = (item.prob * 100).toFixed(1);
                const color = item.prob > 0.5 ? '#ff4b4b' : '#ff9f43';
                
                const row = document.createElement('div');
                row.className = 'finding-row';
                row.innerHTML = `
                    <div class="finding-label">
                        <span>${item.label}</span>
                        <span>${percentage}%</span>
                    </div>
                    <div class="progress-bg">
                        <div class="progress-fill" style="width: 0%; background: ${color};"></div>
                    </div>
                `;
                findingBars.appendChild(row);
                
                // Animate progress bar
                setTimeout(() => {
                    const fill = row.querySelector('.progress-fill');
                    if (fill) fill.style.width = `${percentage}%`;
                }, 100);
            });
        } else {
            findingBars.innerHTML = `
                <div style="text-align: center; padding: 2rem; background: rgba(0, 200, 83, 0.1); border-radius: 12px; border: 1px solid rgba(0, 200, 83, 0.2);">
                    <div style="color: #00c853; font-size: 1.5rem; margin-bottom: 0.5rem;">✔ No Significant Pathologies</div>
                    <div style="color: var(--text-secondary); font-size: 0.9rem;">Model confidence is below diagnostic threshold (< 25%). Case appears healthy.</div>
                </div>
            `;
        }
    }
});

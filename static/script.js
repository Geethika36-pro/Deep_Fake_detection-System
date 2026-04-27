document.addEventListener('DOMContentLoaded', () => {
    // Initialize Particles.js background
    particlesJS('particles-js', {
        particles: {
            number: { value: 60, density: { enable: true, value_area: 800 } },
            color: { value: '#58a6ff' },
            shape: { type: 'circle' },
            opacity: { value: 0.3, random: false },
            size: { value: 3, random: true },
            line_linked: { enable: true, distance: 150, color: '#58a6ff', opacity: 0.2, width: 1 },
            move: { enable: true, speed: 1.5, direction: 'none', random: true, out_mode: 'out' }
        },
        interactivity: {
            detect_on: 'canvas',
            events: { onhover: { enable: true, mode: 'grab' }, resize: true },
            modes: { grab: { distance: 140, line_linked: { opacity: 0.5 } } }
        },
        retina_detect: true
    });

    // Elements
    const uploadArea = document.getElementById('uploadArea');
    const videoInput = document.getElementById('videoInput');
    const browseBtn = document.getElementById('browseBtn');
    const cameraBtn = document.getElementById('cameraBtn');

    // Camera Elements
    const cameraArea = document.getElementById('cameraArea');
    const cameraPreview = document.getElementById('cameraPreview');
    const startRecordBtn = document.getElementById('startRecordBtn');
    const stopRecordBtn = document.getElementById('stopRecordBtn');
    const cancelCameraBtn = document.getElementById('cancelCameraBtn');
    const recordingIndicator = document.getElementById('recordingIndicator');
    const recordingTime = document.getElementById('recordingTime');

    // App state components
    const progressArea = document.getElementById('progressArea');
    const resultArea = document.getElementById('resultArea');
    const resetBtn = document.getElementById('resetBtn');
    const fileInfo = document.getElementById('fileInfo');

    // UI Elements for result
    const resultCard = document.getElementById('resultCard');
    const resultIcon = document.getElementById('resultIcon');
    const resultTitle = document.getElementById('resultTitle');
    const confidenceFill = document.getElementById('confidenceFill');
    const confidenceValue = document.getElementById('confidenceValue');
    const totalFrames = document.getElementById('totalFrames');
    const realFrames = document.getElementById('realFrames');
    const fakeFrames = document.getElementById('fakeFrames');

    // Variables for recording
    let mediaRecorder;
    let recordedChunks = [];
    let stream;
    let timerInterval;
    let secondsRecorded = 0;

    /* --- UPLOAD LOGIC --- */

    browseBtn.addEventListener('click', (e) => {
        e.stopPropagation();
        videoInput.click();
    });

    uploadArea.addEventListener('click', (e) => {
        if (e.target === cameraBtn || cameraBtn.contains(e.target)) return;
        videoInput.click();
    });

    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
        uploadArea.addEventListener(eventName, preventDefaults, false);
    });

    function preventDefaults(e) {
        e.preventDefault();
        e.stopPropagation();
    }

    ['dragenter', 'dragover'].forEach(eventName => {
        uploadArea.addEventListener(eventName, () => {
            uploadArea.classList.add('dragover');
        }, false);
    });

    ['dragleave', 'drop'].forEach(eventName => {
        uploadArea.addEventListener(eventName, () => {
            uploadArea.classList.remove('dragover');
        }, false);
    });

    uploadArea.addEventListener('drop', (e) => {
        const dt = e.dataTransfer;
        const files = dt.files;
        if (files.length > 0) {
            handleFile(files[0]);
        }
    });

    videoInput.addEventListener('change', function () {
        if (this.files.length > 0) {
            handleFile(this.files[0]);
        }
    });

    /* --- CAMERA LOGIC --- */

    cameraBtn.addEventListener('click', async (e) => {
        e.stopPropagation();

        if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
            alert("Your browser does not support camera access, or it is blocked by security settings (try using HTTPS or localhost).");
            return;
        }

        try {
            stream = await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
            cameraPreview.srcObject = stream;

            uploadArea.classList.add('hidden');
            cameraArea.classList.remove('hidden');
        } catch (err) {
            alert("Camera access denied or no camera found!\n\nReason: " + (err.message || err.name) + "\n\nPlease ensure no other app (like Zoom) is using it, and that you clicked 'Allow' in your browser.");
            console.error("Camera error:", err);
        }
    });

    cancelCameraBtn.addEventListener('click', () => {
        stopCameraStreams();
        resetCameraUI();
        cameraArea.classList.add('hidden');
        uploadArea.classList.remove('hidden');
    });

    startRecordBtn.addEventListener('click', () => {
        recordedChunks = [];
        const mimeType = MediaRecorder.isTypeSupported('video/webm; codecs=vp9') ? 'video/webm; codecs=vp9' :
            (MediaRecorder.isTypeSupported('video/webm') ? 'video/webm' : '');
        try {
            mediaRecorder = new MediaRecorder(stream, mimeType ? { mimeType } : {});
        } catch (e) {
            mediaRecorder = new MediaRecorder(stream);
        }

        mediaRecorder.ondataavailable = function (e) {
            if (e.data.size > 0) {
                recordedChunks.push(e.data);
            }
        };

        mediaRecorder.onstop = function () {
            const blob = new Blob(recordedChunks, { type: 'video/mp4' }); // Saving as mp4 structure blob
            const file = new File([blob], "camera_capture.mp4", { type: 'video/mp4' });

            cameraArea.classList.add('hidden');
            stopCameraStreams();
            resetCameraUI();

            handleFile(file); // trigger AI analysis
        };

        mediaRecorder.start();

        // UI transitions
        startRecordBtn.classList.add('hidden');
        stopRecordBtn.classList.remove('hidden');
        cancelCameraBtn.classList.add('hidden');
        recordingIndicator.classList.remove('hidden');

        // Timer logic
        secondsRecorded = 0;
        recordingTime.textContent = '00:00';
        timerInterval = setInterval(() => {
            secondsRecorded++;
            const m = String(Math.floor(secondsRecorded / 60)).padStart(2, '0');
            const s = String(secondsRecorded % 60).padStart(2, '0');
            recordingTime.textContent = `${m}:${s}`;
        }, 1000);
    });

    stopRecordBtn.addEventListener('click', () => {
        if (mediaRecorder && mediaRecorder.state !== 'inactive') {
            mediaRecorder.stop();
        }
    });

    function stopCameraStreams() {
        if (stream) {
            stream.getTracks().forEach(track => track.stop());
        }
    }

    function resetCameraUI() {
        startRecordBtn.classList.remove('hidden');
        stopRecordBtn.classList.add('hidden');
        cancelCameraBtn.classList.remove('hidden');
        recordingIndicator.classList.add('hidden');
        clearInterval(timerInterval);
    }


    /* --- AI API SUBMISSION --- */

    function handleFile(file) {
        if (!file.type.startsWith('video/') && !file.type.startsWith('image/')) {
            fileInfo.innerHTML = '<span style="color:var(--fake-color)">Please upload a valid image or video file.</span>';
            return;
        }

        uploadArea.classList.add('hidden');
        cameraArea.classList.add('hidden');
        progressArea.classList.remove('hidden');

        const formData = new FormData();
        formData.append('file', file);

        fetch('/api/predict', {
            method: 'POST',
            body: formData
        })
            .then(response => response.json())
            .then(data => {
                if (data.error) {
                    showError(data.error);
                    return;
                }
                showResult(data);
            })
            .catch(err => {
                showError("Network error or server is down. Please try again.");
                console.error(err);
            });
    }

    function showResult(data) {
        progressArea.classList.add('hidden');
        resultArea.classList.remove('hidden');

        resultCard.classList.remove('is-fake', 'is-real');
        resultIcon.className = 'fa-solid result-icon';

        totalFrames.textContent = data.total_frames;
        realFrames.textContent = data.real_frames;
        fakeFrames.textContent = data.fake_frames;
        confidenceValue.textContent = `${data.confidence.toFixed(1)}%`;
        console.log("Raw Model Probability Scores:", data.raw_scores);

        if (data.decision === 'FAKE') {
            resultCard.classList.add('is-fake');
            resultIcon.classList.add('fa-triangle-exclamation');
            resultTitle.textContent = 'DEEPFAKE DETECTED';
        } else {
            resultCard.classList.add('is-real');
            resultIcon.classList.add('fa-circle-check');
            resultTitle.textContent = 'AUTHENTIC';
        }

        setTimeout(() => {
            confidenceFill.style.width = `${data.confidence}%`;
        }, 100);
    }

    function showError(message) {
        progressArea.classList.add('hidden');
        uploadArea.classList.remove('hidden');
        fileInfo.innerHTML = `<span style="color:var(--fake-color)">Error: ${message}</span>`;
        if (videoInput) videoInput.value = '';
    }

    resetBtn.addEventListener('click', () => {
        resultArea.classList.add('hidden');
        uploadArea.classList.remove('hidden');
        confidenceFill.style.width = '0%';
        if (videoInput) videoInput.value = '';
        fileInfo.textContent = 'Supported formats: JPG, PNG, MP4, AVI, MOV';
    });
});

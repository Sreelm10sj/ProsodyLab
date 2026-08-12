const API_URL = 'http://127.0.0.1:8000';

export async function checkHealth() {
    try {
        const res = await fetch(`${API_URL}/api/health`);
        return await res.json();
    } catch (e) {
        throw new Error('Backend is not reachable');
    }
}

export async function transcribeAudio(audioBlob, filename) {
    const formData = new FormData();
    formData.append('file', audioBlob, filename || 'audio.webm');

    const response = await fetch(`${API_URL}/api/transcribe`, {
        method: 'POST',
        body: formData,
    });

    if (!response.ok) {
        let errorMsg = 'Inference failed on the backend';
        try {
            const errorData = await response.json();
            errorMsg = errorData.detail || errorMsg;
        } catch(e) {
            // response is not json
        }
        throw new Error(errorMsg);
    }

    return await response.json();
}

export async function generateAnnotation(audioBlob, filename) {
    const formData = new FormData();
    formData.append('file', audioBlob, filename || 'audio.webm');

    const response = await fetch(`${API_URL}/api/annotate`, {
        method: 'POST',
        body: formData,
    });

    if (!response.ok) {
        let errorMsg = 'Annotation failed on the backend';
        try {
            const errorData = await response.json();
            errorMsg = errorData.detail || errorMsg;
        } catch(e) {
            // response is not json
        }
        throw new Error(errorMsg);
    }

    return await response.json();
}

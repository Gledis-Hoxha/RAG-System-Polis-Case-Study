import axios from 'axios';

// Configure the base URL for your Streamlit/FastAPI backend
// For development, it's typically http://localhost:8501
const API_BASE_URL = 'http://localhost:8501';

// Function to process PDF upload
export const processPDF = async (file, apiKey, chunkSize) => {
  const formData = new FormData();
  formData.append('file', file);
  formData.append('api_key', apiKey);
  formData.append('chunk_size', chunkSize);

  try {
    const response = await axios.post(`${API_BASE_URL}/process_pdf`, formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    });
    return response.data;
  } catch (error) {
    console.error('Error processing PDF:', error);
    throw error;
  }
};

// Function to ask questions
export const askQuestion = async (question, apiKey, extraContext = '') => {
  try {
    const response = await axios.post(`${API_BASE_URL}/ask`, {
      question,
      api_key: apiKey,
      extra_context: extraContext,
    });
    return response.data;
  } catch (error) {
    console.error('Error asking question:', error);
    throw error;
  }
};
import { useState } from 'react';
import { processPDF, askQuestion } from './api';
import FileUpload from './components/FileUpload';
import QuestionForm from './components/QuestionForm';
import ResultsDisplay from './components/ResultsDisplay';
import SettingsPanel from './components/SettingsPanel';
import './App.css';

function App() {
  const [apiKey, setApiKey] = useState('');
  const [chunkSize, setChunkSize] = useState(1000);
  const [extraContext, setExtraContext] = useState('');
  const [isProcessing, setIsProcessing] = useState(false);
  const [answer, setAnswer] = useState('');
  const [contextChunks, setContextChunks] = useState([]);
  const [fileName, setFileName] = useState('');
  const [statusMessage, setStatusMessage] = useState('');

  const handleFileUpload = async (file) => {
    setIsProcessing(true);
    setFileName(file.name);
    setStatusMessage('Processing PDF...');
    
    try {
      const result = await processPDF(file, apiKey, chunkSize);
      if (result.status === 'success') {
        setStatusMessage(result.message);
      } else {
        setStatusMessage(`Error: ${result.message}`);
      }
    } catch (error) {
      setStatusMessage(`Error: ${error.message}`);
    } finally {
      setIsProcessing(false);
    }
  };

  const handleQuestionSubmit = async (question) => {
    if (!fileName) {
      setStatusMessage('Please upload a PDF first');
      return;
    }

    setIsProcessing(true);
    setStatusMessage('Generating answer...');
    
    try {
      const response = await askQuestion(question, apiKey, extraContext);
      if (response.status === 'success') {
        setAnswer(response.answer);
        setContextChunks(response.context_chunks || []);
        setStatusMessage('');
      } else {
        setStatusMessage(`Error: ${response.message}`);
      }
    } catch (error) {
      setStatusMessage(`Error: ${error.message}`);
      setAnswer("Sorry, I couldn't process your question.");
    } finally {
      setIsProcessing(false);
    }
  };

  return (
    <div className="app">
      <header>
        <h1>Document Q&A System</h1>
        <p>Upload a PDF and ask questions about its content</p>
      </header>
      
      <main>
        <SettingsPanel
          apiKey={apiKey}
          setApiKey={setApiKey}
          chunkSize={chunkSize}
          setChunkSize={setChunkSize}
          extraContext={extraContext}
          setExtraContext={setExtraContext}
        />
        
        <section className="upload-section">
          <h2>Upload Document</h2>
          <FileUpload 
            onFileUpload={handleFileUpload} 
            isProcessing={isProcessing}
          />
          {fileName && <p className="file-info">Current file: {fileName}</p>}
          {statusMessage && !isProcessing && <p className="status-message">{statusMessage}</p>}
        </section>
        
        <section className="question-section">
          <h2>Ask Questions</h2>
          <QuestionForm 
            onSubmit={handleQuestionSubmit} 
            isProcessing={isProcessing}
            isFileUploaded={!!fileName}
          />
        </section>
        
        <ResultsDisplay 
          answer={answer}
          contextChunks={contextChunks}
          extraContext={extraContext}
        />
      </main>
      
      <footer>
        <p>RAG-based Document Q&A System</p>
      </footer>
    </div>
  );
}

export default App;
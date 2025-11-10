import React, { useState } from 'react';
import './App.css';

function App() {
  const [image, setImage] = useState(null);
  const [result, setResult] = useState('');

  // Image upload handler
  const handleImageChange = (e) => {
    if (e.target.files && e.target.files[0]) {
      setImage(e.target.files[0]);
    }
  };

  // Submit image to API
  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!image) return;
    const formData = new FormData();
    formData.append('file', image);
    const response = await fetch('http://localhost:5000/predict', { // Flask backend
      method: 'POST',
      body: formData,
    });
    const data = await response.json();
    setResult(data.behaviour || 'No behaviour detected');
  };

  return (
    <div className="container">
      <header>
        <h2>Handwriting Behaviour Detection</h2>
      </header>
      <form onSubmit={handleSubmit} className="upload-form">
        <input type="file" accept="image/*" onChange={handleImageChange} />
        <button type="submit">Predict Behaviour</button>
      </form>
      <div className="output">
        {result && <h3>Predicted Behaviour: <span className='result'>{result}</span></h3>}
      </div>
    </div>
  );
}

export default App;
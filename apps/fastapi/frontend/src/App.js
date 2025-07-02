import React, { useState } from "react";
import UploadForm from "./components/UploadForm";
import ResultsCard from "./components/ResultsCard";
import Container from "react-bootstrap/Container";

function App() {
  const [results, setResults] = useState(null);

  return (
    <Container className="mt-5">
      <h2 className="text-center mb-4">🏥 Patient Readmission Predictor</h2>
      <UploadForm setResults={setResults} />
      {results && <ResultsCard results={results} />}
    </Container>
  );
}

export default App;

import React, { useState } from "react";
import axios from "axios";
import { Form, Button, Spinner } from "react-bootstrap";

function UploadForm({ setResults }) {
  const [file, setFile] = useState(null);
  const [loading, setLoading] = useState(false);

  const uploadFile = async (e) => {
    e.preventDefault();
    if (!file) return;

    const formData = new FormData();
    formData.append("file", file);
    setLoading(true);

    try {
      const res = await axios.post("http://localhost:5000/predict", formData);
      setResults(res.data);
    } catch (err) {
      console.error("Prediction failed", err);
    } finally {
      setLoading(false);
    }
  };

  return (
    <Form onSubmit={uploadFile} className="mb-4">
      <Form.Group className="mb-3">
        <Form.Label>Select CSV File</Form.Label>
        <Form.Control
          type="file"
          accept=".csv"
          onChange={(e) => setFile(e.target.files[0])}
        />
      </Form.Group>
      <Button type="submit" variant="primary" disabled={loading}>
        {loading ? <Spinner animation="border" size="sm" /> : "Upload & Predict"}
      </Button>
    </Form>
  );
}

export default UploadForm;

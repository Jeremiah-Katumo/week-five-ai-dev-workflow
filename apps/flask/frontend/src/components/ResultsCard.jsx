import React from "react";
import Card from "react-bootstrap/Card";
import ListGroup from "react-bootstrap/ListGroup";

function ResultsCard({ results }) {
  const { prediction_summary, important_features } = results;

  return (
    <Card className="shadow">
      <Card.Body>
        <Card.Title>Prediction Summary</Card.Title>
        <ListGroup variant="flush">
          <ListGroup.Item>Total Patients: {prediction_summary.total}</ListGroup.Item>
          <ListGroup.Item>Readmissions Predicted: {prediction_summary.positive}</ListGroup.Item>
        </ListGroup>

        <Card.Title className="mt-4">Top Features (SHAP)</Card.Title>
        <ListGroup variant="flush">
          {Object.entries(important_features).map(([key, value]) => (
            <ListGroup.Item key={key}>
              {key}: {value.toFixed(3)}
            </ListGroup.Item>
          ))}
        </ListGroup>
      </Card.Body>
    </Card>
  );
}

export default ResultsCard;

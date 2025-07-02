import React from "react";
import Card from "react-bootstrap/Card";
import ListGroup from "react-bootstrap/ListGroup";

function ResultsCard({ results }) {
  const { prediction_summary, important_features } = results;

  return (
    <Card className="mt-4 shadow">
      <Card.Body>
        <Card.Title>Prediction Summary</Card.Title>
        <ListGroup variant="flush">
          <ListGroup.Item>
            📊 Total Patients: {prediction_summary.total}
          </ListGroup.Item>
          <ListGroup.Item>
            🚨 Predicted Readmissions: {prediction_summary.positive}
          </ListGroup.Item>
        </ListGroup>

        <Card.Title className="mt-4">🔍 Top Influential Features</Card.Title>
        <ListGroup variant="flush">
          {Object.entries(important_features).map(([feat, val]) => (
            <ListGroup.Item key={feat}>
              {feat}: {val.toFixed(3)}
            </ListGroup.Item>
          ))}
        </ListGroup>
      </Card.Body>
    </Card>
  );
}

export default ResultsCard;

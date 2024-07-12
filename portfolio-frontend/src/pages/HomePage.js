// src/pages/HomePage.js
import React from 'react';
import { Card, Container, Row, Col } from 'react-bootstrap';
import RequestForm from '../components/RequestForm';
import RequestList from '../components/RequestList';
import './HomePage.css';

const HomePage = ({ addRequest, requests, deleteRequest }) => {
    return (
        <Container className="mt-5">
            <Row className="mb-4">
                <Col>
                    <Card className="custom-card">
                        <Card.Header className="custom-card-header">Request Form</Card.Header>
                        <Card.Body>
                            <RequestForm addRequest={addRequest} />
                        </Card.Body>
                    </Card>
                </Col>
            </Row>
            <Row>
                <Col>
                    <Card className="custom-card">
                        <Card.Header className="custom-card-header">Request List</Card.Header>
                        <Card.Body>
                            <RequestList requests={requests} deleteRequest={deleteRequest} />
                        </Card.Body>
                    </Card>
                </Col>
            </Row>
        </Container>
    );
};

export default HomePage;

// src/components/RequestList.js
import React from 'react';
import { Table, Button } from 'react-bootstrap';
import { FaTrashAlt } from 'react-icons/fa';
import './RequestList.css';

const RequestList = ({ requests, deleteRequest }) => {
    return (
        <Table striped bordered hover className="custom-table">
            <thead>
            <tr>
                <th>Assets</th>
                <th>Interval</th>
                <th>Look Back Period</th>
                <th>Investment Amount</th>
                <th>Action</th>
            </tr>
            </thead>
            <tbody>
            {requests.map((request, index) => (
                <tr key={request.id}>
                    <td>{request.assets.join(', ')}</td>
                    <td>{request.interval}</td>
                    <td>{request.look_back_period}</td>
                    <td>{request.investment_amount}</td>
                    <td>
                        <Button variant="danger" onClick={() => deleteRequest(request.id)}>
                            <FaTrashAlt />
                        </Button>
                    </td>
                </tr>
            ))}
            </tbody>
        </Table>
    );
};

export default RequestList;

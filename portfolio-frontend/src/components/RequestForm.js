// src/components/RequestForm.js
import React, { useState, useEffect, useRef } from 'react';
import { Form, Button, Row, Col } from 'react-bootstrap';
import $ from 'jquery';
import 'select2/dist/css/select2.min.css';
import 'select2';
import './RequestForm.css';
import './Select2Custom.css';

const RequestForm = ({ addRequest }) => {
    const [formData, setFormData] = useState({
        assets: [],
        interval: '',
        look_back_period: '',
        investment_amount: '',
    });

    const selectRef = useRef(null);
    const cryptoAssets = [
        { symbol: 'BTC', name: 'Bitcoin' },
        { symbol: 'ETH', name: 'Ethereum' },
        { symbol: 'XRP', name: 'Ripple' },
        { symbol: 'LTC', name: 'Litecoin' },
        { symbol: 'BCH', name: 'Bitcoin Cash' }
    ];

    useEffect(() => {
        $(selectRef.current).select2({
            placeholder: 'Select assets',
            width: '100%',
        });

        $(selectRef.current).on('change', function () {
            const selectedAssets = $(this).val();
            setFormData((prevData) => ({
                ...prevData,
                assets: selectedAssets,
            }));
        });
    }, []);

    const handleChange = (e) => {
        const { name, value } = e.target;
        setFormData({
            ...formData,
            [name]: value,
        });
    };

    const handleSubmit = (e) => {
        e.preventDefault();
        addRequest(formData);
        setFormData({
            assets: [],
            interval: '',
            look_back_period: '',
            investment_amount: '',
        });
        $(selectRef.current).val(null).trigger('change'); // Reset select2
    };

    return (
        <Form onSubmit={handleSubmit} className="custom-form">
            <Form.Group className="mb-3" controlId="assets">
                <Form.Label>Assets</Form.Label>
                <select
                    ref={selectRef}
                    className="form-control"
                    multiple
                    name="assets"
                >
                    {cryptoAssets.map((asset) => (
                        <option key={asset.symbol} value={asset.symbol}>{asset.name}</option>
                    ))}
                </select>
            </Form.Group>
            <Row>
                <Col>
                    <Form.Group className="mb-3" controlId="interval">
                        <Form.Label>Interval</Form.Label>
                        <Form.Control
                            type="text"
                            name="interval"
                            value={formData.interval}
                            onChange={handleChange}
                            pattern="(\d+)([smhdMy])"
                            required
                        />
                    </Form.Group>
                </Col>
                <Col>
                    <Form.Group className="mb-3" controlId="look_back_period">
                        <Form.Label>Look Back Period</Form.Label>
                        <Form.Control
                            type="text"
                            name="look_back_period"
                            value={formData.look_back_period}
                            onChange={handleChange}
                            pattern="(\d+)([smhdMy])"
                            required
                        />
                    </Form.Group>
                </Col>
            </Row>
            <Form.Group className="mb-3" controlId="investment_amount">
                <Form.Label>Investment Amount</Form.Label>
                <Form.Control
                    type="number"
                    name="investment_amount"
                    value={formData.investment_amount}
                    onChange={handleChange}
                    required
                />
            </Form.Group>
            <Button type="submit" variant="success">Submit</Button>
        </Form>
    );
};

export default RequestForm;
// src/services/api.js
import axios from 'axios';

const api = axios.create({
    baseURL: 'http://127.0.0.1:5000', // Updated to use 127.0.0.1
});

export const getRequests = async () => {
    const response = await api.get('/requests');
    return response.data;
};

export const createRequest = async (request) => {
    const response = await api.post('/requests', request);
    return response.data;
};

export const deleteRequest = async (id) => {
    const response = await api.delete(`/requests/${id}`);
    return response.data;
};
